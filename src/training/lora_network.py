"""
LoRA Network implementation for Z-Image-Turbo training.

This module provides:
- LoRA linear layer replacement (LoRAModule)
- Full LoRA network management (LoRANetwork)
- Weight extraction and saving in ComfyUI-compatible format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
import math
import re


class LoRAModule(nn.Module):
    """
    A LoRA (Low-Rank Adaptation) module that wraps a linear layer.
    
    Implements W' = W + alpha/rank * (B @ A) where:
    - W is the frozen original weight
    - A is the down-projection (in_features -> rank)
    - B is the up-projection (rank -> out_features)
    - alpha controls the scaling
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_dora: bool = False,
    ):
        """
        Initialize LoRA module.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank
            alpha: LoRA alpha (scaling factor)
            dropout: Dropout probability
            use_dora: Use DoRA (Weight-Decomposed LoRA)
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.use_dora = use_dora
        
        # LoRA matrices
        # A: (rank, in_features) - down projection
        # B: (out_features, rank) - up projection
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # DoRA: decompose weight into magnitude and direction
        if use_dora:
            self.dora_magnitude = nn.Parameter(torch.ones(out_features))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA weights."""
        # Kaiming uniform for A, zeros for B (starts with no effect)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.
        
        Args:
            x: Input tensor
            base_weight: Original frozen weight matrix
            
        Returns:
            Output tensor
        """
        # Base forward: x @ W^T
        base_output = F.linear(x, base_weight)
        
        # LoRA forward: x @ A^T @ B^T
        lora_output = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        
        if self.use_dora:
            # DoRA: normalize and apply learned magnitude
            combined = base_weight + self.scaling * (self.lora_B @ self.lora_A)
            combined_norm = combined / combined.norm(dim=1, keepdim=True)
            return F.linear(x, combined_norm * self.dora_magnitude.view(-1, 1))
        else:
            return base_output + self.scaling * lora_output
    
    def get_delta(self) -> torch.Tensor:
        """Get the weight delta (B @ A) scaled by alpha/rank."""
        return self.scaling * (self.lora_B @ self.lora_A)


class LoRALinear(nn.Module):
    """
    A linear layer with integrated LoRA.
    
    Replaces a standard nn.Linear layer, keeping the base weights frozen
    and training only the LoRA parameters.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_dora: bool = False,
    ):
        """
        Initialize LoRA linear layer.
        
        Args:
            base_layer: Original nn.Linear layer
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: Dropout probability
            use_dora: Use DoRA
        """
        super().__init__()
        
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.has_bias = base_layer.bias is not None
        
        # Store frozen base weights
        self.base_weight = nn.Parameter(base_layer.weight.data.clone(), requires_grad=False)
        if self.has_bias:
            self.base_bias = nn.Parameter(base_layer.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("base_bias", None)
        
        # Create LoRA module
        self.lora = LoRAModule(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            use_dora=use_dora,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA linear layer."""
        # Apply base + LoRA
        output = self.lora(x, self.base_weight)
        
        if self.base_bias is not None:
            output = output + self.base_bias
        
        return output
    
    def get_merged_weight(self) -> torch.Tensor:
        """Get the merged weight (base + LoRA delta)."""
        return self.base_weight + self.lora.get_delta()


class LoRANetwork(nn.Module):
    """
    Manages LoRA adaptation for an entire model.
    
    Provides methods to:
    - Apply LoRA to specific layers
    - Extract LoRA weights
    - Save/load LoRA weights in ComfyUI format
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_dora: bool = False,
    ):
        """
        Initialize LoRA network.
        
        Args:
            model: The base model to add LoRA to
            target_modules: List of module name patterns to target
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: Dropout probability
            use_dora: Use DoRA
        """
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.use_dora = use_dora
        self.target_modules = target_modules
        
        # Track which layers have LoRA
        self.lora_layers: Dict[str, LoRAModule] = nn.ModuleDict()
        self.original_layers: Dict[str, nn.Module] = {}
        
        # Apply LoRA to target layers
        self._apply_lora_to_model(model)
    
    def _should_target(self, name: str) -> bool:
        """Check if a layer should have LoRA applied."""
        for pattern in self.target_modules:
            if pattern in name:
                return True
        return False
    
    def _apply_lora_to_model(self, model: nn.Module):
        """Apply LoRA to all target layers in the model."""
        # Find all linear layers that match target patterns
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self._should_target(name):
                # Create LoRA module for this layer
                lora_key = name.replace(".", "_")
                
                lora_module = LoRAModule(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    use_dora=self.use_dora,
                )
                
                self.lora_layers[lora_key] = lora_module
                
                # Store original layer reference
                self.original_layers[lora_key] = module
                
                # Freeze original weights
                for param in module.parameters():
                    param.requires_grad = False
                
                # Create forward hook to inject LoRA
                self._create_hook(module, lora_module)
    
    def _create_hook(self, layer: nn.Linear, lora: LoRAModule):
        """Create a forward hook to inject LoRA into a layer."""
        original_forward = layer.forward
        
        def hooked_forward(x):
            # Base forward
            base_output = F.linear(x, layer.weight, layer.bias)
            # LoRA forward
            lora_output = F.linear(F.linear(lora.dropout(x), lora.lora_A), lora.lora_B)
            return base_output + lora.scaling * lora_output
        
        layer.forward = hooked_forward
    
    def get_lora_state_dict(self, prefix: str = "diffusion_model.") -> Dict[str, torch.Tensor]:
        """
        Get LoRA weights in ComfyUI-compatible format.
        
        Args:
            prefix: Prefix for keys (ComfyUI uses "diffusion_model.")
            
        Returns:
            State dict with LoRA weights
        """
        state_dict = {}
        
        for lora_key, lora_module in self.lora_layers.items():
            # Convert key back to dot notation
            model_key = lora_key.replace("_", ".")
            
            # Add prefix
            full_key = prefix + model_key
            
            # Save A and B matrices
            state_dict[f"{full_key}.lora_A.weight"] = lora_module.lora_A.data.clone()
            state_dict[f"{full_key}.lora_B.weight"] = lora_module.lora_B.data.clone()
            
            # Save DoRA magnitude if used
            if self.use_dora:
                state_dict[f"{full_key}.dora_magnitude"] = lora_module.dora_magnitude.data.clone()
        
        return state_dict
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor], prefix: str = "diffusion_model."):
        """
        Load LoRA weights from state dict.
        
        Args:
            state_dict: State dict with LoRA weights
            prefix: Prefix used in the state dict keys
        """
        for key, value in state_dict.items():
            # Remove prefix
            if key.startswith(prefix):
                key = key[len(prefix):]
            
            # Extract base key and component
            if ".lora_A.weight" in key:
                base_key = key.replace(".lora_A.weight", "")
                lora_key = base_key.replace(".", "_")
                if lora_key in self.lora_layers:
                    self.lora_layers[lora_key].lora_A.data.copy_(value)
            elif ".lora_B.weight" in key:
                base_key = key.replace(".lora_B.weight", "")
                lora_key = base_key.replace(".", "_")
                if lora_key in self.lora_layers:
                    self.lora_layers[lora_key].lora_B.data.copy_(value)
            elif ".dora_magnitude" in key and self.use_dora:
                base_key = key.replace(".dora_magnitude", "")
                lora_key = base_key.replace(".", "_")
                if lora_key in self.lora_layers:
                    self.lora_layers[lora_key].dora_magnitude.data.copy_(value)
    
    def save_weights(self, path: str, metadata: Optional[Dict[str, str]] = None):
        """
        Save LoRA weights to safetensors file.
        
        Args:
            path: Output path
            metadata: Optional metadata to include
        """
        from safetensors.torch import save_file
        
        state_dict = self.get_lora_state_dict()
        
        # Convert to float16 for efficiency
        state_dict = {k: v.half() for k, v in state_dict.items()}
        
        # Default metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "lora_rank": str(self.rank),
            "lora_alpha": str(self.alpha),
            "format": "comfyui",
            "model": "z-image-turbo",
        })
        
        save_file(state_dict, path, metadata=metadata)
    
    @classmethod
    def load_weights(cls, path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
        """
        Load LoRA weights from safetensors file.
        
        Args:
            path: Path to safetensors file
            
        Returns:
            Tuple of (state_dict, metadata)
        """
        from safetensors import safe_open
        
        state_dict = {}
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata() or {}
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        return state_dict, metadata
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable LoRA parameters."""
        params = []
        for lora in self.lora_layers.values():
            params.extend([lora.lora_A, lora.lora_B])
            if self.use_dora:
                params.append(lora.dora_magnitude)
        return params
    
    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_params())
    
    def merge_into_model(self, model: nn.Module):
        """
        Merge LoRA weights into the base model weights.
        
        This modifies the model weights in-place.
        
        Args:
            model: Model to merge LoRA into
        """
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    lora_key = name.replace(".", "_")
                    if lora_key in self.lora_layers:
                        lora = self.lora_layers[lora_key]
                        delta = lora.get_delta()
                        module.weight.data.add_(delta)
    
    def extract_from_model_diff(
        self,
        original_model: nn.Module,
        trained_model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract LoRA weights from the difference between models.
        
        Uses SVD to decompose the weight difference into LoRA format.
        
        Args:
            original_model: Original model weights
            trained_model: Trained/fine-tuned model weights
            
        Returns:
            State dict with extracted LoRA weights
        """
        state_dict = {}
        
        with torch.no_grad():
            orig_params = dict(original_model.named_parameters())
            train_params = dict(trained_model.named_parameters())
            
            for name in orig_params:
                if not self._should_target(name):
                    continue
                
                orig_weight = orig_params[name]
                train_weight = train_params[name]
                
                # Compute difference
                diff = train_weight - orig_weight
                
                # SVD decomposition
                U, S, Vh = torch.linalg.svd(diff.float(), full_matrices=False)
                
                # Take top-k singular values
                k = min(self.rank, len(S))
                
                # Reconstruct as A and B
                # A = sqrt(S[:k]) @ Vh[:k]
                # B = U[:, :k] @ sqrt(S[:k])
                sqrt_s = torch.sqrt(S[:k])
                A = (sqrt_s.unsqueeze(1) * Vh[:k]).to(orig_weight.dtype)  # (rank, in)
                B = (U[:, :k] * sqrt_s.unsqueeze(0)).to(orig_weight.dtype)  # (out, rank)
                
                # Apply scaling to match LoRA format
                # Original: delta = B @ A
                # LoRA: delta = scaling * (lora_B @ lora_A)
                # So we need to divide by scaling
                scaling = self.alpha / self.rank
                A = A / math.sqrt(scaling)
                B = B / math.sqrt(scaling)
                
                key_base = f"diffusion_model.{name.replace('.weight', '')}"
                state_dict[f"{key_base}.lora_A.weight"] = A
                state_dict[f"{key_base}.lora_B.weight"] = B
        
        return state_dict


def get_target_modules_for_zimage() -> List[str]:
    """
    Get default target modules for Z-Image-Turbo.
    
    Returns list of module name patterns to apply LoRA to.
    """
    return [
        # Attention projections
        "attn.to_q",
        "attn.to_k", 
        "attn.to_v",
        "attn.to_out",
        # Also target attention in transformer blocks
        "attention.to_q",
        "attention.to_k",
        "attention.to_v",
        "attention.to_out",
        # Feed-forward network
        "ff.0",  # proj_in
        "ff.2",  # proj_out
        "feed_forward.0",
        "feed_forward.2",
        # Cross-attention (if present)
        "cross_attn.to_q",
        "cross_attn.to_k",
        "cross_attn.to_v",
        "cross_attn.to_out",
    ]


def count_lora_layers(model: nn.Module, target_modules: List[str]) -> int:
    """
    Count how many layers would have LoRA applied.
    
    Args:
        model: The model
        target_modules: Target module patterns
        
    Returns:
        Number of layers that would have LoRA
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for pattern in target_modules:
                if pattern in name:
                    count += 1
                    break
    return count
