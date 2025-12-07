"""
Training Adapter Manager for Z-Image-Turbo.

This module handles Ostris's de-distillation training adapter, which helps
preserve turbo model capabilities during fine-tuning.

The adapter works by:
1. Merging IN (+1.0 weight) during training - slows down distillation breakdown
2. Inverting (-1.0 weight) during inference - removes adapter effect while keeping your LoRA

Usage:
    # During training setup
    adapter_manager = TrainingAdapterManager(adapter_path)
    adapter_manager.apply_to_model(model, weight=1.0)  # Merge for training
    
    # During inference with trained LoRA
    # The adapter should be inverted (-1.0) or simply not applied
    # Your LoRA will have learned with the adapter present, so it "knows"
    # how to work with the turbo model
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union
from safetensors import safe_open
from safetensors.torch import save_file


class TrainingAdapterManager:
    """
    Manages the Z-Image-Turbo training adapter (de-distillation LoRA).
    
    The adapter is designed to slow down the breakdown of distillation
    that occurs during fine-tuning. It should be:
    - Merged into the model during training (weight = +1.0)
    - Inverted/removed during inference (weight = -1.0 or not applied)
    """
    
    # Default adapter paths
    ADAPTER_V1 = "models/training_adapters/zimage_turbo_training_adapter_v1.safetensors"
    ADAPTER_V2 = "models/training_adapters/zimage_turbo_training_adapter_v2.safetensors"
    
    def __init__(self, adapter_path: Optional[Union[str, Path]] = None):
        """
        Initialize adapter manager.
        
        Args:
            adapter_path: Path to adapter file. If None, uses v2 by default.
        """
        if adapter_path is None:
            # Default to v2
            adapter_path = Path(__file__).parent.parent.parent / self.ADAPTER_V2
        
        self.adapter_path = Path(adapter_path)
        self.weights: Optional[Dict[str, torch.Tensor]] = None
        self.metadata: Dict[str, str] = {}
        self.is_loaded = False
        self.applied_weight: float = 0.0
        
    def load(self) -> Dict[str, torch.Tensor]:
        """
        Load adapter weights from file.
        
        Returns:
            Dictionary of adapter weights
        """
        if not self.adapter_path.exists():
            raise FileNotFoundError(f"Training adapter not found: {self.adapter_path}")
        
        weights = {}
        with safe_open(str(self.adapter_path), framework="pt") as f:
            self.metadata = dict(f.metadata()) if f.metadata() else {}
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        
        self.weights = weights
        self.is_loaded = True
        
        return weights
    
    def get_info(self) -> Dict:
        """
        Get information about the adapter.
        
        Returns:
            Dictionary with adapter info (rank, num_weights, etc.)
        """
        if not self.is_loaded:
            self.load()
        
        # Parse weights to extract info
        lora_pairs = {}
        for key in self.weights:
            if ".lora_A." in key:
                base = key.replace(".lora_A.weight", "").replace(".lora_A.", "")
                if base not in lora_pairs:
                    lora_pairs[base] = {}
                lora_pairs[base]["A"] = self.weights[key]
            elif ".lora_B." in key:
                base = key.replace(".lora_B.weight", "").replace(".lora_B.", "")
                if base not in lora_pairs:
                    lora_pairs[base] = {}
                lora_pairs[base]["B"] = self.weights[key]
        
        # Detect rank from first A matrix
        rank = None
        for pair in lora_pairs.values():
            if "A" in pair:
                rank = pair["A"].shape[0]
                break
        
        return {
            "path": str(self.adapter_path),
            "num_weights": len(self.weights),
            "num_lora_pairs": len(lora_pairs),
            "rank": rank,
            "metadata": self.metadata,
        }
    
    def _map_key(self, adapter_key: str) -> str:
        """
        Map adapter key to model key format.
        
        The adapter uses 'diffusion_model.' prefix like ComfyUI format.
        """
        # Remove lora suffix to get base layer path
        base_key = adapter_key
        
        # Handle .lora_A.weight and .lora_B.weight
        if ".lora_A.weight" in base_key:
            base_key = base_key.replace(".lora_A.weight", "")
        elif ".lora_B.weight" in base_key:
            base_key = base_key.replace(".lora_B.weight", "")
        elif ".lora_A" in base_key:
            base_key = base_key.replace(".lora_A", "")
        elif ".lora_B" in base_key:
            base_key = base_key.replace(".lora_B", "")
        
        # Remove diffusion_model. prefix (ComfyUI format)
        if base_key.startswith("diffusion_model."):
            base_key = base_key[len("diffusion_model."):]
        
        # Handle Sequential wrappers
        base_key = base_key.replace(".to_out.0", ".to_out")
        base_key = base_key.replace(".adaLN_modulation.0", ".adaLN_modulation")
        
        return base_key
    
    def compute_delta(self, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        Compute the weight delta from LoRA matrices.
        
        delta = scale * (B @ A)
        
        Args:
            lora_a: A matrix (rank, in_features)
            lora_b: B matrix (out_features, rank)
            scale: Scaling factor
            
        Returns:
            Weight delta tensor
        """
        delta = lora_b @ lora_a
        return scale * delta
    
    def apply_to_model(
        self,
        model: nn.Module,
        weight: float = 1.0,
        verbose: bool = False,
    ) -> int:
        """
        Apply the training adapter to a model.
        
        Args:
            model: The model to modify
            weight: Adapter weight (+1.0 for training, -1.0 to invert)
            verbose: Print detailed information
            
        Returns:
            Number of weights modified
        """
        if not self.is_loaded:
            self.load()
        
        # Group weights into (A, B) pairs
        lora_pairs: Dict[str, Dict[str, torch.Tensor]] = {}
        
        for key, tensor in self.weights.items():
            base_key = self._map_key(key)
            
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            
            if ".lora_A" in key or "lora_A" in key:
                lora_pairs[base_key]["A"] = tensor
            elif ".lora_B" in key or "lora_B" in key:
                lora_pairs[base_key]["B"] = tensor
        
        if verbose:
            print(f"Found {len(lora_pairs)} LoRA pairs in adapter")
        
        # Apply to model
        applied_count = 0
        model_params = dict(model.named_parameters())
        
        for layer_path, pair in lora_pairs.items():
            if "A" not in pair or "B" not in pair:
                continue
            
            # Find the weight in model
            weight_key = layer_path + ".weight"
            
            if weight_key not in model_params:
                if verbose:
                    print(f"  Skipping {layer_path}: not found in model")
                continue
            
            param = model_params[weight_key]
            
            # Compute delta
            lora_a = pair["A"].to(param.device, param.dtype)
            lora_b = pair["B"].to(param.device, param.dtype)
            delta = self.compute_delta(lora_a, lora_b, weight)
            
            # Check shape compatibility
            if delta.shape != param.shape:
                if verbose:
                    print(f"  Skipping {layer_path}: shape mismatch {delta.shape} vs {param.shape}")
                continue
            
            # Apply delta
            with torch.no_grad():
                param.add_(delta)
            
            applied_count += 1
            if verbose:
                print(f"  Applied adapter to {layer_path}")
        
        self.applied_weight = weight
        
        if verbose:
            print(f"Applied adapter to {applied_count}/{len(lora_pairs)} layers (weight={weight})")
        
        return applied_count
    
    def remove_from_model(self, model: nn.Module, verbose: bool = False) -> int:
        """
        Remove the training adapter from a model.
        
        This applies the inverse weight to undo the adapter effect.
        
        Args:
            model: The model to modify
            verbose: Print detailed information
            
        Returns:
            Number of weights modified
        """
        if self.applied_weight == 0.0:
            if verbose:
                print("Adapter was not applied, nothing to remove")
            return 0
        
        # Apply inverse
        result = self.apply_to_model(model, weight=-self.applied_weight, verbose=verbose)
        self.applied_weight = 0.0
        return result
    
    @staticmethod
    def download_adapter(
        version: str = "v2",
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Download the official training adapter from HuggingFace.
        
        Args:
            version: Adapter version ("v1" or "v2")
            output_dir: Output directory
            
        Returns:
            Path to downloaded adapter
        """
        from huggingface_hub import hf_hub_download
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "models" / "training_adapters"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        repo_id = "ostris/zimage_turbo_training_adapter"
        
        if version == "v1":
            filename = "zimage_turbo_training_adapter_v1.safetensors"
        else:
            filename = "zimage_turbo_training_adapter_v2.safetensors"
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
        )
        
        return Path(local_path)


class AdapterAwareTraining:
    """
    Helper class for training with the de-distillation adapter.
    
    Handles the proper workflow:
    1. Apply adapter before training
    2. Train your LoRA
    3. Save LoRA (without adapter baked in)
    4. For inference: load LoRA, optionally invert adapter
    """
    
    def __init__(
        self,
        adapter_path: Optional[str] = None,
        adapter_weight: float = 1.0,
    ):
        """
        Initialize adapter-aware training.
        
        Args:
            adapter_path: Path to training adapter
            adapter_weight: Weight to apply adapter with (usually 1.0)
        """
        self.adapter_manager = TrainingAdapterManager(adapter_path) if adapter_path else None
        self.adapter_weight = adapter_weight
        self.model_modified = False
    
    def prepare_model_for_training(
        self,
        model: nn.Module,
        verbose: bool = False,
    ) -> nn.Module:
        """
        Prepare model for training by applying adapter.
        
        Args:
            model: Model to prepare
            verbose: Print info
            
        Returns:
            Modified model
        """
        if self.adapter_manager is None:
            if verbose:
                print("No adapter specified, training without adapter")
            return model
        
        if verbose:
            print(f"Applying training adapter with weight {self.adapter_weight}")
        
        self.adapter_manager.apply_to_model(model, self.adapter_weight, verbose)
        self.model_modified = True
        
        return model
    
    def prepare_model_for_inference(
        self,
        model: nn.Module,
        invert_adapter: bool = True,
        verbose: bool = False,
    ) -> nn.Module:
        """
        Prepare model for inference after training.
        
        Args:
            model: Model to prepare
            invert_adapter: Whether to invert the adapter (recommended)
            verbose: Print info
            
        Returns:
            Modified model
        """
        if self.adapter_manager is None or not self.model_modified:
            return model
        
        if invert_adapter:
            if verbose:
                print("Inverting training adapter for inference")
            # Apply inverse weight to remove adapter effect
            self.adapter_manager.apply_to_model(model, -self.adapter_weight, verbose)
        
        return model
    
    def get_inference_instructions(self) -> str:
        """
        Get instructions for using trained LoRA during inference.
        
        Returns:
            Markdown-formatted instructions
        """
        return """
## Using Your Trained LoRA

Your LoRA was trained with the de-distillation adapter applied to the model.
For inference, you have two options:

### Option 1: Standard Inference (Recommended for Turbo)
1. Load the base Z-Image-Turbo model
2. Apply your trained LoRA
3. Generate with turbo settings (8 steps, CFG 1.0)

This works because your LoRA learned to work with the turbo model's 
characteristics, and the adapter effects are "baked into" what your LoRA learned.

### Option 2: Invert Adapter (For longer training runs)
If you trained for many steps (>3000) and notice quality degradation:
1. Load the base Z-Image-Turbo model
2. Apply the training adapter with weight -1.0 (invert)
3. Apply your trained LoRA
4. Generate with standard settings (20-30 steps, CFG 2.0-3.0)

The inverted adapter removes the de-distillation effect while preserving
your LoRA's learned content.
"""
