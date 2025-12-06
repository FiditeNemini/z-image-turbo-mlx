"""
Model Merge Support for Z-Image-Turbo MLX.

This module provides functionality to merge multiple Z-Image-Turbo models
using various algorithms:

1. Weighted Sum: (1 - α) * A + α * B
   - Most intuitive method for blending two models
   - α = 0.0: 100% Model A
   - α = 0.5: 50/50 blend
   - α = 1.0: 100% Model B

2. Add Difference: A + α * (B - C)
   - Extracts the "difference" that B learned relative to C
   - Applies that difference to A with strength α
   - Useful when B is a fine-tune of C

For 3+ models, sequential merging is used: ((A ⊕ B) ⊕ C) ⊕ D...

Memory Management:
- Automatically uses chunked processing when RAM < 32GB
- Falls back to chunked mode if psutil is unavailable

Usage:
    from merge import (
        weighted_sum_merge, 
        add_difference_merge,
        get_available_merge_models,
        merge_models_sequential,
        save_merged_model
    )
    
    # Simple two-model weighted sum
    merged = weighted_sum_merge(weights_a, weights_b, alpha=0.5)
    
    # Add difference (extract B's training relative to C, apply to A)
    merged = add_difference_merge(weights_a, weights_b, weights_c, alpha=1.0)
    
    # Sequential multi-model merge
    merged = merge_models_sequential(
        base_weights,
        [("model_b", 0.3), ("model_c", 0.2)],  # (model_name, weight) pairs
        method="weighted_sum"
    )
"""

import mlx.core as mx
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import json


# Memory threshold for chunked processing (in GB)
CHUNKED_MODE_THRESHOLD_GB = 32.0


def get_available_ram_gb() -> Optional[float]:
    """
    Get available system RAM in GB.
    
    Returns:
        Available RAM in GB, or None if cannot be determined.
        When None is returned, the caller should fall back to chunked mode.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except ImportError:
        # psutil not available, return None to trigger fallback
        return None
    except Exception:
        return None


def should_use_chunked_mode(num_models: int = 2) -> Tuple[bool, str]:
    """
    Determine if chunked processing should be used based on available RAM.
    
    Args:
        num_models: Number of models that will be loaded simultaneously
        
    Returns:
        Tuple of (should_use_chunked, reason_message)
    """
    available = get_available_ram_gb()
    
    if available is None:
        return True, "Memory-safe mode (psutil unavailable)"
    
    # Estimate: each model needs ~8-12GB, we need to hold num_models + output
    estimated_need = (num_models + 1) * 10  # Conservative estimate in GB
    
    if available < CHUNKED_MODE_THRESHOLD_GB:
        return True, f"Memory-safe mode ({available:.1f}GB available, <{CHUNKED_MODE_THRESHOLD_GB}GB threshold)"
    
    if available < estimated_need:
        return True, f"Memory-safe mode ({available:.1f}GB available, need ~{estimated_need}GB)"
    
    return False, f"Standard mode ({available:.1f}GB available)"


def _is_quantized_weights(weights: Dict[str, mx.array]) -> bool:
    """Check if weights are in quantized format (have .scales and .biases keys)."""
    for key in weights.keys():
        if key.endswith('.scales') or key.endswith('.biases'):
            return True
    return False


def load_model_weights(model_path: Union[str, Path]) -> Dict[str, mx.array]:
    """
    Load transformer weights from a model directory.
    
    Args:
        model_path: Path to the MLX model directory
        
    Returns:
        Dictionary of weight tensors
        
    Raises:
        FileNotFoundError: If weights file doesn't exist
        ValueError: If model is FP8 quantized (not supported for merging)
    """
    model_path = Path(model_path)
    weights_file = model_path / "weights.safetensors"
    
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")
    
    weights = mx.load(str(weights_file))
    
    # Check for quantization - we don't support merging quantized models
    if _is_quantized_weights(weights):
        raise ValueError(
            f"Model at {model_path} appears to be FP8 quantized. "
            "Merging is only supported for FP16+ precision models. "
            "Please use non-quantized models for merging."
        )
    
    return weights


def weighted_sum_merge(
    weights_a: Dict[str, mx.array],
    weights_b: Dict[str, mx.array],
    alpha: float = 0.5
) -> Dict[str, mx.array]:
    """
    Merge two weight dictionaries using weighted sum.
    
    Formula: merged = (1 - α) * A + α * B
    
    Args:
        weights_a: First model's weights (base)
        weights_b: Second model's weights
        alpha: Blend ratio (0.0 = 100% A, 1.0 = 100% B)
        
    Returns:
        Merged weight dictionary
    """
    merged = {}
    
    for key in weights_a.keys():
        if key in weights_b:
            # Both models have this weight - blend them
            a = weights_a[key]
            b = weights_b[key]
            
            if a.shape == b.shape:
                merged[key] = (1 - alpha) * a + alpha * b
            else:
                # Shape mismatch - keep A's weight and warn
                print(f"Warning: Shape mismatch for {key}: {a.shape} vs {b.shape}, keeping base")
                merged[key] = a
        else:
            # Only in A - keep it
            merged[key] = weights_a[key]
    
    # Check for weights only in B (shouldn't happen with same architecture)
    for key in weights_b.keys():
        if key not in weights_a:
            print(f"Warning: Key {key} only in model B, skipping")
    
    return merged


def add_difference_merge(
    weights_a: Dict[str, mx.array],
    weights_b: Dict[str, mx.array],
    weights_c: Dict[str, mx.array],
    alpha: float = 1.0
) -> Dict[str, mx.array]:
    """
    Merge using add-difference method.
    
    Formula: merged = A + α * (B - C)
    
    This extracts what B learned relative to C (the "difference"),
    and applies it to A with strength α.
    
    Typically:
    - A: Your base model
    - B: A fine-tuned model
    - C: The model B was fine-tuned from (original)
    
    Args:
        weights_a: Base model weights
        weights_b: Fine-tuned model weights  
        weights_c: Original model (that B was trained from)
        alpha: Strength of the difference (1.0 = full strength)
        
    Returns:
        Merged weight dictionary
    """
    merged = {}
    
    for key in weights_a.keys():
        if key in weights_b and key in weights_c:
            a = weights_a[key]
            b = weights_b[key]
            c = weights_c[key]
            
            if a.shape == b.shape == c.shape:
                # Apply the difference: A + α * (B - C)
                diff = b - c
                merged[key] = a + alpha * diff
            else:
                print(f"Warning: Shape mismatch for {key}, keeping base")
                merged[key] = a
        elif key in weights_b:
            # Missing in C - use weighted sum fallback between A and B
            a = weights_a[key]
            b = weights_b[key]
            if a.shape == b.shape:
                merged[key] = (1 - alpha * 0.5) * a + (alpha * 0.5) * b
            else:
                merged[key] = a
        else:
            # Only in A
            merged[key] = weights_a[key]
    
    return merged


def weighted_sum_merge_chunked(
    model_path_a: Union[str, Path],
    model_path_b: Union[str, Path],
    output_path: Union[str, Path],
    alpha: float = 0.5,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> None:
    """
    Merge two models using chunked processing for low-memory systems.
    
    Processes weights key-by-key to minimize memory usage.
    
    Args:
        model_path_a: Path to first model
        model_path_b: Path to second model
        output_path: Path to save merged weights
        alpha: Blend ratio
        progress_callback: Optional callback(progress, description)
    """
    from safetensors.numpy import save_file
    import numpy as np
    
    weights_file_a = Path(model_path_a) / "weights.safetensors"
    weights_file_b = Path(model_path_b) / "weights.safetensors"
    
    # Load weight keys first
    if progress_callback:
        progress_callback(0.05, "Loading weight keys...")
    
    # We need to load both to iterate, but we'll process one at a time
    weights_a = mx.load(str(weights_file_a))
    weights_b = mx.load(str(weights_file_b))
    
    # Check for quantization
    if _is_quantized_weights(weights_a) or _is_quantized_weights(weights_b):
        raise ValueError("Merging quantized (FP8) models is not supported. Use FP16+ models.")
    
    merged = {}
    keys = list(weights_a.keys())
    total_keys = len(keys)
    
    for i, key in enumerate(keys):
        if progress_callback and i % 50 == 0:
            progress_callback(0.1 + 0.7 * (i / total_keys), f"Merging weight {i+1}/{total_keys}...")
        
        if key in weights_b:
            a = weights_a[key]
            b = weights_b[key]
            
            if a.shape == b.shape:
                merged_weight = (1 - alpha) * a + alpha * b
                # Convert to numpy for safetensors
                merged[key] = np.array(merged_weight)
            else:
                merged[key] = np.array(a)
        else:
            merged[key] = np.array(weights_a[key])
        
        # Evaluate to free memory
        mx.eval(merged[key] if isinstance(merged[key], mx.array) else None)
    
    if progress_callback:
        progress_callback(0.85, "Saving merged weights...")
    
    # Save using numpy safetensors (more memory efficient)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output_path / "weights.safetensors"))


def add_difference_merge_chunked(
    model_path_a: Union[str, Path],
    model_path_b: Union[str, Path],
    model_path_c: Union[str, Path],
    output_path: Union[str, Path],
    alpha: float = 1.0,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> None:
    """
    Merge using add-difference method with chunked processing.
    
    Args:
        model_path_a: Path to base model
        model_path_b: Path to fine-tuned model
        model_path_c: Path to original model
        output_path: Path to save merged weights
        alpha: Difference strength
        progress_callback: Optional callback(progress, description)
    """
    from safetensors.numpy import save_file
    import numpy as np
    
    weights_file_a = Path(model_path_a) / "weights.safetensors"
    weights_file_b = Path(model_path_b) / "weights.safetensors"
    weights_file_c = Path(model_path_c) / "weights.safetensors"
    
    if progress_callback:
        progress_callback(0.05, "Loading model weights...")
    
    weights_a = mx.load(str(weights_file_a))
    weights_b = mx.load(str(weights_file_b))
    weights_c = mx.load(str(weights_file_c))
    
    # Check for quantization
    for weights, name in [(weights_a, "A"), (weights_b, "B"), (weights_c, "C")]:
        if _is_quantized_weights(weights):
            raise ValueError(f"Model {name} is quantized (FP8). Use FP16+ models for merging.")
    
    merged = {}
    keys = list(weights_a.keys())
    total_keys = len(keys)
    
    for i, key in enumerate(keys):
        if progress_callback and i % 50 == 0:
            progress_callback(0.1 + 0.7 * (i / total_keys), f"Merging weight {i+1}/{total_keys}...")
        
        if key in weights_b and key in weights_c:
            a = weights_a[key]
            b = weights_b[key]
            c = weights_c[key]
            
            if a.shape == b.shape == c.shape:
                diff = b - c
                merged_weight = a + alpha * diff
                merged[key] = np.array(merged_weight)
            else:
                merged[key] = np.array(a)
        elif key in weights_b:
            a = weights_a[key]
            b = weights_b[key]
            if a.shape == b.shape:
                merged[key] = np.array((1 - alpha * 0.5) * a + (alpha * 0.5) * b)
            else:
                merged[key] = np.array(a)
        else:
            merged[key] = np.array(weights_a[key])
        
        mx.eval(merged[key] if isinstance(merged[key], mx.array) else None)
    
    if progress_callback:
        progress_callback(0.85, "Saving merged weights...")
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output_path / "weights.safetensors"))


def get_available_merge_models(
    models_dir: Union[str, Path],
    exclude_quantized: bool = True
) -> List[Tuple[str, bool]]:
    """
    Get list of models available for merging.
    
    Args:
        models_dir: Path to models directory
        exclude_quantized: If True, exclude FP8 quantized models
        
    Returns:
        List of (model_name, is_compatible) tuples
        is_compatible is False if model is quantized (when exclude_quantized=True)
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    
    models = []
    
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        weights_file = model_dir / "weights.safetensors"
        if not weights_file.exists():
            continue
        
        is_compatible = True
        
        if exclude_quantized:
            # Check config for precision info first (faster)
            config_file = model_dir / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        config = json.load(f)
                    if config.get("precision") == "FP8":
                        is_compatible = False
                except Exception:
                    pass
            
            # If not marked in config, check the actual weights
            if is_compatible:
                try:
                    # Just load keys to check for quantization markers
                    weights = mx.load(str(weights_file))
                    if _is_quantized_weights(weights):
                        is_compatible = False
                    del weights  # Free memory
                except Exception:
                    is_compatible = False
        
        models.append((model_dir.name, is_compatible))
    
    return sorted(models, key=lambda x: x[0])


def copy_model_configs(
    source_path: Union[str, Path],
    output_path: Union[str, Path],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> None:
    """
    Copy non-weight files (configs, VAE, text encoder, tokenizer, scheduler) from source to output.
    
    Args:
        source_path: Path to source model directory
        output_path: Path to output model directory
        progress_callback: Optional callback(progress, description)
    """
    import shutil
    
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Files and directories to copy (excluding weights.safetensors which we're merging)
    items_to_copy = [
        "config.json",
        "vae_config.json",
        "text_encoder_config.json",
        "vae.safetensors",
        "text_encoder.safetensors",
        "tokenizer",
        "scheduler",
    ]
    
    for item in items_to_copy:
        source_item = source_path / item
        dest_item = output_path / item
        
        if not source_item.exists():
            continue
        
        if progress_callback:
            progress_callback(None, f"Copying {item}...")
        
        if source_item.is_dir():
            if dest_item.exists():
                shutil.rmtree(dest_item)
            shutil.copytree(source_item, dest_item)
        else:
            shutil.copy2(source_item, dest_item)


def merge_models_sequential(
    base_weights: Dict[str, mx.array],
    merge_configs: List[Tuple[Dict[str, mx.array], float]],
    method: str = "weighted_sum"
) -> Dict[str, mx.array]:
    """
    Merge multiple models sequentially.
    
    For weighted_sum: ((A ⊕ B) ⊕ C) ⊕ D...
    Each merge uses the weight as alpha.
    
    Args:
        base_weights: Starting model weights
        merge_configs: List of (weights_dict, weight) tuples
        method: "weighted_sum" only (add_difference requires 3 models per operation)
        
    Returns:
        Final merged weights
    """
    if method != "weighted_sum":
        raise ValueError("Sequential merging only supports weighted_sum method")
    
    result = base_weights
    
    for weights_b, alpha in merge_configs:
        result = weighted_sum_merge(result, weights_b, alpha)
        mx.eval(result)  # Ensure computation is done before next iteration
    
    return result


def save_merged_model(
    merged_weights: Dict[str, mx.array],
    source_model_path: Union[str, Path],
    output_path: Union[str, Path],
    merge_info: Optional[Dict] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> None:
    """
    Save a merged model with all necessary files.
    
    Args:
        merged_weights: The merged weight dictionary
        source_model_path: Path to source model (for copying configs/VAE/etc)
        output_path: Path to save the merged model
        merge_info: Optional metadata about the merge operation
        progress_callback: Optional callback(progress, description)
    """
    from safetensors.numpy import save_file
    import numpy as np
    
    output_path = Path(output_path)
    
    if progress_callback:
        progress_callback(0.1, "Copying model configs...")
    
    # Copy non-weight files from source
    copy_model_configs(source_model_path, output_path, progress_callback)
    
    if progress_callback:
        progress_callback(0.5, "Converting weights for saving...")
    
    # Convert MLX arrays to numpy for saving
    numpy_weights = {}
    for key, value in merged_weights.items():
        numpy_weights[key] = np.array(value)
    
    if progress_callback:
        progress_callback(0.7, "Saving merged weights...")
    
    # Save weights
    save_file(numpy_weights, str(output_path / "weights.safetensors"))
    
    if progress_callback:
        progress_callback(0.9, "Saving merge metadata...")
    
    # Update config with merge info
    config_path = output_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
    
    # Add merge metadata
    if merge_info:
        config["merge_info"] = merge_info
    config["precision"] = "FP16"  # Merged models are always FP16
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Save detailed merge info to separate file
    if merge_info:
        from datetime import datetime
        merge_info["created"] = datetime.now().isoformat()
        with open(output_path / "merge_info.json", "w") as f:
            json.dump(merge_info, f, indent=2)
    
    if progress_callback:
        progress_callback(1.0, "Done!")


def perform_merge(
    base_model_path: Union[str, Path],
    merge_models: List[Tuple[Union[str, Path], float]],
    output_path: Union[str, Path],
    method: str = "weighted_sum",
    model_c_path: Optional[Union[str, Path]] = None,
    add_diff_alpha: float = 1.0,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> str:
    """
    Main merge function that handles the complete merge workflow.
    
    Args:
        base_model_path: Path to the base model (Model A)
        merge_models: List of (model_path, weight) tuples for models to merge
        output_path: Path to save the merged model
        method: "weighted_sum" or "add_difference"
        model_c_path: For add_difference, the "original" model path
        add_diff_alpha: For add_difference, the difference strength
        progress_callback: Optional callback(progress, description)
        
    Returns:
        Success message or error description
    """
    try:
        base_model_path = Path(base_model_path)
        output_path = Path(output_path)
        
        # Check if output already exists
        if output_path.exists():
            return f"❌ Output model already exists: {output_path.name}"
        
        # Determine if we should use chunked mode
        num_models = len(merge_models) + 1 + (1 if model_c_path else 0)
        use_chunked, memory_msg = should_use_chunked_mode(num_models)
        
        if progress_callback:
            progress_callback(0.02, memory_msg)
        
        if method == "add_difference":
            if not model_c_path:
                return "❌ Add Difference requires Model C (the original model)"
            
            if len(merge_models) != 1:
                return "❌ Add Difference requires exactly one model to merge (Model B)"
            
            model_b_path, _ = merge_models[0]  # Weight is ignored, use add_diff_alpha
            
            if use_chunked:
                if progress_callback:
                    progress_callback(0.05, "Using memory-safe chunked merge...")
                add_difference_merge_chunked(
                    base_model_path, model_b_path, model_c_path, output_path,
                    alpha=add_diff_alpha,
                    progress_callback=progress_callback
                )
            else:
                if progress_callback:
                    progress_callback(0.05, "Loading models...")
                
                weights_a = load_model_weights(base_model_path)
                if progress_callback:
                    progress_callback(0.2, "Loading Model B...")
                weights_b = load_model_weights(model_b_path)
                if progress_callback:
                    progress_callback(0.35, "Loading Model C...")
                weights_c = load_model_weights(model_c_path)
                
                if progress_callback:
                    progress_callback(0.5, "Applying Add Difference...")
                merged = add_difference_merge(weights_a, weights_b, weights_c, add_diff_alpha)
                mx.eval(merged)
                
                if progress_callback:
                    progress_callback(0.7, "Saving merged model...")
                
                merge_info = {
                    "method": "add_difference",
                    "base_model": base_model_path.name,
                    "model_b": Path(model_b_path).name,
                    "model_c": Path(model_c_path).name,
                    "alpha": add_diff_alpha,
                }
                save_merged_model(merged, base_model_path, output_path, merge_info, progress_callback)
            
            return f"✅ Created merged model: {output_path.name}"
        
        elif method == "weighted_sum":
            if use_chunked and len(merge_models) == 1:
                # Simple two-model chunked merge
                model_b_path, alpha = merge_models[0]
                if progress_callback:
                    progress_callback(0.05, "Using memory-safe chunked merge...")
                weighted_sum_merge_chunked(
                    base_model_path, model_b_path, output_path,
                    alpha=alpha,
                    progress_callback=progress_callback
                )
                # Copy configs after chunked merge
                copy_model_configs(base_model_path, output_path, progress_callback)
            else:
                # Standard merge (handles multiple models)
                if progress_callback:
                    progress_callback(0.05, "Loading base model...")
                base_weights = load_model_weights(base_model_path)
                
                # Load and merge each model sequentially
                merge_configs = []
                for i, (model_path, weight) in enumerate(merge_models):
                    if progress_callback:
                        progress_callback(0.1 + (0.4 * i / len(merge_models)), 
                                        f"Loading model {i+1}/{len(merge_models)}...")
                    weights = load_model_weights(model_path)
                    merge_configs.append((weights, weight))
                
                if progress_callback:
                    progress_callback(0.5, "Merging models...")
                
                merged = merge_models_sequential(base_weights, merge_configs, "weighted_sum")
                
                if progress_callback:
                    progress_callback(0.7, "Saving merged model...")
                
                merge_info = {
                    "method": "weighted_sum",
                    "base_model": base_model_path.name,
                    "merged_models": [(Path(p).name, w) for p, w in merge_models],
                }
                save_merged_model(merged, base_model_path, output_path, merge_info, progress_callback)
            
            model_names = ", ".join([f"{Path(p).name} ({w:.2f})" for p, w in merge_models])
            return f"✅ Created merged model: {output_path.name}\n   Base: {base_model_path.name}\n   Merged: {model_names}"
        
        else:
            return f"❌ Unknown merge method: {method}"
        
    except ValueError as e:
        return f"❌ {str(e)}"
    except FileNotFoundError as e:
        return f"❌ {str(e)}"
    except Exception as e:
        return f"❌ Error during merge: {str(e)}"


# Convenience functions for testing
def test_merge():
    """Test merge functionality."""
    available = get_available_ram_gb()
    print(f"Available RAM: {available:.1f}GB" if available else "RAM detection unavailable")
    
    use_chunked, msg = should_use_chunked_mode(2)
    print(f"Mode: {msg}")


if __name__ == "__main__":
    test_merge()
