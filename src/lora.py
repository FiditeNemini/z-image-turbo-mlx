"""
LoRA (Low-Rank Adaptation) support for Z-Image-Turbo MLX.

This module provides functionality to load and apply LoRA weights to the
Z-Image-Turbo transformer model. It supports:
- ComfyUI format LoRAs (diffusion_model.* prefix)
- Multiple LoRAs with independent strength/scale values
- Runtime weight merging (W' = W + scale * B @ A)

Usage:
    from lora import load_lora, apply_lora_to_model, get_available_loras
    
    # List available LoRAs
    loras = get_available_loras()
    
    # Load and apply a single LoRA
    lora_weights = load_lora("path/to/lora.safetensors")
    apply_lora_to_model(model, lora_weights, scale=1.0)
    
    # Apply multiple LoRAs
    for path, scale in zip(lora_paths, lora_scales):
        lora_weights = load_lora(path)
        apply_lora_to_model(model, lora_weights, scale=scale)
"""

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re


# Default LoRA directory
LORAS_DIR = Path(__file__).parent.parent / "models" / "loras"


def get_available_loras(loras_dir: Optional[Path] = None, recursive: bool = True) -> List[str]:
    """
    Get list of available LoRA files in the loras directory.
    
    Args:
        loras_dir: Optional path to loras directory. Defaults to project's models/loras/ folder.
        recursive: If True, scan subfolders recursively
        
    Returns:
        List of LoRA filenames (relative paths from loras_dir if in subfolders)
    """
    if loras_dir is None:
        loras_dir = LORAS_DIR
    
    loras_dir = Path(loras_dir)
    if not loras_dir.exists():
        return []
    
    lora_files = []
    
    if recursive:
        # Recursively find all .safetensors files
        for f in loras_dir.rglob("*.safetensors"):
            if f.name != ".gitkeep":
                # Return relative path from loras_dir
                rel_path = f.relative_to(loras_dir)
                lora_files.append(str(rel_path))
    else:
        # Only scan top-level directory
        for f in loras_dir.iterdir():
            if f.suffix.lower() == ".safetensors" and f.name != ".gitkeep":
                lora_files.append(f.name)
    
    return sorted(lora_files)


def get_lora_with_folders(loras_dir: Optional[Path] = None) -> List[Tuple[str, str]]:
    """
    Get list of available LoRAs with their folder information.
    
    Args:
        loras_dir: Optional path to loras directory.
        
    Returns:
        List of (folder, filename) tuples, where folder is "" for root level files
    """
    if loras_dir is None:
        loras_dir = LORAS_DIR
    
    loras_dir = Path(loras_dir)
    if not loras_dir.exists():
        return []
    
    lora_entries = []
    
    for f in loras_dir.rglob("*.safetensors"):
        if f.name != ".gitkeep":
            rel_path = f.relative_to(loras_dir)
            # Get folder (parent relative to loras_dir), empty string if at root
            if rel_path.parent == Path("."):
                folder = ""
            else:
                folder = str(rel_path.parent)
            lora_entries.append((folder, f.name))
    
    # Sort by folder then by name
    return sorted(lora_entries, key=lambda x: (x[0], x[1]))


def load_lora(lora_path: Union[str, Path]) -> Dict[str, mx.array]:
    """
    Load LoRA weights from a safetensors file.
    
    Args:
        lora_path: Path to the LoRA .safetensors file
        
    Returns:
        Dictionary mapping LoRA keys to weight arrays
        
    Raises:
        FileNotFoundError: If the LoRA file doesn't exist
        ValueError: If the file format is invalid
    """
    lora_path = Path(lora_path)
    
    # If just a filename, look in default loras directory
    if not lora_path.exists() and not lora_path.is_absolute():
        lora_path = LORAS_DIR / lora_path
    
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")
    
    weights = mx.load(str(lora_path))
    
    if len(weights) == 0:
        raise ValueError(f"LoRA file is empty: {lora_path}")
    
    return weights


def map_lora_key_to_model_key(lora_key: str) -> Optional[str]:
    """
    Map a LoRA weight key to the corresponding model weight key.
    
    Handles ComfyUI format with diffusion_model.* prefix.
    
    Args:
        lora_key: Key from LoRA file (e.g., "diffusion_model.layers.0.attention.to_q.lora_A.weight")
        
    Returns:
        Corresponding model key (e.g., "layers.0.attention.to_q.weight") or None if not mappable
    """
    # Remove lora_A/lora_B suffix to get base key
    if ".lora_A.weight" in lora_key:
        base_key = lora_key.replace(".lora_A.weight", "")
    elif ".lora_B.weight" in lora_key:
        base_key = lora_key.replace(".lora_B.weight", "")
    else:
        # Not a LoRA weight key
        return None
    
    # Remove diffusion_model. prefix (ComfyUI format)
    if base_key.startswith("diffusion_model."):
        base_key = base_key[len("diffusion_model."):]
    
    # Handle to_out.0 -> to_out mapping (Sequential wrapper in original)
    base_key = base_key.replace(".to_out.0", ".to_out")
    
    # Handle adaLN_modulation.0 -> adaLN_modulation (Sequential wrapper)
    base_key = base_key.replace(".adaLN_modulation.0", ".adaLN_modulation")
    
    # Add .weight suffix for the model key
    model_key = base_key + ".weight"
    
    return model_key


def parse_lora_weights(lora_weights: Dict[str, mx.array]) -> Dict[str, Tuple[mx.array, mx.array]]:
    """
    Parse LoRA weights into (A, B) pairs keyed by model weight path.
    
    Args:
        lora_weights: Raw LoRA weights from load_lora()
        
    Returns:
        Dictionary mapping model weight keys to (lora_A, lora_B) tuples
    """
    # Group weights by their base key
    lora_pairs: Dict[str, Dict[str, mx.array]] = {}
    
    for key, weight in lora_weights.items():
        if ".lora_A.weight" in key:
            model_key = map_lora_key_to_model_key(key)
            if model_key:
                if model_key not in lora_pairs:
                    lora_pairs[model_key] = {}
                lora_pairs[model_key]["A"] = weight
        elif ".lora_B.weight" in key:
            model_key = map_lora_key_to_model_key(key)
            if model_key:
                if model_key not in lora_pairs:
                    lora_pairs[model_key] = {}
                lora_pairs[model_key]["B"] = weight
    
    # Convert to (A, B) tuples, only keeping complete pairs
    result = {}
    for model_key, pair in lora_pairs.items():
        if "A" in pair and "B" in pair:
            result[model_key] = (pair["A"], pair["B"])
        else:
            print(f"Warning: Incomplete LoRA pair for {model_key}, skipping")
    
    return result


def compute_lora_delta(lora_a: mx.array, lora_b: mx.array, scale: float = 1.0) -> mx.array:
    """
    Compute the weight delta from LoRA matrices.
    
    The formula is: delta = scale * (B @ A)
    
    For nn.Linear weights which are stored as (out_features, in_features):
    - lora_A: (rank, in_features) 
    - lora_B: (out_features, rank)
    - delta: (out_features, in_features)
    
    Args:
        lora_a: The A (down-projection) matrix
        lora_b: The B (up-projection) matrix  
        scale: Scaling factor for the LoRA contribution
        
    Returns:
        Weight delta to add to the base weight
    """
    # LoRA format: A is (rank, in_features), B is (out_features, rank)
    # delta = B @ A = (out_features, rank) @ (rank, in_features) = (out_features, in_features)
    delta = lora_b @ lora_a
    return (scale * delta).astype(lora_a.dtype)


def get_nested_attr(obj: nn.Module, path: str):
    """
    Get a nested attribute from an object using dot notation.
    
    Handles list indexing (e.g., "layers.0.attention").
    
    Args:
        obj: The root object
        path: Dot-separated path (e.g., "layers.0.attention.to_q")
        
    Returns:
        The nested attribute
    """
    parts = path.split(".")
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def set_nested_attr(obj: nn.Module, path: str, value):
    """
    Set a nested attribute on an object using dot notation.
    
    Args:
        obj: The root object
        path: Dot-separated path (e.g., "layers.0.attention.to_q.weight")
        value: The value to set
    """
    parts = path.split(".")
    # Navigate to parent
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    # Set the final attribute
    setattr(obj, parts[-1], value)


def apply_lora_to_model(
    model: nn.Module,
    lora_weights: Dict[str, mx.array],
    scale: float = 1.0,
    verbose: bool = False
) -> int:
    """
    Apply LoRA weights to a model by merging them into the base weights.
    
    This modifies the model weights in-place using the formula:
    W' = W + scale * (B @ A)
    
    Args:
        model: The Z-Image transformer model
        lora_weights: LoRA weights from load_lora()
        scale: Strength/scale factor for the LoRA (default 1.0)
        verbose: If True, print information about applied weights
        
    Returns:
        Number of weights successfully modified
    """
    # Parse LoRA weights into (A, B) pairs
    lora_pairs = parse_lora_weights(lora_weights)
    
    if verbose:
        print(f"Found {len(lora_pairs)} LoRA weight pairs to apply")
    
    applied_count = 0
    
    for model_key, (lora_a, lora_b) in lora_pairs.items():
        # Remove .weight suffix to get the layer path
        layer_path = model_key.rsplit(".weight", 1)[0]
        
        try:
            # Get the current weight
            layer = get_nested_attr(model, layer_path)
            
            if not hasattr(layer, "weight"):
                if verbose:
                    print(f"  Skipping {layer_path}: no weight attribute")
                continue
            
            current_weight = layer.weight
            
            # Compute and apply the LoRA delta
            delta = compute_lora_delta(lora_a, lora_b, scale)
            
            # Verify shapes match
            if delta.shape != current_weight.shape:
                if verbose:
                    print(f"  Skipping {layer_path}: shape mismatch "
                          f"(delta {delta.shape} vs weight {current_weight.shape})")
                continue
            
            # Apply the delta
            new_weight = current_weight + delta.astype(current_weight.dtype)
            layer.weight = new_weight
            
            applied_count += 1
            if verbose:
                print(f"  Applied LoRA to {layer_path}")
                
        except (AttributeError, IndexError, KeyError) as e:
            if verbose:
                print(f"  Skipping {layer_path}: {e}")
            continue
    
    # Ensure computation is complete
    mx.eval(model.parameters())
    
    if verbose:
        print(f"Successfully applied LoRA to {applied_count}/{len(lora_pairs)} weights")
    
    return applied_count


def apply_multiple_loras(
    model: nn.Module,
    lora_paths: List[Union[str, Path]],
    scales: Optional[List[float]] = None,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Apply multiple LoRAs to a model sequentially.
    
    Each LoRA is applied additively to the weights:
    W' = W + scale1 * (B1 @ A1) + scale2 * (B2 @ A2) + ...
    
    Args:
        model: The Z-Image transformer model
        lora_paths: List of paths to LoRA files
        scales: Optional list of scale factors (default 1.0 for each)
        verbose: If True, print information about applied weights
        
    Returns:
        Dictionary mapping LoRA names to number of weights applied
    """
    if scales is None:
        scales = [1.0] * len(lora_paths)
    
    if len(scales) != len(lora_paths):
        raise ValueError(f"Number of scales ({len(scales)}) must match number of LoRAs ({len(lora_paths)})")
    
    results = {}
    
    for path, scale in zip(lora_paths, scales):
        path = Path(path)
        lora_name = path.stem if path.suffix else path.name
        
        if verbose:
            print(f"\nApplying LoRA: {lora_name} (scale={scale})")
        
        try:
            lora_weights = load_lora(path)
            applied = apply_lora_to_model(model, lora_weights, scale, verbose)
            results[lora_name] = applied
        except Exception as e:
            print(f"Error applying LoRA {lora_name}: {e}")
            results[lora_name] = 0
    
    return results


def get_lora_metadata(lora_path: Union[str, Path]) -> Dict:
    """
    Get metadata from a LoRA file (trigger words, training info, etc.).
    
    Args:
        lora_path: Path to the LoRA file
        
    Returns:
        Dictionary with metadata fields
    """
    from safetensors import safe_open
    
    lora_path = Path(lora_path)
    if not lora_path.exists() and not lora_path.is_absolute():
        lora_path = LORAS_DIR / lora_path
    
    metadata = {}
    try:
        with safe_open(str(lora_path), framework="numpy", device="cpu") as f:
            raw_metadata = f.metadata()
            if raw_metadata:
                metadata = dict(raw_metadata)
    except Exception as e:
        print(f"Warning: Could not read LoRA metadata: {e}")
    
    return metadata


def get_lora_default_weight(lora_path: Union[str, Path]) -> float:
    """
    Extract recommended/default weight from LoRA metadata.
    
    Looks for weight info in common metadata fields:
    - recommended_weight / preferred_weight / default_weight
    - strength / default_strength
    - multiplier / scale
    - ss_network_alpha (kohya_ss format - normalized by rank)
    
    Args:
        lora_path: Path to the LoRA file
        
    Returns:
        Default weight (1.0 if not found in metadata)
    """
    metadata = get_lora_metadata(lora_path)
    
    # Check common weight fields (in order of priority)
    weight_fields = [
        "recommended_weight",
        "preferred_weight", 
        "default_weight",
        "strength",
        "default_strength",
        "multiplier",
        "scale",
    ]
    
    for field in weight_fields:
        if field in metadata:
            try:
                return float(metadata[field])
            except (ValueError, TypeError):
                continue
    
    # Check CivitAI modelspec format (may contain weight recommendations)
    if "modelspec.metadata" in metadata:
        try:
            import json
            spec = json.loads(metadata["modelspec.metadata"])
            if "preferred_weight" in spec:
                return float(spec["preferred_weight"])
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            pass
    
    # Default to 1.0
    return 1.0


def get_lora_trigger_words(lora_path: Union[str, Path]) -> List[str]:
    """
    Extract trigger words from LoRA metadata.
    
    Looks for trigger words in common metadata fields:
    - ss_tag_frequency (ai-toolkit format)
    - trigger_word / trigger_words
    - activation_text
    
    Args:
        lora_path: Path to the LoRA file
        
    Returns:
        List of trigger words found
    """
    import json
    
    metadata = get_lora_metadata(lora_path)
    trigger_words = []
    
    # Check ss_tag_frequency (ai-toolkit format: {"folder": {"tag": count}})
    if "ss_tag_frequency" in metadata:
        try:
            tag_freq = json.loads(metadata["ss_tag_frequency"])
            for folder_tags in tag_freq.values():
                if isinstance(folder_tags, dict):
                    trigger_words.extend(folder_tags.keys())
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Check common trigger word fields
    for field in ["trigger_word", "trigger_words", "activation_text"]:
        if field in metadata:
            value = metadata[field]
            if isinstance(value, str):
                # Could be comma-separated
                trigger_words.extend([w.strip() for w in value.split(",") if w.strip()])
    
    return list(set(trigger_words))  # Remove duplicates


def get_lora_info(lora_path: Union[str, Path]) -> Dict:
    """
    Get information about a LoRA file without fully loading it.
    
    Args:
        lora_path: Path to the LoRA file
        
    Returns:
        Dictionary with LoRA metadata (rank, target layers, trigger words, etc.)
    """
    lora_weights = load_lora(lora_path)
    lora_pairs = parse_lora_weights(lora_weights)
    
    # Detect rank from the first A matrix
    rank = None
    for model_key, (lora_a, lora_b) in lora_pairs.items():
        rank = lora_a.shape[0]  # A is (rank, in_features)
        break
    
    # Get target layer types
    target_types = set()
    for key in lora_pairs.keys():
        if "attention.to_q" in key or "attention.to_k" in key or "attention.to_v" in key:
            target_types.add("attention_qkv")
        elif "attention.to_out" in key:
            target_types.add("attention_out")
        elif "feed_forward" in key:
            target_types.add("feed_forward")
        elif "adaLN_modulation" in key:
            target_types.add("adaLN")
    
    # Get layer indices
    layer_indices = set()
    for key in lora_pairs.keys():
        match = re.search(r"layers\.(\d+)\.", key)
        if match:
            layer_indices.add(int(match.group(1)))
    
    # Get metadata and trigger words
    metadata = get_lora_metadata(lora_path)
    trigger_words = get_lora_trigger_words(lora_path)
    
    return {
        "path": str(lora_path),
        "name": metadata.get("name", Path(lora_path).stem),
        "rank": rank,
        "num_weights": len(lora_pairs),
        "target_types": sorted(target_types),
        "layer_indices": sorted(layer_indices),
        "layer_range": (min(layer_indices), max(layer_indices)) if layer_indices else None,
        "trigger_words": trigger_words,
        "metadata": metadata,
    }


# Convenience function for testing
def test_lora_loading():
    """Test LoRA loading functionality with available LoRAs."""
    print("Available LoRAs:")
    loras = get_available_loras()
    for lora in loras:
        print(f"  - {lora}")
        info = get_lora_info(LORAS_DIR / lora)
        print(f"    Rank: {info['rank']}, Weights: {info['num_weights']}")
        print(f"    Targets: {info['target_types']}")
        print(f"    Layers: {info['layer_range']}")
        if info['trigger_words']:
            print(f"    Trigger words: {', '.join(info['trigger_words'])}")
        if info['metadata'].get('name'):
            print(f"    Name: {info['metadata']['name']}")
    
    return loras


if __name__ == "__main__":
    test_lora_loading()
