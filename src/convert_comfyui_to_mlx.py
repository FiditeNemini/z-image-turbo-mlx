#!/usr/bin/env python3
"""
Convert ComfyUI single-file Z-Image-Turbo checkpoint to MLX format.

ComfyUI checkpoints have all components in one file with prefixes:
- diffusion_* : Transformer weights
- text_encoders.qwen3_4b.* : Text encoder weights  
- vae.* : VAE weights (may or may not be present)
"""

import argparse
import os
import json
import shutil
from pathlib import Path
from safetensors import safe_open
from tqdm import tqdm

try:
    import mlx.core as mx
except ImportError:
    print("MLX not available. Install with: pip install mlx")
    mx = None


# Reference configs for Z-Image-Turbo architecture
DEFAULT_TRANSFORMER_CONFIG = {
    "hidden_size": 3840,
    "num_attention_heads": 30,
    "intermediate_size": 10240,
    "num_hidden_layers": 30,
    "n_refiner_layers": 2,
    "in_channels": 16,
    "text_embed_dim": 2560,
    "patch_size": 2,
    "rope_theta": 256.0,
    "axes_dims": [32, 48, 48],
    "axes_lens": [1536, 512, 512],
}

DEFAULT_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 16,
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
}

DEFAULT_TEXT_ENCODER_CONFIG = {
    "hidden_size": 2560,
    "intermediate_size": 13696,
    "num_attention_heads": 20,
    "num_hidden_layers": 36,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
}

DEFAULT_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.33.0.dev0",
    "base_image_seq_len": 4096,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "shift_terminal": None,
    "use_dynamic_shifting": False,
    "use_karras_sigmas": False,
}


def analyze_checkpoint(checkpoint_path: str) -> dict:
    """Analyze a checkpoint to identify its components."""
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    components = {
        "transformer": [],
        "text_encoder": [],
        "vae": [],
        "unknown": [],
    }
    
    with safe_open(checkpoint_path, framework="pt") as f:
        keys = list(f.keys())
        
    for key in keys:
        # ComfyUI format: diffusion_* or model.diffusion_model.*
        if key.startswith("diffusion_") or key.startswith("model.diffusion_model."):
            components["transformer"].append(key)
        elif key.startswith("text_encoders.qwen3_4b."):
            components["text_encoder"].append(key)
        elif key.startswith("vae."):
            components["vae"].append(key)
        else:
            # Could be transformer without prefix
            if any(x in key for x in ["layers.", "x_embedder", "t_embedder", "final_layer", "context_refiner", "noise_refiner"]):
                components["transformer"].append(key)
            else:
                components["unknown"].append(key)
    
    print(f"  Transformer keys: {len(components['transformer'])}")
    print(f"  Text encoder keys: {len(components['text_encoder'])}")
    print(f"  VAE keys: {len(components['vae'])}")
    print(f"  Unknown keys: {len(components['unknown'])}")
    
    if components["unknown"]:
        print(f"  Unknown key samples: {components['unknown'][:5]}")
    
    return components


def convert_transformer_key(key: str) -> str:
    """Convert transformer key from ComfyUI format to MLX format.
    
    Note: This handles key name mapping but NOT QKV splitting.
    QKV splitting must be done separately when the weight is fused.
    """
    new_key = key
    
    # Remove model.diffusion_model. prefix (ComfyUI format)
    if new_key.startswith("model.diffusion_model."):
        new_key = new_key[len("model.diffusion_model."):]
    
    # Remove diffusion_ prefix
    if new_key.startswith("diffusion_"):
        new_key = new_key[len("diffusion_"):]
    
    # Map keys (same as convert_to_mlx.py)
    if new_key.startswith("all_final_layer.2-1."):
        new_key = new_key.replace("all_final_layer.2-1.", "final_layer.")
    elif new_key.startswith("all_x_embedder.2-1."):
        new_key = new_key.replace("all_x_embedder.2-1.", "x_embedder.")
    elif "t_embedder.mlp." in new_key:
        new_key = new_key.replace("t_embedder.mlp.", "t_embedder.mlp.layers.")
    
    # Handle adaLN_modulation
    if "adaLN_modulation.0." in new_key:
        new_key = new_key.replace("adaLN_modulation.0.", "adaLN_modulation.")
    if "adaLN_modulation.1." in new_key:
        new_key = new_key.replace("adaLN_modulation.1.", "adaLN_modulation.")
    
    # Handle to_out (but NOT qkv.out which needs different handling)
    if "to_out.0." in new_key:
        new_key = new_key.replace("to_out.0.", "to_out.")
    
    # Handle cap_embedder
    if "cap_embedder.0." in new_key:
        new_key = new_key.replace("cap_embedder.0.", "cap_embedder.layers.0.")
    if "cap_embedder.1." in new_key:
        new_key = new_key.replace("cap_embedder.1.", "cap_embedder.layers.1.")
    
    # Map attention norm names: q_norm -> norm_q, k_norm -> norm_k
    if ".attention.q_norm." in new_key:
        new_key = new_key.replace(".attention.q_norm.", ".attention.norm_q.")
    if ".attention.k_norm." in new_key:
        new_key = new_key.replace(".attention.k_norm.", ".attention.norm_k.")
    
    # Map out -> to_out for attention output projection
    if ".attention.out." in new_key:
        new_key = new_key.replace(".attention.out.", ".attention.to_out.")
    
    return new_key


def convert_text_encoder_key(key: str) -> str:
    """Convert text encoder key from ComfyUI format to MLX format."""
    new_key = key
    
    # Remove text_encoders.qwen3_4b. prefix
    if new_key.startswith("text_encoders.qwen3_4b."):
        new_key = new_key[len("text_encoders.qwen3_4b."):]
    
    # Remove transformer. prefix if present (ComfyUI adds this)
    if new_key.startswith("transformer."):
        new_key = new_key[len("transformer."):]
    
    return new_key


def convert_vae_key(key: str, layers_per_block: int = 2) -> str:
    """Convert VAE key from ComfyUI format to MLX format."""
    import torch  # Only for potential weight transposition
    
    new_key = key
    
    # Remove vae. prefix if present
    if new_key.startswith("vae."):
        new_key = new_key[4:]
    
    parts = new_key.split(".")
    
    # Map down_blocks
    if "down_blocks" in new_key:
        block_idx = int(parts[parts.index("down_blocks") + 1])
        if "resnets" in new_key:
            resnet_idx = int(parts[parts.index("resnets") + 1])
            rest_idx = parts.index("resnets") + 2
            new_parts = parts[:parts.index("resnets")] + ["layers", str(resnet_idx)] + parts[rest_idx:]
            new_key = ".".join(new_parts)
        elif "downsamplers" in new_key:
            rest_idx = parts.index("downsamplers") + 2
            new_parts = parts[:parts.index("downsamplers")] + ["layers", str(layers_per_block)] + parts[rest_idx:]
            new_key = ".".join(new_parts)
    
    # Map up_blocks
    elif "up_blocks" in new_key:
        if "resnets" in new_key:
            resnet_idx = int(parts[parts.index("resnets") + 1])
            rest_idx = parts.index("resnets") + 2
            new_parts = parts[:parts.index("resnets")] + ["layers", str(resnet_idx)] + parts[rest_idx:]
            new_key = ".".join(new_parts)
        elif "upsamplers" in new_key:
            rest_idx = parts.index("upsamplers") + 2
            new_parts = parts[:parts.index("upsamplers")] + ["layers", str(layers_per_block + 1)] + parts[rest_idx:]
            new_key = ".".join(new_parts)
    
    # Map mid_block
    if "mid_block" in new_key:
        if "resnets.0" in new_key:
            new_key = new_key.replace("resnets.0", "layers.0")
        elif "attentions.0" in new_key:
            new_key = new_key.replace("attentions.0", "layers.1")
        elif "resnets.1" in new_key:
            new_key = new_key.replace("resnets.1", "layers.2")
    
    # Map to_out
    if "to_out.0" in new_key:
        new_key = new_key.replace("to_out.0", "to_out")
    
    return new_key


def convert_checkpoint(
    checkpoint_path: str,
    output_path: str,
    reference_model_path: str = None,
    model_name: str = None,
):
    """Convert a ComfyUI checkpoint to MLX format."""
    if mx is None:
        raise RuntimeError("MLX is required for conversion")
    
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    
    if model_name is None:
        model_name = checkpoint_path.stem
    
    output_dir = output_path / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConverting: {checkpoint_path}")
    print(f"Output: {output_dir}")
    
    # Analyze checkpoint
    components = analyze_checkpoint(str(checkpoint_path))
    
    if len(components["transformer"]) == 0:
        raise ValueError("No transformer weights found in checkpoint")
    
    # Load checkpoint using PyTorch framework to handle bfloat16
    print("\nLoading checkpoint...")
    import torch
    with safe_open(str(checkpoint_path), framework="pt") as f:
        all_keys = list(f.keys())
        
        # Convert transformer weights
        print("\nConverting transformer weights...")
        transformer_weights = {}
        qkv_split_count = 0
        skipped_keys = []
        
        for key in tqdm(components["transformer"], desc="Transformer"):
            # Skip keys that don't exist in the MLX model architecture
            if "pad_token" in key:
                continue
            if "norm_final" in key:
                skipped_keys.append(key)
                continue
            
            new_key = convert_transformer_key(key)
            value = f.get_tensor(key)
            
            # Handle fused QKV weights - split into separate Q, K, V
            # ComfyUI format: *.attention.qkv.weight [3*dim, dim]
            if ".attention.qkv.weight" in new_key:
                base_key = new_key.replace(".qkv.weight", "")
                # Split along first axis: [3*dim, dim] -> 3x [dim, dim]
                q, k, v = torch.chunk(value, 3, dim=0)
                transformer_weights[f"{base_key}.to_q.weight"] = mx.array(q.float().half().numpy())
                transformer_weights[f"{base_key}.to_k.weight"] = mx.array(k.float().half().numpy())
                transformer_weights[f"{base_key}.to_v.weight"] = mx.array(v.float().half().numpy())
                qkv_split_count += 1
                continue
            
            # Convert bfloat16 to float16 via float32
            transformer_weights[new_key] = mx.array(value.float().half().numpy())
        
        if qkv_split_count > 0:
            print(f"  Split {qkv_split_count} fused QKV weights into separate Q/K/V")
        if skipped_keys:
            print(f"  Skipped {len(skipped_keys)} unsupported keys: {skipped_keys[:3]}...")
        print(f"  Converted {len(transformer_weights)} transformer weights")
        mx.save_safetensors(str(output_dir / "weights.safetensors"), transformer_weights)
        
        # Convert text encoder weights
        if len(components["text_encoder"]) > 0:
            print("\nConverting text encoder weights...")
            te_weights = {}
            for key in tqdm(components["text_encoder"], desc="Text Encoder"):
                if "logit_scale" in key:
                    continue
                new_key = convert_text_encoder_key(key)
                value = f.get_tensor(key)
                # Convert to float32
                te_weights[new_key] = mx.array(value.float().numpy())
            
            print(f"  Converted {len(te_weights)} text encoder weights")
            mx.save_safetensors(str(output_dir / "text_encoder.safetensors"), te_weights)
        
        # NOTE: ComfyUI VAE format uses completely different key naming than MLX
        # (e.g., decoder.mid.block_1 vs decoder.mid_block.layers.0)
        # Most fine-tunes don't modify the VAE, so we always copy from reference.
        # If VAE keys are present in the checkpoint, just note it.
        if len(components["vae"]) > 0:
            print(f"\nNote: Found {len(components['vae'])} VAE keys in checkpoint.")
            print("  ComfyUI VAE format differs from MLX - will copy from reference model instead.")
    
    # Save configs - copy from reference model if available, otherwise use defaults
    print("\nSaving configs...")
    
    if reference_model_path:
        ref_path = Path(reference_model_path)
        
        # Copy config files from reference (they contain correct architecture details)
        for config_file in ["config.json", "text_encoder_config.json", "vae_config.json"]:
            ref_config = ref_path / config_file
            if ref_config.exists():
                shutil.copy(ref_config, output_dir / config_file)
                print(f"  Copied {config_file} from reference")
            else:
                # Use defaults if reference doesn't have the config
                if config_file == "config.json":
                    with open(output_dir / config_file, "w") as f:
                        json.dump(DEFAULT_TRANSFORMER_CONFIG, f, indent=2)
                elif config_file == "text_encoder_config.json":
                    with open(output_dir / config_file, "w") as f:
                        json.dump(DEFAULT_TEXT_ENCODER_CONFIG, f, indent=2)
                elif config_file == "vae_config.json":
                    with open(output_dir / config_file, "w") as f:
                        json.dump(DEFAULT_VAE_CONFIG, f, indent=2)
    else:
        # No reference - use defaults
        with open(output_dir / "config.json", "w") as f:
            json.dump(DEFAULT_TRANSFORMER_CONFIG, f, indent=2)
        with open(output_dir / "text_encoder_config.json", "w") as f:
            json.dump(DEFAULT_TEXT_ENCODER_CONFIG, f, indent=2)
        with open(output_dir / "vae_config.json", "w") as f:
            json.dump(DEFAULT_VAE_CONFIG, f, indent=2)
    
    # Always copy VAE from reference model (ComfyUI VAE format is incompatible)
    print("\nCopying VAE from reference model...")
    if reference_model_path:
        ref_vae = Path(reference_model_path) / "vae.safetensors"
        if ref_vae.exists():
            shutil.copy(ref_vae, output_dir / "vae.safetensors")
            print(f"  Copied VAE weights from {ref_vae}")
        else:
            print("  WARNING: Reference VAE not found at", ref_vae)
            print("  You'll need to copy vae.safetensors from another MLX model.")
    else:
        print("  WARNING: No reference model specified!")
        print("  You'll need to copy vae.safetensors from another MLX model.")
    
    # Copy tokenizer and scheduler from reference
    if reference_model_path:
        ref_path = Path(reference_model_path)
        
        # Tokenizer
        ref_tokenizer = ref_path / "tokenizer"
        if ref_tokenizer.exists():
            dest_tokenizer = output_dir / "tokenizer"
            if dest_tokenizer.exists():
                shutil.rmtree(dest_tokenizer)
            shutil.copytree(ref_tokenizer, dest_tokenizer)
            print(f"  Copied tokenizer from {ref_tokenizer}")
        
        # Scheduler
        ref_scheduler = ref_path / "scheduler"
        if ref_scheduler.exists():
            dest_scheduler = output_dir / "scheduler"
            if dest_scheduler.exists():
                shutil.rmtree(dest_scheduler)
            shutil.copytree(ref_scheduler, dest_scheduler)
            print(f"  Copied scheduler from {ref_scheduler}")
    else:
        # Create default scheduler config
        scheduler_dir = output_dir / "scheduler"
        scheduler_dir.mkdir(exist_ok=True)
        with open(scheduler_dir / "scheduler_config.json", "w") as f:
            json.dump(DEFAULT_SCHEDULER_CONFIG, f, indent=2)
        print("  Created default scheduler config")
        print("  WARNING: No tokenizer copied. You'll need to copy the tokenizer folder from another MLX model.")
    
    print(f"\n✓ Conversion complete: {output_dir}")
    print("\nNext steps:")
    print("1. Ensure tokenizer/ folder exists (copy from mlx_model if needed)")
    print("2. Ensure vae.safetensors exists (copy from mlx_model if checkpoint didn't include VAE)")
    print("3. Test the model in the app")
    
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert ComfyUI single-file Z-Image-Turbo checkpoint to MLX format"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to ComfyUI .safetensors checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/mlx",
        help="Output directory for MLX models (default: ./models/mlx)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name (default: checkpoint filename without extension)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="./models/mlx/mlx_model",
        help="Reference MLX model to copy VAE/tokenizer/scheduler from",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze the checkpoint, don't convert",
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_checkpoint(args.checkpoint)
    else:
        convert_checkpoint(
            args.checkpoint,
            args.output,
            args.reference,
            args.name,
        )


if __name__ == "__main__":
    main()
