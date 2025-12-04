#!/usr/bin/env python3
"""
Convert ComfyUI single-file Z-Image-Turbo checkpoint to PyTorch/Diffusers format.

This is a direct conversion that skips the MLX intermediate step.
ComfyUI checkpoints have all components in one file with prefixes:
- diffusion_* : Transformer weights
- text_encoders.qwen3_4b.* : Text encoder weights  
- vae.* : VAE weights (may or may not be present)

Precision options:
- Original: Keep as bfloat16 (default PyTorch training precision)
- FP16: Convert to float16 (smaller, compatible with most hardware)
- FP8: Convert to float8_e4m3fn (smallest, requires PyTorch 2.1+ and compatible hardware)
"""

import argparse
import os
import json
import shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file as torch_save_safetensors
from tqdm import tqdm
import torch


def get_target_dtype(precision: str):
    """Get the target PyTorch dtype based on precision string.
    
    Args:
        precision: "Original", "FP16", or "FP8"
        
    Returns:
        tuple: (torch.dtype for weight storage, actual precision string used)
        
    Note: FP8 requires PyTorch 2.1+ and compatible hardware for inference.
    Models saved in FP8 are ~50% smaller than FP16/BF16.
    Diffusers will upcast FP8 weights to the compute dtype at load time.
    """
    if precision == "FP8":
        if hasattr(torch, 'float8_e4m3fn'):
            return torch.float8_e4m3fn, "FP8"
        else:
            print("  Warning: FP8 not available in this PyTorch version, using FP16")
            return torch.float16, "FP16"
    elif precision == "FP16":
        return torch.float16, "FP16"
    else:  # Original
        return torch.bfloat16, "Original"


# Reference configs for Z-Image-Turbo architecture (Diffusers format)
DEFAULT_TRANSFORMER_CONFIG = {
    "_class_name": "ZImageTransformer2DModel",
    "_diffusers_version": "0.36.0.dev0",
    "all_patch_size": [2],
    "attention_bias": True,
    "axes_dims": [32, 48, 48],
    "axes_lens": [1536, 512, 512],
    "cap_feat_dim": 2560,
    "dim": 3840,
    "dtype": "bfloat16",
    "guidance_embeds": False,
    "in_channels": 16,
    "mlp_ratio": 8.0,
    "n_heads": 30,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "out_channels": 16,
    "patch_size": 2,
    "qk_norm": "rms_norm",
    "rope_theta": 256.0,
    "text_modulation": True,
}

DEFAULT_VAE_CONFIG = {
    "_class_name": "AutoencoderKL",
    "_diffusers_version": "0.33.0.dev0",
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 16,
    "latents_mean": None,
    "latents_std": None,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
    "use_post_quant_conv": False,
    "use_quant_conv": False,
}

DEFAULT_TEXT_ENCODER_CONFIG = {
    "_name_or_path": "Qwen/Qwen3-4B",
    "architectures": ["Qwen3Model"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9728,
    "max_position_embeddings": 40960,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000.0,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
    "transformers_version": "4.51.0.dev0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
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

MODEL_INDEX = {
    "_class_name": "ZImagePipeline",
    "_diffusers_version": "0.36.0.dev0",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3Model"],
    "tokenizer": ["transformers", "Qwen2Tokenizer"],
    "transformer": ["diffusers", "ZImageTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"],
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
    """Convert transformer key from ComfyUI format to Diffusers format.
    
    Key mappings:
    - model.diffusion_model. prefix -> removed
    - diffusion_ prefix -> removed
    - final_layer. -> all_final_layer.2-1.
    - x_embedder. -> all_x_embedder.2-1.
    - attention.out. -> attention.to_out.0.
    - attention.q_norm. -> attention.norm_q.
    - attention.k_norm. -> attention.norm_k.
    - attention.qkv.weight -> split into to_q/to_k/to_v (done separately)
    """
    new_key = key
    
    # Remove model.diffusion_model. prefix (ComfyUI format)
    if new_key.startswith("model.diffusion_model."):
        new_key = new_key[len("model.diffusion_model."):]
    
    # Remove diffusion_ prefix
    if new_key.startswith("diffusion_"):
        new_key = new_key[len("diffusion_"):]
    
    # Map ComfyUI keys to Diffusers format
    # ComfyUI: final_layer. -> Diffusers: all_final_layer.2-1.
    # ComfyUI: x_embedder. -> Diffusers: all_x_embedder.2-1.
    if new_key.startswith("final_layer."):
        new_key = new_key.replace("final_layer.", "all_final_layer.2-1.")
    elif new_key.startswith("x_embedder."):
        new_key = new_key.replace("x_embedder.", "all_x_embedder.2-1.")
    
    # ComfyUI attention format to Diffusers:
    # out.weight -> to_out.0.weight
    # q_norm -> norm_q
    # k_norm -> norm_k
    if ".attention.out.weight" in new_key:
        new_key = new_key.replace(".attention.out.weight", ".attention.to_out.0.weight")
    if ".attention.out.bias" in new_key:
        new_key = new_key.replace(".attention.out.bias", ".attention.to_out.0.bias")
    if ".attention.q_norm." in new_key:
        new_key = new_key.replace(".attention.q_norm.", ".attention.norm_q.")
    if ".attention.k_norm." in new_key:
        new_key = new_key.replace(".attention.k_norm.", ".attention.norm_k.")
    
    return new_key


def convert_text_encoder_key(key: str) -> str:
    """Convert text encoder key from ComfyUI format to Diffusers format."""
    new_key = key
    
    # Remove text_encoders.qwen3_4b. prefix
    if new_key.startswith("text_encoders.qwen3_4b."):
        new_key = new_key[len("text_encoders.qwen3_4b."):]
    
    # Remove transformer. prefix if present (ComfyUI adds this)
    if new_key.startswith("transformer."):
        new_key = new_key[len("transformer."):]
    
    return new_key


def convert_checkpoint(
    checkpoint_path: str,
    output_path: str,
    reference_model_path: str = None,
    model_name: str = None,
    precision: str = "FP16",
):
    """Convert a ComfyUI checkpoint to PyTorch/Diffusers format.
    
    Args:
        checkpoint_path: Path to the ComfyUI .safetensors checkpoint
        output_path: Output directory for converted model
        reference_model_path: Path to reference model for VAE/tokenizer/configs
        model_name: Model name (default: checkpoint filename)
        precision: "Original" (bfloat16), "FP16" (float16), or "FP8" (float8_e4m3fn)
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    
    if model_name is None:
        model_name = checkpoint_path.stem
    
    output_dir = output_path / model_name
    
    print(f"\nConverting: {checkpoint_path}")
    print(f"Output: {output_dir}")
    
    # Create output structure
    (output_dir / "transformer").mkdir(parents=True, exist_ok=True)
    (output_dir / "vae").mkdir(parents=True, exist_ok=True)
    (output_dir / "text_encoder").mkdir(parents=True, exist_ok=True)
    (output_dir / "scheduler").mkdir(parents=True, exist_ok=True)
    (output_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    
    # Analyze checkpoint
    components = analyze_checkpoint(str(checkpoint_path))
    
    if len(components["transformer"]) == 0:
        raise ValueError("No transformer weights found in checkpoint")
    
    # Determine target dtype for weights
    target_dtype, actual_precision = get_target_dtype(precision)
    print(f"  Target precision: {precision} -> {target_dtype}")
    
    # Load and convert weights
    print("\nLoading checkpoint...")
    with safe_open(str(checkpoint_path), framework="pt") as f:
        
        # Convert transformer weights
        print("\nConverting transformer weights...")
        transformer_weights = {}
        skipped_keys = []
        qkv_split_count = 0
        
        for key in tqdm(components["transformer"], desc="Transformer"):
            # Skip unsupported keys
            if "pad_token" in key:
                continue
            if "norm_final" in key:
                skipped_keys.append(key)
                continue
            
            new_key = convert_transformer_key(key)
            value = f.get_tensor(key)
            
            # Handle fused QKV weights - Diffusers expects separate to_q, to_k, to_v
            # ComfyUI format: *.attention.qkv.weight [3*dim, dim]
            if ".attention.qkv.weight" in new_key:
                base_key = new_key.replace(".qkv.weight", "")
                # Split along first axis: [3*dim, dim] -> 3x [dim, dim]
                q, k, v = torch.chunk(value, 3, dim=0)
                transformer_weights[f"{base_key}.to_q.weight"] = q.to(target_dtype)
                transformer_weights[f"{base_key}.to_k.weight"] = k.to(target_dtype)
                transformer_weights[f"{base_key}.to_v.weight"] = v.to(target_dtype)
                qkv_split_count += 1
                continue
            
            # Convert to target precision
            transformer_weights[new_key] = value.to(target_dtype)
        
        if qkv_split_count > 0:
            print(f"  Split {qkv_split_count} fused QKV weights into separate Q/K/V")
        if skipped_keys:
            print(f"  Skipped {len(skipped_keys)} unsupported keys")
        print(f"  Converted {len(transformer_weights)} transformer weights")
        
        # Save transformer
        torch_save_safetensors(
            transformer_weights, 
            str(output_dir / "transformer" / "diffusion_pytorch_model.safetensors")
        )
        
        # Convert text encoder weights (if present)
        if len(components["text_encoder"]) > 0:
            print("\nConverting text encoder weights...")
            te_weights = {}
            
            for key in tqdm(components["text_encoder"], desc="Text Encoder"):
                new_key = convert_text_encoder_key(key)
                value = f.get_tensor(key)
                # Text encoder: keep at FP16 for FP8 mode (embedding ops need higher precision)
                # This is similar to keeping VAE at FP16 for MLX
                if actual_precision == "FP8":
                    te_weights[new_key] = value.to(torch.float16)
                else:
                    te_weights[new_key] = value.to(target_dtype)
            
            te_precision = "FP16" if actual_precision == "FP8" else actual_precision
            print(f"  Converted {len(te_weights)} text encoder weights ({te_precision})")
            torch_save_safetensors(
                te_weights, 
                str(output_dir / "text_encoder" / "model.safetensors")
            )
        
        # Note about VAE
        if len(components["vae"]) > 0:
            print(f"\nNote: Found {len(components['vae'])} VAE keys in checkpoint.")
            print("  ComfyUI VAE format differs from Diffusers - will copy from reference model.")
    
    # Save configs
    print("\nSaving configs...")
    
    if reference_model_path:
        ref_path = Path(reference_model_path)
        
        # Copy configs from reference model
        for subdir, config_name in [
            ("transformer", "config.json"),
            ("vae", "config.json"),
            ("text_encoder", "config.json"),
        ]:
            ref_config = ref_path / subdir / config_name
            if ref_config.exists():
                shutil.copy(ref_config, output_dir / subdir / config_name)
                print(f"  Copied {subdir}/{config_name} from reference")
            else:
                # Use defaults
                if subdir == "transformer":
                    with open(output_dir / subdir / config_name, "w") as cfg_f:
                        json.dump(DEFAULT_TRANSFORMER_CONFIG, cfg_f, indent=2)
                elif subdir == "vae":
                    with open(output_dir / subdir / config_name, "w") as cfg_f:
                        json.dump(DEFAULT_VAE_CONFIG, cfg_f, indent=2)
                elif subdir == "text_encoder":
                    with open(output_dir / subdir / config_name, "w") as cfg_f:
                        json.dump(DEFAULT_TEXT_ENCODER_CONFIG, cfg_f, indent=2)
    else:
        # Use defaults
        with open(output_dir / "transformer" / "config.json", "w") as f:
            json.dump(DEFAULT_TRANSFORMER_CONFIG, f, indent=2)
        with open(output_dir / "vae" / "config.json", "w") as f:
            json.dump(DEFAULT_VAE_CONFIG, f, indent=2)
        with open(output_dir / "text_encoder" / "config.json", "w") as f:
            json.dump(DEFAULT_TEXT_ENCODER_CONFIG, f, indent=2)
    
    # Copy VAE weights from reference model
    print("\nCopying VAE from reference model...")
    if reference_model_path:
        ref_vae = Path(reference_model_path) / "vae" / "diffusion_pytorch_model.safetensors"
        if ref_vae.exists():
            shutil.copy(ref_vae, output_dir / "vae" / "diffusion_pytorch_model.safetensors")
            print(f"  Copied VAE weights from {ref_vae}")
        else:
            print("  WARNING: Reference VAE not found!")
            print("  You'll need to copy vae/diffusion_pytorch_model.safetensors from another model.")
    else:
        print("  WARNING: No reference model specified!")
        print("  You'll need to copy VAE from another Diffusers model.")
    
    # Copy tokenizer from reference
    if reference_model_path:
        ref_tokenizer = Path(reference_model_path) / "tokenizer"
        if ref_tokenizer.exists():
            dest_tokenizer = output_dir / "tokenizer"
            if dest_tokenizer.exists():
                shutil.rmtree(dest_tokenizer)
            shutil.copytree(ref_tokenizer, dest_tokenizer)
            print(f"  Copied tokenizer from reference")
    
    # Scheduler config
    scheduler_config = output_dir / "scheduler" / "scheduler_config.json"
    if reference_model_path:
        ref_scheduler = Path(reference_model_path) / "scheduler" / "scheduler_config.json"
        if ref_scheduler.exists():
            shutil.copy(ref_scheduler, scheduler_config)
            print("  Copied scheduler config from reference")
        else:
            with open(scheduler_config, "w") as f:
                json.dump(DEFAULT_SCHEDULER_CONFIG, f, indent=2)
    else:
        with open(scheduler_config, "w") as f:
            json.dump(DEFAULT_SCHEDULER_CONFIG, f, indent=2)
    
    # Create model_index.json
    with open(output_dir / "model_index.json", "w") as f:
        json.dump(MODEL_INDEX, f, indent=4)
    
    print(f"\n✓ Conversion complete: {output_dir}")
    print("\nThe model can now be loaded with:")
    print(f"  from diffusers import ZImagePipeline")
    print(f"  pipe = ZImagePipeline.from_pretrained('{output_dir}')")
    
    return str(output_dir)


def convert_comfyui_to_pytorch(
    checkpoint_path: str,
    output_path: str,
    precision: str = "FP16",
    reference_model_path: str = None,
):
    """Convenience wrapper for converting ComfyUI checkpoint to PyTorch format.
    
    This is the main entry point used by app.py for single-file imports.
    
    Args:
        checkpoint_path: Path to the ComfyUI .safetensors checkpoint
        output_path: Output directory for converted model
        precision: "Original", "FP16", or "FP8"
        reference_model_path: Optional path to reference model for VAE/tokenizer
    """
    # Default to Z-Image-Turbo reference if available
    if reference_model_path is None:
        default_ref = Path("./models/pytorch/Z-Image-Turbo")
        if default_ref.exists():
            reference_model_path = str(default_ref)
    
    # Extract model name from output path
    output_path = Path(output_path)
    model_name = output_path.name
    parent_path = output_path.parent
    
    return convert_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=str(parent_path),
        reference_model_path=reference_model_path,
        model_name=model_name,
        precision=precision,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert ComfyUI single-file Z-Image-Turbo checkpoint to PyTorch/Diffusers format"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to ComfyUI .safetensors checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/pytorch",
        help="Output directory for converted model (default: ./models/pytorch)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference PyTorch/Diffusers Z-Image-Turbo model for VAE/tokenizer/configs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name (default: checkpoint filename without extension)",
    )
    
    args = parser.parse_args()
    
    convert_checkpoint(
        args.checkpoint,
        args.output,
        args.reference,
        args.name,
    )


if __name__ == "__main__":
    main()
