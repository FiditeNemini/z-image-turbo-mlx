#!/usr/bin/env python3
"""
Convert MLX Z-Image-Turbo model to PyTorch/Diffusers format.

This allows MLX models to be used with the standard diffusers pipeline
or shared with others who don't have Apple Silicon.
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
import numpy as np


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
    "intermediate_size": 13696,
    "max_position_embeddings": 40960,
    "model_type": "qwen3",
    "num_attention_heads": 20,
    "num_hidden_layers": 36,
    "num_key_value_heads": 4,
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

MODEL_INDEX = {
    "_class_name": "ZImagePipeline",
    "_diffusers_version": "0.36.0.dev0",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3Model"],
    "tokenizer": ["transformers", "Qwen2Tokenizer"],
    "transformer": ["diffusers", "ZImageTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}


def convert_transformer_key_to_pytorch(key: str) -> str:
    """Convert MLX transformer key to PyTorch/Diffusers format."""
    new_key = key
    
    # Reverse mappings from MLX to PyTorch
    # to_q/to_k/to_v -> these need to be fused back to qkv (handled separately)
    # to_out -> to_out.0 (PyTorch uses Sequential)
    if ".attention.to_out.weight" in new_key:
        new_key = new_key.replace(".attention.to_out.weight", ".attention.to_out.0.weight")
    if ".attention.to_out.bias" in new_key:
        new_key = new_key.replace(".attention.to_out.bias", ".attention.to_out.0.bias")
    
    # norm_q -> norm_q (same in diffusers)
    # norm_k -> norm_k (same in diffusers)
    
    # cap_embedder.layers.X -> cap_embedder.X
    if "cap_embedder.layers.0." in new_key:
        new_key = new_key.replace("cap_embedder.layers.0.", "cap_embedder.0.")
    if "cap_embedder.layers.1." in new_key:
        new_key = new_key.replace("cap_embedder.layers.1.", "cap_embedder.1.")
    
    # t_embedder.mlp.layers. -> t_embedder.mlp.
    if "t_embedder.mlp.layers." in new_key:
        new_key = new_key.replace("t_embedder.mlp.layers.", "t_embedder.mlp.")
    
    # adaLN_modulation. -> adaLN_modulation.1.
    # Note: PyTorch has SiLU at index 0, Linear at index 1
    if "adaLN_modulation.weight" in new_key:
        new_key = new_key.replace("adaLN_modulation.weight", "adaLN_modulation.1.weight")
    if "adaLN_modulation.bias" in new_key:
        new_key = new_key.replace("adaLN_modulation.bias", "adaLN_modulation.1.bias")
    
    return new_key


def convert_vae_key_to_pytorch(key: str, layers_per_block: int = 2) -> str:
    """Convert MLX VAE key to PyTorch/Diffusers format."""
    new_key = key
    parts = new_key.split(".")
    
    # Map layers.X back to resnets/downsamplers/upsamplers
    if "down_blocks" in new_key and ".layers." in new_key:
        block_idx = int(parts[parts.index("down_blocks") + 1])
        layer_idx = int(parts[parts.index("layers") + 1])
        rest_idx = parts.index("layers") + 2
        
        if layer_idx < layers_per_block:
            # It's a resnet
            new_parts = parts[:parts.index("layers")] + ["resnets", str(layer_idx)] + parts[rest_idx:]
        else:
            # It's a downsampler
            new_parts = parts[:parts.index("layers")] + ["downsamplers", "0"] + parts[rest_idx:]
        new_key = ".".join(new_parts)
    
    elif "up_blocks" in new_key and ".layers." in new_key:
        layer_idx = int(parts[parts.index("layers") + 1])
        rest_idx = parts.index("layers") + 2
        
        if layer_idx <= layers_per_block:
            # It's a resnet
            new_parts = parts[:parts.index("layers")] + ["resnets", str(layer_idx)] + parts[rest_idx:]
        else:
            # It's an upsampler
            new_parts = parts[:parts.index("layers")] + ["upsamplers", "0"] + parts[rest_idx:]
        new_key = ".".join(new_parts)
    
    # Map mid_block layers back
    if "mid_block" in new_key and ".layers." in new_key:
        layer_idx = int(parts[parts.index("layers") + 1])
        rest_idx = parts.index("layers") + 2
        
        if layer_idx == 0:
            new_parts = parts[:parts.index("layers")] + ["resnets", "0"] + parts[rest_idx:]
        elif layer_idx == 1:
            new_parts = parts[:parts.index("layers")] + ["attentions", "0"] + parts[rest_idx:]
        elif layer_idx == 2:
            new_parts = parts[:parts.index("layers")] + ["resnets", "1"] + parts[rest_idx:]
        new_key = ".".join(new_parts)
    
    # Map to_out back to to_out.0
    if "to_out.weight" in new_key and "to_out.0" not in new_key:
        new_key = new_key.replace("to_out.weight", "to_out.0.weight")
    if "to_out.bias" in new_key and "to_out.0" not in new_key:
        new_key = new_key.replace("to_out.bias", "to_out.0.bias")
    
    return new_key


def convert_mlx_to_pytorch(
    mlx_model_path: str,
    output_path: str,
    model_name: str = None,
):
    """Convert an MLX model to PyTorch/Diffusers format."""
    mlx_path = Path(mlx_model_path)
    output_path = Path(output_path)
    
    if model_name is None:
        model_name = mlx_path.name
    
    output_dir = output_path / model_name
    
    print(f"\nConverting MLX to PyTorch/Diffusers")
    print(f"Source: {mlx_path}")
    print(f"Output: {output_dir}")
    
    # Create output structure
    (output_dir / "transformer").mkdir(parents=True, exist_ok=True)
    (output_dir / "vae").mkdir(parents=True, exist_ok=True)
    (output_dir / "text_encoder").mkdir(parents=True, exist_ok=True)
    
    # Convert Transformer weights
    print("\nConverting transformer weights...")
    weights_path = mlx_path / "weights.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    with safe_open(str(weights_path), framework="numpy") as f:
        mlx_keys = list(f.keys())
        
        # Collect Q, K, V weights for fusion
        qkv_groups = {}  # base_key -> {q, k, v}
        other_weights = {}
        
        for key in tqdm(mlx_keys, desc="Processing"):
            value = f.get_tensor(key)
            
            # Check if this is a Q/K/V weight that needs fusion
            if ".attention.to_q.weight" in key:
                base_key = key.replace(".to_q.weight", "")
                if base_key not in qkv_groups:
                    qkv_groups[base_key] = {}
                qkv_groups[base_key]["q"] = torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
            elif ".attention.to_k.weight" in key:
                base_key = key.replace(".to_k.weight", "")
                if base_key not in qkv_groups:
                    qkv_groups[base_key] = {}
                qkv_groups[base_key]["k"] = torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
            elif ".attention.to_v.weight" in key:
                base_key = key.replace(".to_v.weight", "")
                if base_key not in qkv_groups:
                    qkv_groups[base_key] = {}
                qkv_groups[base_key]["v"] = torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
            else:
                # Regular weight - convert key and store
                new_key = convert_transformer_key_to_pytorch(key)
                other_weights[new_key] = torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
        
        # Fuse Q, K, V weights
        pytorch_weights = {}
        for base_key, qkv in qkv_groups.items():
            if "q" in qkv and "k" in qkv and "v" in qkv:
                fused = torch.cat([qkv["q"], qkv["k"], qkv["v"]], dim=0)
                pytorch_weights[f"{base_key}.qkv.weight"] = fused
            else:
                # Incomplete group - keep separate
                if "q" in qkv:
                    pytorch_weights[f"{base_key}.to_q.weight"] = qkv["q"]
                if "k" in qkv:
                    pytorch_weights[f"{base_key}.to_k.weight"] = qkv["k"]
                if "v" in qkv:
                    pytorch_weights[f"{base_key}.to_v.weight"] = qkv["v"]
        
        pytorch_weights.update(other_weights)
        print(f"  Fused {len(qkv_groups)} QKV groups")
        print(f"  Total: {len(pytorch_weights)} weights")
    
    # Save transformer weights (may need to shard for large models)
    print("  Saving transformer weights...")
    torch_save_safetensors(pytorch_weights, str(output_dir / "transformer" / "diffusion_pytorch_model.safetensors"))
    
    # Save transformer config
    with open(output_dir / "transformer" / "config.json", "w") as f:
        json.dump(DEFAULT_TRANSFORMER_CONFIG, f, indent=2)
    
    # Convert VAE weights
    print("\nConverting VAE weights...")
    vae_path = mlx_path / "vae.safetensors"
    if vae_path.exists():
        with safe_open(str(vae_path), framework="numpy") as f:
            vae_weights = {}
            for key in tqdm(f.keys(), desc="VAE"):
                value = f.get_tensor(key)
                new_key = convert_vae_key_to_pytorch(key)
                
                # Transpose conv weights back: [Out, H, W, In] -> [Out, In, H, W]
                if "conv" in new_key and "weight" in new_key and len(value.shape) == 4:
                    value = np.transpose(value, (0, 3, 1, 2))
                
                vae_weights[new_key] = torch.from_numpy(value.astype(np.float32))
        
        torch_save_safetensors(vae_weights, str(output_dir / "vae" / "diffusion_pytorch_model.safetensors"))
        with open(output_dir / "vae" / "config.json", "w") as f:
            json.dump(DEFAULT_VAE_CONFIG, f, indent=2)
        print(f"  Converted {len(vae_weights)} VAE weights")
    else:
        print("  No VAE found, skipping")
    
    # Convert Text Encoder weights
    print("\nConverting text encoder weights...")
    te_path = mlx_path / "text_encoder.safetensors"
    if te_path.exists():
        with safe_open(str(te_path), framework="numpy") as f:
            te_weights = {}
            for key in tqdm(f.keys(), desc="Text Encoder"):
                value = f.get_tensor(key)
                te_weights[key] = torch.from_numpy(value.astype(np.float32))
        
        # Text encoder may be large - save as sharded
        torch_save_safetensors(te_weights, str(output_dir / "text_encoder" / "model.safetensors"))
        with open(output_dir / "text_encoder" / "config.json", "w") as f:
            json.dump(DEFAULT_TEXT_ENCODER_CONFIG, f, indent=2)
        print(f"  Converted {len(te_weights)} text encoder weights")
    else:
        print("  No text encoder found, skipping")
    
    # Copy tokenizer and scheduler
    print("\nCopying tokenizer and scheduler...")
    tokenizer_src = mlx_path / "tokenizer"
    if tokenizer_src.exists():
        tokenizer_dst = output_dir / "tokenizer"
        if tokenizer_dst.exists():
            shutil.rmtree(tokenizer_dst)
        shutil.copytree(tokenizer_src, tokenizer_dst)
        print("  Copied tokenizer")
    
    scheduler_src = mlx_path / "scheduler"
    if scheduler_src.exists():
        scheduler_dst = output_dir / "scheduler"
        if scheduler_dst.exists():
            shutil.rmtree(scheduler_dst)
        shutil.copytree(scheduler_src, scheduler_dst)
        print("  Copied scheduler")
    
    # Create model_index.json
    with open(output_dir / "model_index.json", "w") as f:
        json.dump(MODEL_INDEX, f, indent=4)
    
    print(f"\n✓ Conversion complete: {output_dir}")
    print("\nThe model can now be loaded with:")
    print(f"  pipe = ZImagePipeline.from_pretrained('{output_dir}')")
    
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MLX Z-Image-Turbo model to PyTorch/Diffusers format"
    )
    parser.add_argument(
        "mlx_model",
        type=str,
        help="Path to MLX model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/pytorch",
        help="Output directory for PyTorch model (default: ./models/pytorch)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name (default: same as source folder name)",
    )
    
    args = parser.parse_args()
    
    convert_mlx_to_pytorch(
        args.mlx_model,
        args.output,
        args.name,
    )


if __name__ == "__main__":
    main()
