#!/usr/bin/env python3
"""
Convert PyTorch/Diffusers Z-Image-Turbo model to ComfyUI single-file checkpoint.

This creates a single .safetensors file that can be used directly in ComfyUI.
The output format uses these prefixes:
- diffusion_* : Transformer weights
- text_encoders.qwen3_4b.* : Text encoder weights  
- vae.* : VAE weights
"""

import argparse
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file as torch_save_safetensors
from tqdm import tqdm
import torch


def convert_transformer_key_to_comfyui(key: str) -> str:
    """Convert Diffusers transformer key to ComfyUI format."""
    new_key = key
    
    # Map Diffusers keys to ComfyUI format
    # Diffusers: all_final_layer.2-1. -> ComfyUI: final_layer.
    # Diffusers: all_x_embedder.2-1. -> ComfyUI: x_embedder.
    if new_key.startswith("all_final_layer.2-1."):
        new_key = new_key.replace("all_final_layer.2-1.", "final_layer.")
    elif new_key.startswith("all_x_embedder.2-1."):
        new_key = new_key.replace("all_x_embedder.2-1.", "x_embedder.")
    
    # Add model.diffusion_model. prefix (ComfyUI format)
    new_key = f"model.diffusion_model.{new_key}"
    
    # Map attention keys:
    # to_out.0.weight -> out.weight
    # norm_q -> q_norm
    # norm_k -> k_norm
    # qkv.weight stays as qkv.weight
    if ".attention.to_out.0.weight" in new_key:
        new_key = new_key.replace(".attention.to_out.0.weight", ".attention.out.weight")
    if ".attention.to_out.0.bias" in new_key:
        new_key = new_key.replace(".attention.to_out.0.bias", ".attention.out.bias")
    if ".attention.norm_q." in new_key:
        new_key = new_key.replace(".attention.norm_q.", ".attention.q_norm.")
    if ".attention.norm_k." in new_key:
        new_key = new_key.replace(".attention.norm_k.", ".attention.k_norm.")
    
    return new_key


def convert_text_encoder_key_to_comfyui(key: str) -> str:
    """Convert Diffusers text encoder key to ComfyUI format."""
    # Add ComfyUI prefix
    return f"text_encoders.qwen3_4b.transformer.{key}"


def convert_vae_key_to_comfyui(key: str, layers_per_block: int = 2) -> str:
    """Convert Diffusers VAE key to ComfyUI format.
    
    ComfyUI VAE uses different naming:
    - decoder.mid_block.resnets.0 -> decoder.mid.block_1
    - decoder.mid_block.attentions.0 -> decoder.mid.attn_1
    - decoder.up_blocks.X.resnets.Y -> decoder.up.X.block.Y
    - decoder.up_blocks.X.upsamplers.0 -> decoder.up.X.upsample
    - decoder.conv_norm_out -> decoder.norm_out
    - group_norm -> norm
    - to_q/to_k/to_v -> q/k/v
    - to_out.0 -> proj_out
    - conv_shortcut -> nin_shortcut
    """
    new_key = key
    parts = new_key.split(".")
    
    # Handle mid_block
    if "mid_block" in new_key:
        # mid_block.resnets.0 -> mid.block_1
        # mid_block.resnets.1 -> mid.block_2
        # mid_block.attentions.0 -> mid.attn_1
        if ".resnets.0." in new_key:
            new_key = new_key.replace("mid_block.resnets.0.", "mid.block_1.")
        elif ".resnets.1." in new_key:
            new_key = new_key.replace("mid_block.resnets.1.", "mid.block_2.")
        elif ".attentions.0." in new_key:
            new_key = new_key.replace("mid_block.attentions.0.", "mid.attn_1.")
            # Also fix attention-specific keys
            new_key = new_key.replace(".group_norm.", ".norm.")
            new_key = new_key.replace(".to_q.", ".q.")
            new_key = new_key.replace(".to_k.", ".k.")
            new_key = new_key.replace(".to_v.", ".v.")
            new_key = new_key.replace(".to_out.0.", ".proj_out.")
            new_key = new_key.replace(".to_out.", ".proj_out.")
    
    # Handle up_blocks
    elif "up_blocks" in new_key:
        block_idx = int(parts[parts.index("up_blocks") + 1])
        # Reverse block index: up_blocks.0 -> up.3, up_blocks.1 -> up.2, etc.
        reversed_idx = 3 - block_idx
        
        if ".resnets." in new_key:
            resnet_idx = int(parts[parts.index("resnets") + 1])
            rest_idx = parts.index("resnets") + 2
            # up_blocks.X.resnets.Y -> up.reversed_X.block.Y
            new_parts = ["decoder", "up", str(reversed_idx), "block", str(resnet_idx)] + parts[rest_idx:]
            new_key = ".".join(new_parts)
        elif ".upsamplers." in new_key:
            rest_idx = parts.index("upsamplers") + 2
            # up_blocks.X.upsamplers.0 -> up.reversed_X.upsample
            new_parts = ["decoder", "up", str(reversed_idx), "upsample"] + parts[rest_idx:]
            new_key = ".".join(new_parts)
    
    # Handle down_blocks  
    elif "down_blocks" in new_key:
        block_idx = int(parts[parts.index("down_blocks") + 1])
        
        if ".resnets." in new_key:
            resnet_idx = int(parts[parts.index("resnets") + 1])
            rest_idx = parts.index("resnets") + 2
            new_parts = ["encoder", "down", str(block_idx), "block", str(resnet_idx)] + parts[rest_idx:]
            new_key = ".".join(new_parts)
        elif ".downsamplers." in new_key:
            rest_idx = parts.index("downsamplers") + 2
            new_parts = ["encoder", "down", str(block_idx), "downsample"] + parts[rest_idx:]
            new_key = ".".join(new_parts)
    
    # Handle conv_norm_out -> norm_out
    if "conv_norm_out" in new_key:
        new_key = new_key.replace("conv_norm_out", "norm_out")
    
    # Handle conv_shortcut -> nin_shortcut
    if "conv_shortcut" in new_key:
        new_key = new_key.replace("conv_shortcut", "nin_shortcut")
    
    # Add vae. prefix
    new_key = f"vae.{new_key}"
    
    return new_key


def convert_pytorch_to_comfyui(
    pytorch_model_path: str,
    output_path: str,
    model_name: str = None,
):
    """Convert a PyTorch/Diffusers model to ComfyUI single-file format."""
    pytorch_path = Path(pytorch_model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if model_name is None:
        model_name = pytorch_path.name
    
    output_file = output_path / f"{model_name}.safetensors"
    
    print(f"\nConverting PyTorch/Diffusers to ComfyUI")
    print(f"Source: {pytorch_path}")
    print(f"Output: {output_file}")
    
    all_weights = {}
    
    # Convert Transformer weights
    print("\nConverting transformer weights...")
    transformer_path = pytorch_path / "transformer"
    transformer_files = list(transformer_path.glob("*.safetensors"))
    
    if not transformer_files:
        raise FileNotFoundError(f"No transformer weights found in {transformer_path}")
    
    # First pass: collect all weights and identify Q/K/V groups for fusion
    raw_weights = {}
    for tf_file in transformer_files:
        with safe_open(str(tf_file), framework="pt") as f:
            for key in tqdm(f.keys(), desc=f"Loading ({tf_file.name})"):
                raw_weights[key] = f.get_tensor(key)
    
    # Identify Q/K/V groups and fuse them
    qkv_groups = {}  # base_key -> {q, k, v}
    other_keys = []
    
    for key in raw_weights.keys():
        if ".attention.to_q.weight" in key:
            base_key = key.replace(".to_q.weight", "")
            if base_key not in qkv_groups:
                qkv_groups[base_key] = {}
            qkv_groups[base_key]["q"] = key
        elif ".attention.to_k.weight" in key:
            base_key = key.replace(".to_k.weight", "")
            if base_key not in qkv_groups:
                qkv_groups[base_key] = {}
            qkv_groups[base_key]["k"] = key
        elif ".attention.to_v.weight" in key:
            base_key = key.replace(".to_v.weight", "")
            if base_key not in qkv_groups:
                qkv_groups[base_key] = {}
            qkv_groups[base_key]["v"] = key
        else:
            other_keys.append(key)
    
    # Fuse Q, K, V weights
    fused_count = 0
    for base_key, qkv_keys in qkv_groups.items():
        if "q" in qkv_keys and "k" in qkv_keys and "v" in qkv_keys:
            q = raw_weights[qkv_keys["q"]]
            k = raw_weights[qkv_keys["k"]]
            v = raw_weights[qkv_keys["v"]]
            fused = torch.cat([q, k, v], dim=0)
            comfyui_key = convert_transformer_key_to_comfyui(f"{base_key}.qkv.weight")
            all_weights[comfyui_key] = fused
            fused_count += 1
        else:
            # Incomplete group - convert separately
            for qkv_type, orig_key in qkv_keys.items():
                new_key = convert_transformer_key_to_comfyui(orig_key)
                all_weights[new_key] = raw_weights[orig_key]
    
    # Convert other transformer weights
    for key in other_keys:
        new_key = convert_transformer_key_to_comfyui(key)
        all_weights[new_key] = raw_weights[key]
    
    if fused_count > 0:
        print(f"  Fused {fused_count} Q/K/V groups into QKV weights")
    print(f"  Converted {len([k for k in all_weights if k.startswith('model.diffusion_model.')])} transformer weights")
    
    # Convert Text Encoder weights
    print("\nConverting text encoder weights...")
    te_path = pytorch_path / "text_encoder"
    te_files = list(te_path.glob("*.safetensors"))
    
    if te_files:
        for te_file in te_files:
            with safe_open(str(te_file), framework="pt") as f:
                for key in tqdm(f.keys(), desc=f"Text Encoder ({te_file.name})"):
                    value = f.get_tensor(key)
                    new_key = convert_text_encoder_key_to_comfyui(key)
                    all_weights[new_key] = value
        print(f"  Converted {len([k for k in all_weights if k.startswith('text_encoders.')])} text encoder weights")
    else:
        print("  No text encoder weights found, skipping")
    
    # Convert VAE weights
    print("\nConverting VAE weights...")
    vae_path = pytorch_path / "vae"
    vae_files = list(vae_path.glob("*.safetensors"))
    
    if vae_files:
        for vae_file in vae_files:
            with safe_open(str(vae_file), framework="pt") as f:
                for key in tqdm(f.keys(), desc=f"VAE ({vae_file.name})"):
                    value = f.get_tensor(key)
                    new_key = convert_vae_key_to_comfyui(key)
                    all_weights[new_key] = value
        print(f"  Converted {len([k for k in all_weights if k.startswith('vae.')])} VAE weights")
    else:
        print("  No VAE weights found, skipping")
    
    # Save combined checkpoint
    print(f"\nSaving combined checkpoint ({len(all_weights)} total weights)...")
    torch_save_safetensors(all_weights, str(output_file))
    
    # Calculate file size
    file_size = output_file.stat().st_size / (1024 * 1024 * 1024)
    print(f"  File size: {file_size:.2f} GB")
    
    print(f"\n✓ Conversion complete: {output_file}")
    print("\nTo use in ComfyUI:")
    print(f"  1. Copy {output_file.name} to ComfyUI/models/checkpoints/")
    print("  2. Use the Z-Image-Turbo workflow in ComfyUI")
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch/Diffusers Z-Image-Turbo model to ComfyUI single-file checkpoint"
    )
    parser.add_argument(
        "pytorch_model",
        type=str,
        help="Path to PyTorch/Diffusers model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/comfyui",
        help="Output directory for ComfyUI checkpoint (default: ./models/comfyui)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name for output file (default: same as source folder name)",
    )
    
    args = parser.parse_args()
    
    convert_pytorch_to_comfyui(
        args.pytorch_model,
        args.output,
        args.name,
    )


if __name__ == "__main__":
    main()
