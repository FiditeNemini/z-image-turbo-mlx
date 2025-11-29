"""
Debug VAE mid_block attention weights
"""
import torch
import mlx.core as mx
import numpy as np
from diffusers import ZImagePipeline
from vae import AutoencoderKL
import json

print("Loading models...")
pipe = ZImagePipeline.from_pretrained("models/Z-Image-Turbo", torch_dtype=torch.float32)

with open("models/mlx_model/vae_config.json") as f:
    vae_config = json.load(f)
mlx_vae = AutoencoderKL(vae_config)
vae_weights = mx.load("models/mlx_model/vae.safetensors")
mlx_vae.load_weights(list(vae_weights.items()), strict=False)
mlx_vae.eval()

# Check the structure of PyTorch mid_block
print("\n=== PyTorch mid_block structure ===")
print(pipe.vae.decoder.mid_block)

# Check attention weights
print("\n=== PyTorch attention weight names ===")
for name, param in pipe.vae.decoder.mid_block.attentions[0].named_parameters():
    print(f"  {name}: {param.shape}")

print("\n=== MLX attention attributes ===")
attn = mlx_vae.decoder.mid_block.layers[1]  # The Attention module
print(f"  to_q.weight: {attn.to_q.weight.shape}")
print(f"  to_k.weight: {attn.to_k.weight.shape}")
print(f"  to_v.weight: {attn.to_v.weight.shape}")
print(f"  to_out.weight: {attn.to_out.weight.shape}")

# Check what the PT names look like
print("\n=== Checking weight correspondence ===")
pt_attn = pipe.vae.decoder.mid_block.attentions[0]

# PT uses to_q, to_k, to_v, to_out projections
pt_q = pt_attn.to_q.weight.detach().numpy()  # [out, in]
pt_k = pt_attn.to_k.weight.detach().numpy()
pt_v = pt_attn.to_v.weight.detach().numpy()
pt_out = pt_attn.to_out[0].weight.detach().numpy()  # to_out is a ModuleList

mlx_q = np.array(attn.to_q.weight)
mlx_k = np.array(attn.to_k.weight)
mlx_v = np.array(attn.to_v.weight)
mlx_out = np.array(attn.to_out.weight)

print(f"PT q shape: {pt_q.shape}, MLX q shape: {mlx_q.shape}")
print(f"PT k shape: {pt_k.shape}, MLX k shape: {mlx_k.shape}")
print(f"PT v shape: {pt_v.shape}, MLX v shape: {mlx_v.shape}")
print(f"PT out shape: {pt_out.shape}, MLX out shape: {mlx_out.shape}")

# For MLX nn.Linear, weights are [out, in]
# For PT nn.Linear, weights are [out, in]
# They should match directly

q_diff = np.abs(pt_q - mlx_q).max()
k_diff = np.abs(pt_k - mlx_k).max()
v_diff = np.abs(pt_v - mlx_v).max()
out_diff = np.abs(pt_out - mlx_out).max()

print(f"q weight diff: {q_diff:.6f}")
print(f"k weight diff: {k_diff:.6f}")
print(f"v weight diff: {v_diff:.6f}")
print(f"out weight diff: {out_diff:.6f}")

# Check group_norm weights
print("\n=== Group norm weights ===")
pt_gn = pt_attn.group_norm
mlx_gn = attn.group_norm

pt_gn_w = pt_gn.weight.detach().numpy()
pt_gn_b = pt_gn.bias.detach().numpy()
mlx_gn_w = np.array(mlx_gn.weight)
mlx_gn_b = np.array(mlx_gn.bias)

print(f"GN weight diff: {np.abs(pt_gn_w - mlx_gn_w).max():.6f}")
print(f"GN bias diff: {np.abs(pt_gn_b - mlx_gn_b).max():.6f}")

# Check what the VAE safetensors file actually contains
print("\n=== Check VAE safetensors keys ===")
vae_keys = sorted([k for k in vae_weights.keys() if 'mid_block' in k and 'attn' in k])
print("Attention-related keys:")
for k in vae_keys[:20]:
    print(f"  {k}")
