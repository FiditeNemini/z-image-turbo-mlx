"""
Debug VAE ResnetBlock2D weights
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

# Compare resnet[0] weights
print("\n=== ResnetBlock2D[0] weights ===")
pt_resnet = pipe.vae.decoder.mid_block.resnets[0]
mlx_resnet = mlx_vae.decoder.mid_block.layers[0]

# norm1
print("norm1:")
pt_norm1_w = pt_resnet.norm1.weight.detach().numpy()
pt_norm1_b = pt_resnet.norm1.bias.detach().numpy()
mlx_norm1_w = np.array(mlx_resnet.norm1.weight)
mlx_norm1_b = np.array(mlx_resnet.norm1.bias)
print(f"  weight diff: {np.abs(pt_norm1_w - mlx_norm1_w).max():.6f}")
print(f"  bias diff: {np.abs(pt_norm1_b - mlx_norm1_b).max():.6f}")

# conv1 - this needs transposition
print("conv1:")
pt_conv1_w = pt_resnet.conv1.weight.detach().numpy()  # [Out, In, Kh, Kw]
pt_conv1_b = pt_resnet.conv1.bias.detach().numpy()
mlx_conv1_w = np.array(mlx_resnet.conv1.weight)  # [Out, Kh, Kw, In]
mlx_conv1_b = np.array(mlx_resnet.conv1.bias)

print(f"  PT conv1 shape: {pt_conv1_w.shape}")
print(f"  MLX conv1 shape: {mlx_conv1_w.shape}")

# PT [Out, In, Kh, Kw] -> MLX [Out, Kh, Kw, In]
pt_conv1_transposed = pt_conv1_w.transpose(0, 2, 3, 1)
print(f"  weight diff (after transpose): {np.abs(pt_conv1_transposed - mlx_conv1_w).max():.6f}")
print(f"  bias diff: {np.abs(pt_conv1_b - mlx_conv1_b).max():.6f}")

# norm2
print("norm2:")
pt_norm2_w = pt_resnet.norm2.weight.detach().numpy()
pt_norm2_b = pt_resnet.norm2.bias.detach().numpy()
mlx_norm2_w = np.array(mlx_resnet.norm2.weight)
mlx_norm2_b = np.array(mlx_resnet.norm2.bias)
print(f"  weight diff: {np.abs(pt_norm2_w - mlx_norm2_w).max():.6f}")
print(f"  bias diff: {np.abs(pt_norm2_b - mlx_norm2_b).max():.6f}")

# conv2
print("conv2:")
pt_conv2_w = pt_resnet.conv2.weight.detach().numpy()
pt_conv2_b = pt_resnet.conv2.bias.detach().numpy()
mlx_conv2_w = np.array(mlx_resnet.conv2.weight)
mlx_conv2_b = np.array(mlx_resnet.conv2.bias)

pt_conv2_transposed = pt_conv2_w.transpose(0, 2, 3, 1)
print(f"  weight diff (after transpose): {np.abs(pt_conv2_transposed - mlx_conv2_w).max():.6f}")
print(f"  bias diff: {np.abs(pt_conv2_b - mlx_conv2_b).max():.6f}")

# Now let's check what keys are in the VAE safetensors
print("\n=== VAE safetensors keys for mid_block.resnets ===")
vae_keys = sorted([k for k in vae_weights.keys() if 'mid_block' in k and 'resnet' in k])
for k in vae_keys[:30]:
    print(f"  {k}")

# Check if layers[0] is getting the right weights
print("\n=== Expected vs Actual weight paths ===")
print("Expected path: decoder.mid_block.resnets.0.norm1.weight")
print("Actual path in MLX model: decoder.mid_block.layers.0.norm1.weight")
print("Weight file has 'resnets' but model expects 'layers'!")

# Check all mid_block keys in safetensors
print("\n=== All mid_block safetensors keys ===")
mid_keys = sorted([k for k in vae_weights.keys() if 'mid_block' in k])
for k in mid_keys:
    print(f"  {k}")
