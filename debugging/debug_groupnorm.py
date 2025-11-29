"""
Debug GroupNorm in detail
"""
import torch
import mlx.core as mx
import mlx.nn as nn
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

# Create test latent
torch.manual_seed(42)
test_latent = torch.randn(1, 16, 64, 64)

# Convert for MLX
mlx_latent = mx.array(test_latent.numpy())
mlx_latent = mlx_latent.transpose(0, 2, 3, 1)  # [1, 64, 64, 16]

# Get conv_in output
with torch.no_grad():
    pt_input = pipe.vae.decoder.conv_in(test_latent)
    
mlx_input = mlx_vae.decoder.conv_in(mlx_latent)
mx.eval(mlx_input)

pt_resnet = pipe.vae.decoder.mid_block.resnets[0]
mlx_resnet = mlx_vae.decoder.mid_block.layers[0]

print("Input to GroupNorm:")
pt_input_np = pt_input.numpy()
mlx_input_np = np.array(mlx_input).transpose(0, 3, 1, 2)
print(f"  PT: mean={pt_input_np.mean():.6f}, std={pt_input_np.std():.6f}")
print(f"  MLX: mean={mlx_input_np.mean():.6f}, std={mlx_input_np.std():.6f}")
print(f"  input diff: max={np.abs(pt_input_np - mlx_input_np).max():.8f}")

# GroupNorm test
print("\n=== GroupNorm test ===")
print(f"PT GroupNorm: {pt_resnet.norm1}")
print(f"MLX GroupNorm: {mlx_resnet.norm1}")

# Check GroupNorm weights
print("\nGroupNorm weights:")
pt_w = pt_resnet.norm1.weight.detach().numpy()
pt_b = pt_resnet.norm1.bias.detach().numpy()
mlx_w = np.array(mlx_resnet.norm1.weight)
mlx_b = np.array(mlx_resnet.norm1.bias)
print(f"  PT weight shape: {pt_w.shape}, MLX weight shape: {mlx_w.shape}")
print(f"  weight diff: {np.abs(pt_w - mlx_w).max():.8f}")
print(f"  bias diff: {np.abs(pt_b - mlx_b).max():.8f}")

# Test GroupNorm with identical inputs
print("\n=== GroupNorm outputs ===")
with torch.no_grad():
    pt_norm_out = pt_resnet.norm1(pt_input)

mlx_norm_out = mlx_resnet.norm1(mlx_input)
mx.eval(mlx_norm_out)

pt_norm_np = pt_norm_out.numpy()
mlx_norm_np = np.array(mlx_norm_out).transpose(0, 3, 1, 2)

print(f"PT GroupNorm output: mean={pt_norm_np.mean():.6f}, std={pt_norm_np.std():.6f}")
print(f"MLX GroupNorm output: mean={mlx_norm_np.mean():.6f}, std={mlx_norm_np.std():.6f}")
print(f"diff: max={np.abs(pt_norm_np - mlx_norm_np).max():.6f}, mean={np.abs(pt_norm_np - mlx_norm_np).mean():.6f}")

# Detailed check on first few values
print("\nFirst 5 output values at [0, 0, 0, :]:")
print(f"  PT: {pt_norm_np[0, :5, 0, 0]}")
print(f"  MLX: {mlx_norm_np[0, :5, 0, 0]}")

# Check if MLX GroupNorm is computing over correct axes
print("\n=== Checking MLX GroupNorm implementation ===")
# MLX GroupNorm expects NHWC, PyTorch expects NCHW
# MLX's GroupNorm should normalize over the last axis (channels)

# Manual check - PyTorch groups along C dimension
# For 32 groups with 512 channels: each group has 16 channels
num_groups = 32
num_channels = 512
group_size = num_channels // num_groups

print(f"\nGroup setup: {num_groups} groups, {group_size} channels per group")

# Look at statistics within first group (channels 0-15)
pt_group0 = pt_input_np[0, :16, :, :]  # [16, 64, 64]
mlx_transposed = np.array(mlx_input)  # [1, 64, 64, 512]
mlx_group0 = mlx_transposed[0, :, :, :16]  # [64, 64, 16]
mlx_group0 = mlx_group0.transpose(2, 0, 1)  # [16, 64, 64] to match PT

print(f"\nFirst group stats:")
print(f"  PT group0: mean={pt_group0.mean():.6f}, std={pt_group0.std():.6f}")
print(f"  MLX group0: mean={mlx_group0.mean():.6f}, std={mlx_group0.std():.6f}")

# Now check if the issue is in GroupNorm computation
# MLX GroupNorm uses axis=-1, so input should be NHWC
print("\n\n=== Test simple GroupNorm case ===")
# Create simple test
pt_simple = torch.randn(1, 4, 2, 2)  # [B=1, C=4, H=2, W=2]
mlx_simple = mx.array(pt_simple.numpy()).transpose(0, 2, 3, 1)  # [1, 2, 2, 4] - NHWC

pt_gn = torch.nn.GroupNorm(2, 4, eps=1e-6)  # 2 groups, 4 channels
mlx_gn = nn.GroupNorm(2, 4, eps=1e-6, pytorch_compatible=True)  # Check if there's a compatibility flag

# Set identical weights
with torch.no_grad():
    pt_gn.weight.fill_(1.0)
    pt_gn.bias.fill_(0.0)
mlx_gn.weight = mx.ones(4)
mlx_gn.bias = mx.zeros(4)

with torch.no_grad():
    pt_simple_out = pt_gn(pt_simple)
mlx_simple_out = mlx_gn(mlx_simple)
mx.eval(mlx_simple_out)

pt_simple_out_np = pt_simple_out.numpy()  # [1, 4, 2, 2]
mlx_simple_out_np = np.array(mlx_simple_out).transpose(0, 3, 1, 2)  # [1, 4, 2, 2]

print(f"PT simple input:\n{pt_simple.numpy()[0]}")
print(f"\nPT simple output:\n{pt_simple_out_np[0]}")
print(f"\nMLX simple output:\n{mlx_simple_out_np[0]}")
print(f"\ndiff: max={np.abs(pt_simple_out_np - mlx_simple_out_np).max():.6f}")
