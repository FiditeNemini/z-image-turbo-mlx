"""
Debug GroupNorm pytorch_compatible flag
"""
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np

print("=== Test GroupNorm pytorch_compatible flag ===")

# Create test tensor 
torch.manual_seed(42)
pt_input = torch.randn(1, 512, 64, 64)  # [B, C, H, W]
mlx_input = mx.array(pt_input.numpy()).transpose(0, 2, 3, 1)  # [B, H, W, C]

# PyTorch GroupNorm
pt_gn = torch.nn.GroupNorm(32, 512, eps=1e-6)
with torch.no_grad():
    pt_gn.weight.fill_(1.0)
    pt_gn.bias.fill_(0.0)

# MLX GroupNorm without pytorch_compatible (default=False)
mlx_gn_default = nn.GroupNorm(32, 512, eps=1e-6)
mlx_gn_default.weight = mx.ones(512)
mlx_gn_default.bias = mx.zeros(512)

# MLX GroupNorm with pytorch_compatible=True
mlx_gn_compat = nn.GroupNorm(32, 512, eps=1e-6, pytorch_compatible=True)
mlx_gn_compat.weight = mx.ones(512)
mlx_gn_compat.bias = mx.zeros(512)

# Compute outputs
with torch.no_grad():
    pt_out = pt_gn(pt_input)

mlx_out_default = mlx_gn_default(mlx_input)
mlx_out_compat = mlx_gn_compat(mlx_input)
mx.eval(mlx_out_default, mlx_out_compat)

# Convert to numpy
pt_out_np = pt_out.numpy()
mlx_default_np = np.array(mlx_out_default).transpose(0, 3, 1, 2)
mlx_compat_np = np.array(mlx_out_compat).transpose(0, 3, 1, 2)

print(f"\nPyTorch output: mean={pt_out_np.mean():.6f}, std={pt_out_np.std():.6f}")
print(f"MLX default: mean={mlx_default_np.mean():.6f}, std={mlx_default_np.std():.6f}")
print(f"MLX compat: mean={mlx_compat_np.mean():.6f}, std={mlx_compat_np.std():.6f}")

print(f"\nDiff MLX default vs PT: max={np.abs(mlx_default_np - pt_out_np).max():.6f}")
print(f"Diff MLX compat vs PT: max={np.abs(mlx_compat_np - pt_out_np).max():.6f}")

print("\nFirst 5 values comparison:")
print(f"  PT: {pt_out_np[0, :5, 0, 0]}")
print(f"  MLX default: {mlx_default_np[0, :5, 0, 0]}")
print(f"  MLX compat: {mlx_compat_np[0, :5, 0, 0]}")
