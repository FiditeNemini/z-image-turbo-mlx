"""
Debug ResnetBlock2D step by step
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

print("Input to resnet:")
pt_input_np = pt_input.numpy()
mlx_input_np = np.array(mlx_input).transpose(0, 3, 1, 2)
print(f"  PT: mean={pt_input_np.mean():.6f}, std={pt_input_np.std():.6f}")
print(f"  MLX: mean={mlx_input_np.mean():.6f}, std={mlx_input_np.std():.6f}")
print(f"  diff: max={np.abs(pt_input_np - mlx_input_np).max():.8f}")

# Step 1: norm1
print("\n=== Step 1: norm1 ===")
with torch.no_grad():
    pt_h = pt_resnet.norm1(pt_input)

mlx_h = mlx_resnet.norm1(mlx_input)
mx.eval(mlx_h)

pt_h_np = pt_h.numpy()
mlx_h_np = np.array(mlx_h).transpose(0, 3, 1, 2)
print(f"  PT: mean={pt_h_np.mean():.6f}, std={pt_h_np.std():.6f}")
print(f"  MLX: mean={mlx_h_np.mean():.6f}, std={mlx_h_np.std():.6f}")
print(f"  diff: max={np.abs(pt_h_np - mlx_h_np).max():.8f}")

# Step 2: silu
print("\n=== Step 2: silu ===")
with torch.no_grad():
    pt_h = torch.nn.functional.silu(pt_h)

mlx_h = mx.nn.silu(mlx_h)
mx.eval(mlx_h)

pt_h_np = pt_h.numpy()
mlx_h_np = np.array(mlx_h).transpose(0, 3, 1, 2)
print(f"  PT: mean={pt_h_np.mean():.6f}, std={pt_h_np.std():.6f}")
print(f"  MLX: mean={mlx_h_np.mean():.6f}, std={mlx_h_np.std():.6f}")
print(f"  diff: max={np.abs(pt_h_np - mlx_h_np).max():.8f}")

# Step 3: conv1
print("\n=== Step 3: conv1 ===")
with torch.no_grad():
    pt_h = pt_resnet.conv1(pt_h)

mlx_h = mlx_resnet.conv1(mlx_h)
mx.eval(mlx_h)

pt_h_np = pt_h.numpy()
mlx_h_np = np.array(mlx_h).transpose(0, 3, 1, 2)
print(f"  PT: mean={pt_h_np.mean():.6f}, std={pt_h_np.std():.6f}")
print(f"  MLX: mean={mlx_h_np.mean():.6f}, std={mlx_h_np.std():.6f}")
print(f"  diff: max={np.abs(pt_h_np - mlx_h_np).max():.8f}")

# Step 4: norm2
print("\n=== Step 4: norm2 ===")
with torch.no_grad():
    pt_h = pt_resnet.norm2(pt_h)

mlx_h = mlx_resnet.norm2(mlx_h)
mx.eval(mlx_h)

pt_h_np = pt_h.numpy()
mlx_h_np = np.array(mlx_h).transpose(0, 3, 1, 2)
print(f"  PT: mean={pt_h_np.mean():.6f}, std={pt_h_np.std():.6f}")
print(f"  MLX: mean={mlx_h_np.mean():.6f}, std={mlx_h_np.std():.6f}")
print(f"  diff: max={np.abs(pt_h_np - mlx_h_np).max():.8f}")

# Step 5: silu again
print("\n=== Step 5: silu ===")
with torch.no_grad():
    pt_h = torch.nn.functional.silu(pt_h)

mlx_h = mx.nn.silu(mlx_h)
mx.eval(mlx_h)

pt_h_np = pt_h.numpy()
mlx_h_np = np.array(mlx_h).transpose(0, 3, 1, 2)
print(f"  PT: mean={pt_h_np.mean():.6f}, std={pt_h_np.std():.6f}")
print(f"  MLX: mean={mlx_h_np.mean():.6f}, std={mlx_h_np.std():.6f}")
print(f"  diff: max={np.abs(pt_h_np - mlx_h_np).max():.8f}")

# Step 6: conv2
print("\n=== Step 6: conv2 ===")
with torch.no_grad():
    pt_h = pt_resnet.conv2(pt_h)

mlx_h = mlx_resnet.conv2(mlx_h)
mx.eval(mlx_h)

pt_h_np = pt_h.numpy()
mlx_h_np = np.array(mlx_h).transpose(0, 3, 1, 2)
print(f"  PT: mean={pt_h_np.mean():.6f}, std={pt_h_np.std():.6f}")
print(f"  MLX: mean={mlx_h_np.mean():.6f}, std={mlx_h_np.std():.6f}")
print(f"  diff: max={np.abs(pt_h_np - mlx_h_np).max():.8f}")

# Final: residual
print("\n=== Step 7: residual x + h ===")
with torch.no_grad():
    pt_out = pt_input + pt_h

mlx_out = mlx_input + mlx_h
mx.eval(mlx_out)

pt_out_np = pt_out.numpy()
mlx_out_np = np.array(mlx_out).transpose(0, 3, 1, 2)
print(f"  PT: mean={pt_out_np.mean():.6f}, std={pt_out_np.std():.6f}")
print(f"  MLX: mean={mlx_out_np.mean():.6f}, std={mlx_out_np.std():.6f}")
print(f"  diff: max={np.abs(pt_out_np - mlx_out_np).max():.8f}")

# Let's also check what GroupNorm does with batch statistics
print("\n\n=== Check GroupNorm axis behavior ===")
# PyTorch GroupNorm: normalizes over C dimension, grouped
# MLX GroupNorm: should do the same

# Sample small tensor to verify
pt_test = torch.randn(1, 4, 2, 2)  # [B, C, H, W]
mlx_test = mx.array(pt_test.numpy()).transpose(0, 2, 3, 1)  # [B, H, W, C]

pt_gn = torch.nn.GroupNorm(2, 4)  # 2 groups, 4 channels
mlx_gn = mx.nn.GroupNorm(2, 4)

# Set same weights
with torch.no_grad():
    pt_gn.weight.fill_(1.0)
    pt_gn.bias.fill_(0.0)
mlx_gn.weight = mx.ones_like(mlx_gn.weight)
mlx_gn.bias = mx.zeros_like(mlx_gn.bias)

with torch.no_grad():
    pt_gn_out = pt_gn(pt_test)
    
mlx_gn_out = mlx_gn(mlx_test)
mx.eval(mlx_gn_out)

pt_gn_out_np = pt_gn_out.numpy()
mlx_gn_out_np = np.array(mlx_gn_out).transpose(0, 3, 1, 2)

print(f"PT GroupNorm output:")
print(pt_gn_out_np[0])
print(f"\nMLX GroupNorm output:")
print(mlx_gn_out_np[0])
print(f"\ndiff: {np.abs(pt_gn_out_np - mlx_gn_out_np).max():.8f}")
