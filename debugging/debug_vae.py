"""
Debug VAE specifically
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

# PyTorch decode
with torch.no_grad():
    pt_decoded = pipe.vae.decode(test_latent).sample
    
print(f"PT VAE output: shape={pt_decoded.shape}, mean={pt_decoded.mean().item():.4f}, std={pt_decoded.std().item():.4f}")

# MLX decode
# Need to transpose: [B, C, H, W] -> [B, H, W, C]
mlx_latent = mx.array(test_latent.numpy())
mlx_latent = mlx_latent.transpose(0, 2, 3, 1)  # [1, 64, 64, 16]

mlx_decoded = mlx_vae.decode(mlx_latent)
mx.eval(mlx_decoded)

print(f"MLX VAE output: shape={mlx_decoded.shape}, mean={float(mlx_decoded.mean()):.4f}, std={float(mlx_decoded.std()):.4f}")

# Compare
pt_np = pt_decoded.numpy()  # [1, 3, 512, 512]
mlx_np = np.array(mlx_decoded)  # [1, 512, 512, 3]
mlx_np = mlx_np.transpose(0, 3, 1, 2)  # [1, 3, 512, 512]

diff = np.abs(pt_np - mlx_np)
print(f"VAE diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

# Debug conv_in
print("\n=== Debug conv_in ===")
pt_conv_in_w = pipe.vae.decoder.conv_in.weight.detach().numpy()  # [Out, In, K, K]
pt_conv_in_b = pipe.vae.decoder.conv_in.bias.detach().numpy()

mlx_conv_in_w = np.array(mlx_vae.decoder.conv_in.weight)  # Should be [Out, K, K, In] for MLX
mlx_conv_in_b = np.array(mlx_vae.decoder.conv_in.bias)

print(f"PT conv_in weight shape: {pt_conv_in_w.shape}")
print(f"MLX conv_in weight shape: {mlx_conv_in_w.shape}")

# MLX Conv2d expects [Out, Kh, Kw, In]
# PyTorch uses [Out, In, Kh, Kw]
# We need to transpose (0, 2, 3, 1) to convert PT -> MLX

# Check if they match after transpose
pt_conv_in_w_transposed = pt_conv_in_w.transpose(0, 2, 3, 1)  # [Out, K, K, In]
conv_w_diff = np.abs(pt_conv_in_w_transposed - mlx_conv_in_w)
print(f"Conv weight diff (after transpose): max={conv_w_diff.max():.6f}")

# Check bias
conv_b_diff = np.abs(pt_conv_in_b - mlx_conv_in_b)
print(f"Conv bias diff: max={conv_b_diff.max():.6f}")

# Check conv_in output
print("\n=== Debug conv_in output ===")
with torch.no_grad():
    pt_conv_in_out = pipe.vae.decoder.conv_in(test_latent)
    
mlx_conv_in_out = mlx_vae.decoder.conv_in(mlx_latent)
mx.eval(mlx_conv_in_out)

pt_conv_out_np = pt_conv_in_out.numpy()  # [1, 512, 64, 64]
mlx_conv_out_np = np.array(mlx_conv_in_out)  # [1, 64, 64, 512]
mlx_conv_out_np = mlx_conv_out_np.transpose(0, 3, 1, 2)  # [1, 512, 64, 64]

conv_out_diff = np.abs(pt_conv_out_np - mlx_conv_out_np)
print(f"PT conv_in output: mean={pt_conv_out_np.mean():.4f}, std={pt_conv_out_np.std():.4f}")
print(f"MLX conv_in output: mean={mlx_conv_out_np.mean():.4f}, std={mlx_conv_out_np.std():.4f}")
print(f"conv_in output diff: max={conv_out_diff.max():.6f}, mean={conv_out_diff.mean():.6f}")

# Check mid_block output
print("\n=== Debug mid_block output ===")
with torch.no_grad():
    pt_mid_out = pipe.vae.decoder.mid_block(pt_conv_in_out)
    
# Need to run mid_block on MLX
mlx_mid_out = mlx_vae.decoder.mid_block(mlx_conv_in_out)
mx.eval(mlx_mid_out)

pt_mid_np = pt_mid_out.numpy()
mlx_mid_np = np.array(mlx_mid_out).transpose(0, 3, 1, 2)

mid_diff = np.abs(pt_mid_np - mlx_mid_np)
print(f"PT mid output: mean={pt_mid_np.mean():.4f}, std={pt_mid_np.std():.4f}")
print(f"MLX mid output: mean={mlx_mid_np.mean():.4f}, std={mlx_mid_np.std():.4f}")
print(f"mid_block output diff: max={mid_diff.max():.6f}, mean={mid_diff.mean():.6f}")
