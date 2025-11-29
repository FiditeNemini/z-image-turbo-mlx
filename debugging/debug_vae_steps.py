"""
Debug VAE mid_block step by step
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
    pt_conv_in_out = pipe.vae.decoder.conv_in(test_latent)
    
mlx_conv_in_out = mlx_vae.decoder.conv_in(mlx_latent)
mx.eval(mlx_conv_in_out)

print("After conv_in:")
print(f"  PT shape: {pt_conv_in_out.shape}, MLX shape: {mlx_conv_in_out.shape}")

# Check how PyTorch mid_block processes
# UNetMidBlock2D forward: resnets[0] -> attentions[0] -> resnets[1]
print("\n=== PyTorch mid_block order ===")
print("  1. resnets[0]")
print("  2. attentions[0]")
print("  3. resnets[1]")

# My MLX mid_block is: Sequential([ResnetBlock, Attention, ResnetBlock])
# This should match!

# Step through manually
print("\n=== Step through mid_block ===")

# Step 1: ResnetBlock 0
with torch.no_grad():
    pt_step1 = pipe.vae.decoder.mid_block.resnets[0](pt_conv_in_out, temb=None)

mlx_step1 = mlx_vae.decoder.mid_block.layers[0](mlx_conv_in_out, temb=None)
mx.eval(mlx_step1)

pt_step1_np = pt_step1.numpy()
mlx_step1_np = np.array(mlx_step1).transpose(0, 3, 1, 2)

step1_diff = np.abs(pt_step1_np - mlx_step1_np)
print(f"After resnet[0]:")
print(f"  PT: mean={pt_step1_np.mean():.4f}, std={pt_step1_np.std():.4f}")
print(f"  MLX: mean={mlx_step1_np.mean():.4f}, std={mlx_step1_np.std():.4f}")
print(f"  diff: max={step1_diff.max():.6f}, mean={step1_diff.mean():.6f}")

# Step 2: Attention
with torch.no_grad():
    pt_step2 = pipe.vae.decoder.mid_block.attentions[0](pt_step1)

mlx_step2 = mlx_vae.decoder.mid_block.layers[1](mlx_step1)
mx.eval(mlx_step2)

pt_step2_np = pt_step2.numpy()
mlx_step2_np = np.array(mlx_step2).transpose(0, 3, 1, 2)

step2_diff = np.abs(pt_step2_np - mlx_step2_np)
print(f"\nAfter attention:")
print(f"  PT: mean={pt_step2_np.mean():.4f}, std={pt_step2_np.std():.4f}")
print(f"  MLX: mean={mlx_step2_np.mean():.4f}, std={mlx_step2_np.std():.4f}")
print(f"  diff: max={step2_diff.max():.6f}, mean={step2_diff.mean():.6f}")

# Step 3: ResnetBlock 1
with torch.no_grad():
    pt_step3 = pipe.vae.decoder.mid_block.resnets[1](pt_step2, temb=None)

mlx_step3 = mlx_vae.decoder.mid_block.layers[2](mlx_step2, temb=None)
mx.eval(mlx_step3)

pt_step3_np = pt_step3.numpy()
mlx_step3_np = np.array(mlx_step3).transpose(0, 3, 1, 2)

step3_diff = np.abs(pt_step3_np - mlx_step3_np)
print(f"\nAfter resnet[1]:")
print(f"  PT: mean={pt_step3_np.mean():.4f}, std={pt_step3_np.std():.4f}")
print(f"  MLX: mean={mlx_step3_np.mean():.4f}, std={mlx_step3_np.std():.4f}")
print(f"  diff: max={step3_diff.max():.6f}, mean={step3_diff.mean():.6f}")

# So where does my mid_block.layers[1] (Attention) call ResnetBlock2D or expect it?
print("\n=== Check MLX mid_block layers ===")
for i, layer in enumerate(mlx_vae.decoder.mid_block.layers):
    print(f"  Layer {i}: {type(layer).__name__}")
