"""Debug script to compare MLX Base model with PyTorch"""
import sys
sys.path.insert(0, '/Users/willdee/Documents/Projects/z-image-turbo-mlx')

import torch
import numpy as np
import mlx.core as mx
import json
from pathlib import Path

# Load PyTorch model
print("Loading PyTorch Base model...")
from diffusers import ZImagePipeline
pipe = ZImagePipeline.from_pretrained(
    "models/pytorch/Z-Image",
    torch_dtype=torch.float32,
)

# Load MLX model
print("Loading MLX Base model...")
from src.z_image_mlx import ZImageTransformer2DModel

mlx_model_path = Path("models/mlx/Z-Image")
with open(mlx_model_path / "config.json") as f:
    config = json.load(f)

mlx_model = ZImageTransformer2DModel(config)

# Load weights
weights = mx.load(str(mlx_model_path / "weights.safetensors"))
mlx_model.load_weights(list(weights.items()))
mx.eval(mlx_model.parameters())

# Create test inputs
print("\nCreating test inputs...")
batch_size = 1
channels = 16
height = 128
width = 128

torch.manual_seed(42)
latents_pt = torch.randn(batch_size, channels, 1, height, width)
latents_mx = mx.array(latents_pt.numpy())

# Simple text embedding (zeros for testing)
text_embed_pt = torch.zeros(batch_size, 512, 2560)
text_embed_mx = mx.zeros((batch_size, 512, 2560))

# Test at timestep 0.0 (t=1000 in diffusers scale)
t_pt = torch.tensor([0.0])  # PyTorch uses 0-1 range
t_mx = mx.array([0.0])

print(f"Latents shape: {latents_pt.shape}")
print(f"Text embed shape: {text_embed_pt.shape}")
print(f"Timestep: {t_pt.item()}")

# Run PyTorch model
print("\nRunning PyTorch transformer...")
with torch.no_grad():
    pt_transformer = pipe.transformer
    pt_result = pt_transformer(
        x=latents_pt,
        t=t_pt,
        cap_feats=text_embed_pt,
        return_dict=False,
    )
    print(f"PyTorch result type: {type(pt_result)}")
    if isinstance(pt_result, tuple):
        print(f"Tuple length: {len(pt_result)}, types: {[type(x) for x in pt_result]}")
        pt_out = pt_result[0]
        if isinstance(pt_out, list):
            print(f"List length: {len(pt_out)}, types: {[type(x) for x in pt_out]}")
            pt_out = pt_out[0]
    elif isinstance(pt_result, list):
        print(f"List length: {len(pt_result)}")
        pt_out = pt_result[0]
    else:
        pt_out = pt_result
    print(f"PyTorch output shape: {pt_out.shape}")
    print(f"PyTorch output stats: mean={pt_out.mean().item():.6f}, std={pt_out.std().item():.6f}, min={pt_out.min().item():.6f}, max={pt_out.max().item():.6f}")

# Run MLX model
print("\nRunning MLX transformer...")
text_embed_list = [text_embed_mx[i] for i in range(batch_size)]
mlx_out = mlx_model(latents_mx, t_mx, text_embed_list)
mx.eval(mlx_out)
print(f"MLX output shape: {mlx_out.shape}")
mlx_out_np = np.array(mlx_out)
print(f"MLX output stats: mean={mlx_out_np.mean():.6f}, std={mlx_out_np.std():.6f}, min={mlx_out_np.min():.6f}, max={mlx_out_np.max():.6f}")

# Compare
print("\n=== Comparison ===")
pt_np = pt_out.numpy()
diff = np.abs(pt_np - mlx_out_np)
print(f"Max absolute difference: {diff.max():.6f}")
print(f"Mean absolute difference: {diff.mean():.6f}")

if diff.max() < 0.1:
    print("✓ Models produce similar outputs")
else:
    print("✗ Models produce DIFFERENT outputs - weights may be incorrect")
