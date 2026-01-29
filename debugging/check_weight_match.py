"""Check if MLX weights match PyTorch weights"""
import torch
import numpy as np
from diffusers import ZImagePipeline
import mlx.core as mx

print("Loading PyTorch Base model...")
pipe = ZImagePipeline.from_pretrained(
    "models/pytorch/Z-Image",
    torch_dtype=torch.float32,
)

# Check specific weight
w = pipe.transformer.state_dict()['all_x_embedder.2-1.weight'].numpy()
print(f"\nPyTorch all_x_embedder.2-1.weight (before transpose):")
print(f"  shape={w.shape}, dtype={w.dtype}")
print(f"  mean={w.mean():.6f}, std={w.std():.6f}")
print(f"  min={w.min():.6f}, max={w.max():.6f}")

# After transpose (what conversion should do for Linear layers)
w_t = w.T
print(f"\nPyTorch all_x_embedder.2-1.weight (after transpose):")
print(f"  shape={w_t.shape}, dtype={w_t.dtype}")
print(f"  mean={w_t.mean():.6f}, std={w_t.std():.6f}")
print(f"  min={w_t.min():.6f}, max={w_t.max():.6f}")

# Compare with MLX weight
base_weights = mx.load("models/mlx/Z-Image-New/weights.safetensors")
mlx_w = np.array(base_weights['x_embedder.weight'])
print(f"\nMLX x_embedder.weight:")
print(f"  shape={mlx_w.shape}, dtype={mlx_w.dtype}")
print(f"  mean={mlx_w.mean():.6f}, std={mlx_w.std():.6f}")
print(f"  min={mlx_w.min():.6f}, max={mlx_w.max():.6f}")

# Check if they match
print(f"\n=== Comparison ===")
if mlx_w.shape == w.shape:
    diff = np.abs(mlx_w.astype(np.float32) - w.astype(np.float32)).max()
    print(f"MLX matches PyTorch (no transpose): max diff = {diff:.6f}")
elif mlx_w.shape == w_t.shape:
    diff = np.abs(mlx_w.astype(np.float32) - w_t.astype(np.float32)).max()
    print(f"MLX matches PyTorch (with transpose): max diff = {diff:.6f}")
else:
    print(f"Shape mismatch! MLX={mlx_w.shape}, PT={w.shape}, PT.T={w_t.shape}")
