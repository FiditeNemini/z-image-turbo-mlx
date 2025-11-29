import torch
import mlx.core as mx
import numpy as np
from diffusers import ZImagePipeline

def main():
    print("Loading PyTorch Model...")
    pt_path = "models/Z-Image-Turbo"
    pipe = ZImagePipeline.from_pretrained(pt_path)
    
    print("Loading MLX Weights...")
    mlx_weights = mx.load("models/mlx_model/weights.safetensors")
    
    # Check cap_embedder weights
    # PT: cap_embedder.0.weight, cap_embedder.1.weight, cap_embedder.1.bias
    # MLX: cap_embedder.layers.0.weight, cap_embedder.layers.1.weight, cap_embedder.layers.1.bias
    # Wait, MLX implementation uses nn.Sequential?
    # In z_image_mlx.py:
    # self.cap_embedder = nn.Sequential(...)
    # MLX nn.Sequential keys are layers.0, layers.1 etc.
    
    pt_norm_w = pipe.transformer.cap_embedder[0].weight.detach().numpy()
    pt_lin_w = pipe.transformer.cap_embedder[1].weight.detach().numpy()
    pt_lin_b = pipe.transformer.cap_embedder[1].bias.detach().numpy()
    
    # MLX keys
    # Check what keys are in mlx_weights
    keys = list(mlx_weights.keys())
    cap_keys = [k for k in keys if "cap_embedder" in k]
    print("MLX cap_embedder keys:", cap_keys)
    
    # Assuming standard mapping
    # MLX Linear weight is (Out, In) same as PT?
    # MLX Linear weight shape is (Out, In) in recent versions? 
    # Wait, I found earlier that MLX Linear weight is (Out, In).
    # Let's check shapes.
    
    if "cap_embedder.layers.0.weight" in mlx_weights:
        mx_norm_w = mlx_weights["cap_embedder.layers.0.weight"]
        mx_lin_w = mlx_weights["cap_embedder.layers.1.weight"]
        mx_lin_b = mlx_weights["cap_embedder.layers.1.bias"]
    else:
        # Maybe mapped differently?
        # Let's try to find them.
        mx_norm_w = mlx_weights.get("cap_embedder.0.weight")
        mx_lin_w = mlx_weights.get("cap_embedder.1.weight")
        mx_lin_b = mlx_weights.get("cap_embedder.1.bias")

    if mx_norm_w is None:
        print("Could not find MLX weights")
        return

    print(f"PT Norm Shape: {pt_norm_w.shape}")
    print(f"MLX Norm Shape: {mx_norm_w.shape}")
    diff = np.abs(pt_norm_w - np.array(mx_norm_w)).max()
    print(f"Norm Weight Max Diff: {diff}")
    
    print(f"PT Linear Weight Shape: {pt_lin_w.shape}")
    print(f"MLX Linear Weight Shape: {mx_lin_w.shape}")
    
    # MLX Linear might be transposed if I transposed it during conversion
    # PT Linear is (Out, In)
    # MLX Linear is (Out, In)
    
    diff = np.abs(pt_lin_w - np.array(mx_lin_w)).max()
    print(f"Linear Weight Max Diff: {diff}")
    
    if pt_lin_w.shape != mx_lin_w.shape:
        print("Shape mismatch! Checking transpose...")
        diff_T = np.abs(pt_lin_w - np.array(mx_lin_w).T).max()
        print(f"Linear Weight Transposed Max Diff: {diff_T}")

    print(f"PT Linear Bias Shape: {pt_lin_b.shape}")
    print(f"MLX Linear Bias Shape: {mx_lin_b.shape}")
    diff = np.abs(pt_lin_b - np.array(mx_lin_b)).max()
    print(f"Linear Bias Max Diff: {diff}")

if __name__ == "__main__":
    main()
