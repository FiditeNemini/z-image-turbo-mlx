#!/usr/bin/env python3
"""
Test script for true FP8/8-bit quantization with MLX.

Goal: Save weights in quantized format to reduce file size by ~50%
(8-bit vs 16-bit = half the storage)
"""

import mlx.core as mx
from pathlib import Path
import json

def quantize_weights_file(input_path: str, output_path: str, bits: int = 8, group_size: int = 64):
    """
    Quantize a safetensors file and save in MLX's quantized format.
    
    MLX quantization stores:
    - wq: quantized weights (uint8 for 8-bit, uint4 for 4-bit)
    - scales: per-group scales
    - biases: per-group biases (for affine mode)
    
    The quantized format uses naming convention:
    - original.weight -> original.weight (quantized uint8)
    - original.scales (float16)
    - original.biases (float16)
    """
    print(f"Loading {input_path}...")
    weights = mx.load(input_path)
    
    quantized_weights = {}
    stats = {"quantized": 0, "kept": 0, "total_params": 0}
    
    for key, value in weights.items():
        stats["total_params"] += value.size
        
        # Only quantize 2D+ weight tensors with compatible dimensions
        # group_size must divide the last dimension
        can_quantize = (
            len(value.shape) >= 2 and 
            value.shape[-1] >= group_size and 
            value.shape[-1] % group_size == 0 and
            "weight" in key  # Only quantize weight matrices, not biases
        )
        
        if can_quantize:
            try:
                # Quantize with affine mode (includes bias for better accuracy)
                wq, scales, biases = mx.quantize(value, group_size=group_size, bits=bits)
                
                # Store quantized weights with MLX's naming convention
                quantized_weights[key] = wq
                quantized_weights[key.replace(".weight", ".scales")] = scales
                quantized_weights[key.replace(".weight", ".biases")] = biases
                
                stats["quantized"] += 1
            except Exception as e:
                print(f"  Warning: Could not quantize {key}: {e}")
                quantized_weights[key] = value
                stats["kept"] += 1
        else:
            # Keep as-is (biases, small tensors, etc.)
            quantized_weights[key] = value
            stats["kept"] += 1
    
    print(f"Quantized {stats['quantized']} tensors, kept {stats['kept']} as-is")
    print(f"Saving to {output_path}...")
    mx.save_safetensors(output_path, quantized_weights)
    
    # Compare sizes
    input_size = Path(input_path).stat().st_size
    output_size = Path(output_path).stat().st_size
    ratio = output_size / input_size
    print(f"Size: {input_size / 1e9:.2f} GB -> {output_size / 1e9:.2f} GB ({ratio:.1%})")
    
    return stats


def test_dequantize(quantized_path: str, original_path: str, sample_key: str = None):
    """Test that we can dequantize and get reasonable values back."""
    print(f"\nTesting dequantization...")
    
    orig = mx.load(original_path)
    quant = mx.load(quantized_path)
    
    # Find a quantized weight to test
    if sample_key is None:
        for key in orig.keys():
            if "weight" in key and key in quant and key.replace(".weight", ".scales") in quant:
                sample_key = key
                break
    
    if sample_key and sample_key in quant:
        scales_key = sample_key.replace(".weight", ".scales")
        biases_key = sample_key.replace(".weight", ".biases")
        
        if scales_key in quant and biases_key in quant:
            wq = quant[sample_key]
            scales = quant[scales_key]
            biases = quant[biases_key]
            
            # Dequantize
            dequantized = mx.dequantize(wq, scales, biases, group_size=64, bits=8)
            original = orig[sample_key]
            
            # Compare
            diff = mx.abs(dequantized - original)
            max_diff = mx.max(diff).item()
            mean_diff = mx.mean(diff).item()
            
            print(f"Sample key: {sample_key}")
            print(f"  Shape: {original.shape}")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  Original range: [{mx.min(original).item():.4f}, {mx.max(original).item():.4f}]")
        else:
            print(f"  Scales/biases not found for {sample_key}")
    else:
        print(f"  No suitable quantized weight found to test")


if __name__ == "__main__":
    test_dir = Path("models/mlx/ZITCenter-FP8-Test")
    
    # Test on the transformer weights first (largest file)
    input_file = test_dir / "weights.safetensors"
    output_file = test_dir / "weights_quantized.safetensors"
    
    print("=" * 60)
    print("Testing 8-bit Quantization on Transformer Weights")
    print("=" * 60)
    
    stats = quantize_weights_file(str(input_file), str(output_file), bits=8, group_size=64)
    test_dequantize(str(output_file), str(input_file))
    
    print("\n" + "=" * 60)
    print("Done! Check the file sizes above.")
    print("=" * 60)
