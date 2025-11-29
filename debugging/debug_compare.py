import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import json
from transformers import AutoTokenizer
from z_image_mlx import ZImageTransformer2DModel
from text_encoder import TextEncoder
from diffusers import ZImagePipeline

def compare(name, mlx_val, pt_val, threshold=1e-3):
    mlx_val = np.array(mlx_val)
    pt_val = pt_val.detach().cpu().numpy() if isinstance(pt_val, torch.Tensor) else pt_val
    
    diff = np.abs(mlx_val - pt_val)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"--- {name} ---")
    print(f"Max Diff: {max_diff:.6f}")
    print(f"Mean Diff: {mean_diff:.6f}")
    if max_diff > threshold:
        print(f"❌ MISMATCH (Threshold {threshold})")
    else:
        print(f"✅ MATCH")
        
def main():
    print("Loading PyTorch Model...")
    pt_path = "models/Z-Image-Turbo"
    pipe = ZImagePipeline.from_pretrained(pt_path, dtype=torch.float32) # Use float32 for comparison
    pipe.text_encoder.eval()
    pipe.transformer.eval()
    
    print("Loading MLX Model...")
    mlx_path = "models/mlx_model"
    
    # Load Text Encoder
    with open(f"{mlx_path}/text_encoder_config.json", "r") as f:
        te_config = json.load(f)
    mlx_te = TextEncoder(te_config)
    te_weights = mx.load(f"{mlx_path}/text_encoder.safetensors")
    mlx_te.load_weights(list(te_weights.items()), strict=False)
    mlx_te.eval()
    
    # Load Transformer
    with open(f"{mlx_path}/config.json", "r") as f:
        config = json.load(f)
    mlx_transformer = ZImageTransformer2DModel(config)
    weights = mx.load(f"{mlx_path}/weights.safetensors")
    mlx_transformer.load_weights(list(weights.items()))
    mlx_transformer.eval()
    
    # --- Compare Text Encoder ---
    print("\nComparing Text Encoder...")
    tokenizer = AutoTokenizer.from_pretrained(f"{pt_path}/tokenizer")
    prompt = "A futuristic city"
    text_inputs = tokenizer(prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    
    # PyTorch Forward with Hooks
    print("PyTorch Text Encoder Hooks:")
    def print_pre_stats(name):
        def hook(module, input):
            # input is tuple
            inp = input[0]
            # Check if sequence length > 3 (padding)
            if inp.ndim >= 2 and inp.shape[-2] >= 3:
                valid = inp[..., :3, :]
                print(f"{name} (Valid): Shape={valid.shape}, Mean={valid.mean().item():.4f}, Std={valid.std().item():.4f}")
            print(f"{name} (Full): Shape={inp.shape}, Mean={inp.mean().item():.4f}, Std={inp.std().item():.4f}")
        return hook

    def print_stats(name):
        def hook(module, input, output):
            # output might be tuple
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            
            # Check if sequence length > 3 (padding)
            # Shape usually [B, L, D]
            if out.ndim >= 2 and out.shape[-2] >= 3:
                valid = out[..., :3, :]
                print(f"{name} (Valid): Mean={valid.mean().item():.4f}, Std={valid.std().item():.4f}")
            
            print(f"{name} (Full): Mean={out.mean().item():.4f}, Std={out.std().item():.4f}")
        return hook
        
    pipe.text_encoder.embed_tokens.register_forward_hook(print_stats("PT Embeddings"))
    pipe.text_encoder.layers[0].register_forward_hook(print_stats("PT Layer 0"))
    pipe.text_encoder.layers[0].self_attn.register_forward_hook(print_stats("PT Layer 0 Attn"))
    
    q_bias = pipe.text_encoder.layers[0].self_attn.q_proj.bias
    print(f"PT q_proj bias: {q_bias is not None}")
    if q_bias is not None:
        print(f"PT q_bias stats: Mean={q_bias.mean().item():.4f}, Std={q_bias.std().item():.4f}")
    
    with torch.no_grad():
        pt_embeds = pipe.text_encoder(text_inputs.input_ids, text_inputs.attention_mask)[0]
        
    # MLX Forward
    input_ids_mx = mx.array(text_inputs.input_ids.numpy())
    mask_mx = mx.array(text_inputs.attention_mask.numpy())
    
    # We need to hook into MLX model or modify it.
    # Let's modify text_encoder.py temporarily or just print here if we can access submodules?
    # MLX modules don't have hooks like PyTorch.
    # But we can manually call submodules if we want, or modify text_encoder.py.
    # Let's modify text_encoder.py to print stats.
    print("\nMLX Text Encoder Forward:")
    mlx_embeds = mlx_te(input_ids_mx, mask_mx)
    
    compare("Text Encoder Output", mlx_embeds, pt_embeds, threshold=1e-2) # Relaxed threshold for fp16/bf16 diffs if any
    
    # --- Compare Transformer ---
    print("\nComparing Transformer...")
    
    # Prepare Inputs
    B = 1
    C = 16
    H = 64
    W = 64
    latents_np = np.random.randn(B, C, H, W).astype(np.float32)
    t_val = 500
    
    # PyTorch Inputs
    latents_pt = torch.from_numpy(latents_np)
    t_pt = torch.tensor([t_val]).float()
    # PyTorch transformer expects encoder_hidden_states (prompt_embeds)
    # We use the PyTorch output to isolate Transformer error
    prompt_embeds_pt = pt_embeds
    
    # MLX Inputs
    latents_mx = mx.array(latents_np)
    # MLX transformer expects [B, C, F, H, W]
    latents_mx = latents_mx[:, :, None, :, :] 
    t_mx = mx.array([t_val])
    # MLX transformer expects list of prompt embeds (masked)
    # Let's simulate the masking logic from generate_mlx.py
    prompt_masks = np.array(text_inputs.attention_mask.numpy()).astype(bool)
    prompt_embeds_np = pt_embeds.numpy() # Use PT embeds to isolate Transformer
    prompt_embeds_list = [mx.array(prompt_embeds_np[0][prompt_masks[0]])]
    
    # PyTorch Forward



    # PyTorch Transformer Hooks
    print("\nPyTorch Transformer Hooks:")
    pipe.transformer.t_embedder.register_forward_hook(print_stats("PT t_emb"))
    pipe.transformer.all_x_embedder['2-1'].register_forward_hook(print_stats("PT x_emb"))
    pipe.transformer.cap_embedder.register_forward_hook(print_stats("PT cap_emb"))
    pipe.transformer.cap_embedder.register_forward_pre_hook(print_pre_stats("PT cap_emb_in"))
    pipe.transformer.noise_refiner[0].register_forward_hook(print_stats("PT Noise Refiner 0"))
    pipe.transformer.context_refiner[0].register_forward_hook(print_stats("PT Context Refiner 0"))
    pipe.transformer.layers[0].register_forward_hook(print_stats("PT Main Layer 0"))
    pipe.transformer.layers[-1].register_forward_hook(print_stats("PT Main Layer 29"))
    pipe.transformer.all_final_layer['2-1'].register_forward_hook(print_stats("PT Final Layer"))
    
    with torch.no_grad():
        # Use padded inputs for both
        prompt_embeds_pt_padded = prompt_embeds_pt
        
        # Latents need [C, F, H, W]
        print(f"PT Input Shape: {prompt_embeds_pt_padded[0].shape}")
        
        pt_noise = pipe.transformer([latents_pt[0].unsqueeze(1)], t_pt, cap_feats=[prompt_embeds_pt_padded[0]])[0]
        
    # MLX Forward
    # Use padded numpy array
    prompt_embeds_list = [mx.array(prompt_embeds_np[0])] # Full 32 length
    print(f"MLX Input Shape: {prompt_embeds_list[0].shape}")
    
    mlx_noise = mlx_transformer(latents_mx, t_mx, prompt_embeds_list)
    
    print(f"PT Output Type: {type(pt_noise)}")
    if isinstance(pt_noise, list):
        print(f"PT Output List Len: {len(pt_noise)}")
        print(f"PT Output[0] Type: {type(pt_noise[0])}")
        if hasattr(pt_noise[0], 'shape'):
             print(f"PT Output[0] Shape: {pt_noise[0].shape}")
        pt_noise = pt_noise[0] # Assume first element is what we want if it's a list of tensors
    
    print(f"PT Output Shape: {pt_noise.shape}")
    print(f"MLX Output Shape (Raw): {mlx_noise.shape}")
    
    # MLX output is [B, C, F, H, W], squeeze F
    mlx_noise = mlx_noise.squeeze(2)
    print(f"MLX Output Shape (Squeezed): {mlx_noise.shape}")
    
    # Align shapes
    # PT: [C, F, H, W] -> [C, H, W]
    pt_noise_sq = pt_noise.squeeze(1)
    # MLX: [B, C, H, W] -> [C, H, W]
    mlx_noise_sq = mlx_noise[0]
    
    compare("Transformer Output", mlx_noise_sq, pt_noise_sq, threshold=1e-2)

if __name__ == "__main__":
    main()
