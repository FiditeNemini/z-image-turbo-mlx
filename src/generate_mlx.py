import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
# from diffusers import FlowMatchEulerDiscreteScheduler # Still need scheduler logic, but maybe we can implement simple Euler?
# For now, keep scheduler from diffusers but use it with numpy/mlx
from diffusers import FlowMatchEulerDiscreteScheduler
import json
import argparse
from z_image_mlx import ZImageTransformer2DModel
from vae import AutoencoderKL
from text_encoder import TextEncoder

def load_mlx_model(model_path):
    # Load Transformer
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    model = ZImageTransformer2DModel(config)
    weights = mx.load(f"{model_path}/weights.safetensors")
    model.load_weights(list(weights.items()))
    
    # Load VAE
    with open(f"{model_path}/vae_config.json", "r") as f:
        vae_config = json.load(f)
    vae = AutoencoderKL(vae_config)
    vae_weights = mx.load(f"{model_path}/vae.safetensors")
    vae.load_weights(list(vae_weights.items()), strict=False)
    
    # Load Text Encoder
    with open(f"{model_path}/text_encoder_config.json", "r") as f:
        te_config = json.load(f)
    text_encoder = TextEncoder(te_config)
    te_weights = mx.load(f"{model_path}/text_encoder.safetensors")
    text_encoder.load_weights(list(te_weights.items()), strict=False)
    
    return model, vae, text_encoder

def numpy_to_pil(images):
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    return [Image.fromarray(image) for image in images]

def main():
    parser = argparse.ArgumentParser(description='Generate images using Z-Image-Turbo on MLX')
    parser.add_argument('--prompt', type=str, 
                        default="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.",
                        help='Text prompt for image generation')
    parser.add_argument('--steps', type=int, default=9,
                        help='Number of inference steps (default: 9)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='generated_mlx.png',
                        help='Output file path (default: generated_mlx.png)')
    parser.add_argument('--height', type=int, default=1024,
                        help='Image height (default: 1024)')
    parser.add_argument('--width', type=int, default=1024,
                        help='Image width (default: 1024)')
    parser.add_argument('--model_path', type=str, default='../models/mlx_model',
                        help='Path to the MLX model (default: ../models/mlx_model)')
    parser.add_argument('--cache', type=str, default=None, choices=['slow', 'medium', 'fast'],
                        help='LeMiCa cache mode for speed: slow (~14%% faster), medium (~22%% faster), fast (~30%% faster)')
    args = parser.parse_args()

    print("Loading MLX Models...")
    model, vae, text_encoder = load_mlx_model(args.model_path)
    model.eval()
    vae.eval()
    text_encoder.eval()

    print("Loading Tokenizer and Scheduler...")
    # Load Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}/tokenizer", trust_remote_code=True)
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{args.model_path}/scheduler")

    # 1. Encode Prompt
    print("Encoding prompt...")
    
    # Apply chat template
    messages = [{"role": "user", "content": args.prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
             prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    else:
        prompt_text = args.prompt
        
    text_inputs = tokenizer(
        prompt_text,
        padding="max_length",
        max_length=512, 
        truncation=True,
        return_tensors="np", # Return numpy for MLX
    )
    
    input_ids = mx.array(text_inputs["input_ids"])
    attention_mask = mx.array(text_inputs["attention_mask"])
    
    # MLX Text Encoder Forward
    # Qwen2Model returns hidden_states
    prompt_embeds = text_encoder(input_ids, attention_mask=attention_mask) 
    
    # Mask and convert to list
    prompt_embeds_np = np.array(prompt_embeds)
    
    prompt_embeds_list = []
    for i in range(len(prompt_embeds)):
        # Keep padding! Debugging showed that the model expects padded inputs.
        # We just pass the full sequence.
        # However, we might want to pad to a multiple of 32 if it's not already?
        # Tokenizer max_length is 512, which is a multiple of 32.
        valid_embeds = prompt_embeds_np[i]
        print(f"Prompt Embeds {i} Stats: Mean={valid_embeds.mean():.4f}, Std={valid_embeds.std():.4f}, Shape={valid_embeds.shape}")
        prompt_embeds_list.append(mx.array(valid_embeds))
        
    # 2. Prepare Latents
    print("Preparing latents...")
    height = args.height
    width = args.width
    num_channels_latents = 16
    batch_size = 1
    
    # VAE scale factor is 8, but pipeline uses vae_scale_factor * 2 = 16 for divisibility
    # Latent dimensions: height = 2 * (height // 16), width = 2 * (width // 16)
    # For 1024x1024: latents are 128x128
    vae_scale_factor = 8
    latent_height = 2 * (height // (vae_scale_factor * 2))
    latent_width = 2 * (width // (vae_scale_factor * 2))
    
    # Use PyTorch's random generator for reproducibility with PyTorch pipeline
    torch.manual_seed(args.seed)
    latents_pt = torch.randn(batch_size, num_channels_latents, 1, latent_height, latent_width)
    latents = mx.array(latents_pt.numpy())
    print(f"Latent shape: {latents.shape}")
    
    # 3. Denoising Loop
    print("Denoising...")
    scheduler.set_timesteps(args.steps)
    timesteps = scheduler.timesteps
    
    # Configure LeMiCa caching if enabled
    if args.cache:
        model.configure_lemica(args.cache, args.steps)
        print(f"LeMiCa acceleration: {args.cache} mode")
    else:
        model.configure_lemica(None)  # Ensure disabled
    

    
    for i, t in enumerate(timesteps):
        print(f"Step {i}/{args.steps} t={t.item()}")
        
        # Timestep
        # Model expects t in [0, 1], scheduler gives [0, 1000]
        # IMPORTANT: PyTorch pipeline uses (1000 - t) / 1000, NOT t / 1000
        t_mx = mx.array([(1000.0 - t.item()) / 1000.0])
        
        # Forward
        noise_pred = model(latents, t_mx, prompt_embeds_list)
        
        # IMPORTANT: PyTorch pipeline negates the noise prediction
        noise_pred = -noise_pred
        
        print(f"Noise Pred Step {i} Stats: Mean={noise_pred.mean().item():.4f}, Std={noise_pred.std().item():.4f}")
        

        
        if mx.isnan(noise_pred).any():
            print(f"NaN detected in noise_pred at step {i}")
            break
            
        # Squeeze for scheduler (B, C, 1, H, W) -> (B, C, H, W)
        noise_pred_sq = noise_pred.squeeze(2)
        latents_sq = latents.squeeze(2)
        
        # Scheduler Step (PyTorch/Numpy)
        # We convert to numpy for scheduler
        noise_pred_np = np.array(noise_pred_sq)
        latents_np = np.array(latents_sq)
        
        # Scheduler expects torch tensors usually, but we can pass numpy if we convert?
        # Diffusers schedulers usually work with torch.
        # Let's convert to torch just for scheduler step to be safe, or implement Euler step manually.
        # Implementing Euler step is easy: prev_sample = sample + (prev_t - t) * model_output
        # But FlowMatchEuler is slightly different.
        # Let's stick to using scheduler from diffusers but pass torch tensors created from numpy.
        
        noise_pred_pt = torch.from_numpy(noise_pred_np)
        latents_pt = torch.from_numpy(latents_np)
        
        step_output = scheduler.step(noise_pred_pt, t, latents_pt, return_dict=True)
        latents_pt = step_output.prev_sample
        
        # Convert back to MLX and unsqueeze
        latents = mx.array(latents_pt.numpy())
        latents = latents[:, :, None, :, :] # Add F dim back
        
    # 4. Decode
    print("Decoding...")
    # MLX VAE Decode
    # Latents: [B, C, 1, H, W] -> [B, H, W, C] for MLX VAE?
    # My VAE implementation expects [B, H, W, C]?
    # Let's check vae.py.
    # Encoder: conv_in(x). Conv2d expects [N, H, W, C].
    # So yes, MLX expects [N, H, W, C].
    # Current latents: [B, C, 1, H, W].
    # Squeeze F dim: [B, C, H, W]
    latents = latents.squeeze(2)
    # Transpose to [B, H, W, C]
    latents = latents.transpose(0, 2, 3, 1)
    
    print(f"Latents Stats: Mean={latents.mean().item():.4f}, Std={latents.std().item():.4f}")
    
    # Scale and Shift
    # vae_config from json
    # We need to access scaling_factor from config dict
    scaling_factor = vae.config.get("scaling_factor", 0.3611)
    shift_factor = vae.config.get("shift_factor", 0.1159)
    
    latents = (latents / scaling_factor) + shift_factor
    
    # Decode
    image = vae.decode(latents)
    print(f"Decoded Image Stats (pre-clip): Mean={image.mean().item():.4f}, Std={image.std().item():.4f}, Min={image.min().item():.4f}, Max={image.max().item():.4f}")
    
    # Save
    image = (image / 2 + 0.5)
    image = mx.clip(image, 0, 1)
    image = np.array(image) # [B, H, W, C]
    
    images = numpy_to_pil(image)
    images[0].save(args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
