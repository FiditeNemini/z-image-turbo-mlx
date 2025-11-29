"""
Full pipeline comparison with identical latents at each step
"""
import torch
import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler
from z_image_mlx import ZImageTransformer2DModel
from text_encoder import TextEncoder
from vae import AutoencoderKL
import json

print("Loading PyTorch pipeline...")
pipe = ZImagePipeline.from_pretrained("models/Z-Image-Turbo", torch_dtype=torch.float32)

print("Loading MLX models...")
with open("models/mlx_model/config.json") as f:
    config = json.load(f)
mlx_transformer = ZImageTransformer2DModel(config)
weights = mx.load("models/mlx_model/weights.safetensors")
mlx_transformer.load_weights(list(weights.items()))
mlx_transformer.eval()

with open("models/mlx_model/text_encoder_config.json") as f:
    te_config = json.load(f)
mlx_text_encoder = TextEncoder(te_config)
te_weights = mx.load("models/mlx_model/text_encoder.safetensors")
mlx_text_encoder.load_weights(list(te_weights.items()), strict=False)
mlx_text_encoder.eval()

with open("models/mlx_model/vae_config.json") as f:
    vae_config = json.load(f)
mlx_vae = AutoencoderKL(vae_config)
vae_weights = mx.load("models/mlx_model/vae.safetensors")
mlx_vae.load_weights(list(vae_weights.items()), strict=False)
mlx_vae.eval()

# Test input
tokenizer = AutoTokenizer.from_pretrained("models/Z-Image-Turbo/tokenizer", trust_remote_code=True)

test_prompt = "A beautiful sunset over the ocean"
messages = [{"role": "user", "content": test_prompt}]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

text_inputs = tokenizer(
    prompt_text,
    padding="max_length",
    max_length=512,
    truncation=True,
    return_tensors="pt",
)

# Get text encoder outputs
with torch.no_grad():
    pt_output = pipe.text_encoder(
        text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        output_hidden_states=True
    )
    pt_last_hidden = pt_output.last_hidden_state

mlx_input_ids = mx.array(text_inputs["input_ids"].numpy())
mlx_attn_mask = mx.array(text_inputs["attention_mask"].numpy())
mlx_text_output = mlx_text_encoder(mlx_input_ids, attention_mask=mlx_attn_mask)
mx.eval(mlx_text_output)

print(f"Text encoder outputs match: {np.allclose(pt_last_hidden.numpy(), np.array(mlx_text_output), atol=0.01)}")

# Create IDENTICAL initial latents using PyTorch
torch.manual_seed(42)
height, width = 1024, 1024
# Latent dimensions: 2 * (height // 16) = 128 for 1024
vae_scale_factor = 8
latent_height = 2 * (height // (vae_scale_factor * 2))
latent_width = 2 * (width // (vae_scale_factor * 2))
latents = torch.randn(1, 16, 1, latent_height, latent_width)
print(f"Latent shape: {latents.shape}")

# Use same latents for MLX
mlx_latents = mx.array(latents.numpy())

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("models/Z-Image-Turbo/scheduler")
scheduler.set_timesteps(9)

print("\n=== Step-by-step comparison ===")

for i, t in enumerate(scheduler.timesteps):
    print(f"\n--- Step {i} (t={t.item():.2f}) ---")
    
    # Prepare inputs
    pt_x_list = [latents[j] for j in range(1)]
    pt_cap_list = [pt_last_hidden[j] for j in range(1)]
    # IMPORTANT: PyTorch pipeline uses (1000 - t) / 1000, NOT t / 1000
    pt_t = torch.tensor([(1000.0 - t.item()) / 1000.0])
    
    mlx_cap_list = [mlx_text_output[j] for j in range(1)]
    mlx_t = mx.array([(1000.0 - t.item()) / 1000.0])
    
    # Forward pass
    with torch.no_grad():
        pt_noise_pred = pipe.transformer(pt_x_list, pt_t, pt_cap_list)[0]
        
    mlx_noise_pred = mlx_transformer(mlx_latents, mlx_t, mlx_cap_list)
    mx.eval(mlx_noise_pred)
    
    pt_np = pt_noise_pred[0].numpy()
    mlx_np = np.array(mlx_noise_pred[0])
    
    diff = np.abs(pt_np - mlx_np)
    print(f"PT noise pred: mean={pt_np.mean():.4f}, std={pt_np.std():.4f}")
    print(f"MLX noise pred: mean={mlx_np.mean():.4f}, std={mlx_np.std():.4f}")
    print(f"Diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    # Scheduler step
    noise_pred_pt = torch.from_numpy(np.array(mlx_noise_pred.squeeze(2)))  # Use MLX prediction for both
    latents_pt = torch.from_numpy(np.array(mlx_latents.squeeze(2)))
    
    step_output = scheduler.step(noise_pred_pt, t, latents_pt, return_dict=True)
    latents = step_output.prev_sample.unsqueeze(2)
    mlx_latents = mx.array(latents.numpy())
    
    print(f"Updated latents: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")

# Final latent comparison
print("\n=== Final latents ===")
print(f"Shape: {latents.shape}")
print(f"Mean: {latents.mean().item():.4f}, Std: {latents.std().item():.4f}")

# VAE Decode comparison
print("\n=== VAE Decode ===")
# PyTorch VAE expects [B, C, H, W]
pt_latents_for_vae = latents.squeeze(2)  # [1, 16, 64, 64]

# Get scaling factors
scaling_factor = vae_config.get("scaling_factor", 0.3611)
shift_factor = vae_config.get("shift_factor", 0.1159)

# PyTorch decode
with torch.no_grad():
    pt_decoded = pipe.vae.decode((pt_latents_for_vae / scaling_factor) + shift_factor).sample
    
print(f"PT decoded shape: {pt_decoded.shape}")
print(f"PT decoded stats: mean={pt_decoded.mean().item():.4f}, std={pt_decoded.std().item():.4f}")

# MLX decode
# MLX VAE expects [B, H, W, C]
mlx_latents_for_vae = mlx_latents.squeeze(2)  # [1, 16, 64, 64]
mlx_latents_for_vae = mlx_latents_for_vae.transpose(0, 2, 3, 1)  # [1, 64, 64, 16]
mlx_latents_for_vae = (mlx_latents_for_vae / scaling_factor) + shift_factor

mlx_decoded = mlx_vae.decode(mlx_latents_for_vae)
mx.eval(mlx_decoded)

print(f"MLX decoded shape: {mlx_decoded.shape}")
print(f"MLX decoded stats: mean={float(mlx_decoded.mean()):.4f}, std={float(mlx_decoded.std()):.4f}")

# Compare (need to transpose MLX output to match PT)
pt_decoded_np = pt_decoded.numpy()  # [1, 3, 1024, 1024]
mlx_decoded_np = np.array(mlx_decoded)  # [1, 1024, 1024, 3]
mlx_decoded_np = mlx_decoded_np.transpose(0, 3, 1, 2)  # [1, 3, 1024, 1024]

vae_diff = np.abs(pt_decoded_np - mlx_decoded_np)
print(f"VAE diff: max={vae_diff.max():.6f}, mean={vae_diff.mean():.6f}")

# Save both images
from PIL import Image

def save_image(tensor, path, is_mlx=False):
    if is_mlx:
        # MLX: [B, H, W, C]
        img = np.array(tensor)[0]
    else:
        # PT: [B, C, H, W]
        img = tensor[0].numpy().transpose(1, 2, 0)
    
    img = (img / 2 + 0.5)
    img = np.clip(img, 0, 1)
    img = (img * 255).round().astype("uint8")
    Image.fromarray(img).save(path)

save_image(pt_decoded, "debug_pt_output.png", is_mlx=False)
save_image(mlx_decoded, "debug_mlx_output.png", is_mlx=True)
print("\nSaved debug_pt_output.png and debug_mlx_output.png")
