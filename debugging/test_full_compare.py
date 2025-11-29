"""
Complete step-by-step comparison of PyTorch and MLX pipelines
"""
import torch
import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler
from z_image_mlx import ZImageTransformer2DModel
from text_encoder import TextEncoder
from vae import AutoencoderKL
from PIL import Image
import json

print('Loading models...')
pipe = ZImagePipeline.from_pretrained('models/Z-Image-Turbo', torch_dtype=torch.float32)

with open('models/mlx_model/config.json') as f:
    config = json.load(f)
mlx_transformer = ZImageTransformer2DModel(config)
weights = mx.load('models/mlx_model/weights.safetensors')
mlx_transformer.load_weights(list(weights.items()))
mlx_transformer.eval()

with open('models/mlx_model/text_encoder_config.json') as f:
    te_config = json.load(f)
mlx_text_encoder = TextEncoder(te_config)
te_weights = mx.load('models/mlx_model/text_encoder.safetensors')
mlx_text_encoder.load_weights(list(te_weights.items()), strict=False)

with open('models/mlx_model/vae_config.json') as f:
    vae_config = json.load(f)
mlx_vae = AutoencoderKL(vae_config)
vae_weights = mx.load('models/mlx_model/vae.safetensors')
mlx_vae.load_weights(list(vae_weights.items()), strict=False)

# Same prompt
tokenizer = AutoTokenizer.from_pretrained('models/Z-Image-Turbo/tokenizer', trust_remote_code=True)
prompt = 'A Ukrainian woman holding a sign that says UKRAINE'
messages = [{'role': 'user', 'content': prompt}]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

text_inputs = tokenizer(prompt_text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')

# Text encoding
with torch.no_grad():
    pt_output = pipe.text_encoder(text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'], output_hidden_states=True)
    pt_emb = pt_output.last_hidden_state

mlx_input_ids = mx.array(text_inputs['input_ids'].numpy())
mlx_attn_mask = mx.array(text_inputs['attention_mask'].numpy())
mlx_emb = mlx_text_encoder(mlx_input_ids, attention_mask=mlx_attn_mask)
mx.eval(mlx_emb)

print(f'Text embeddings match: {np.allclose(pt_emb.numpy(), np.array(mlx_emb), atol=0.01)}')

# Same initial latents using PyTorch
torch.manual_seed(123)
latents_init = torch.randn(1, 16, 1, 128, 128)
mlx_latents = mx.array(latents_init.numpy())

# Run scheduler - create two separate scheduler instances
scheduler_pt = FlowMatchEulerDiscreteScheduler.from_pretrained('models/Z-Image-Turbo/scheduler')
scheduler_pt.set_timesteps(9)

scheduler_mlx = FlowMatchEulerDiscreteScheduler.from_pretrained('models/Z-Image-Turbo/scheduler')
scheduler_mlx.set_timesteps(9)

latents_pt = latents_init.clone()

print('\n=== Running denoising loop ===')
for i, t in enumerate(scheduler_pt.timesteps):
    print(f'Step {i} (t={t.item():.2f})', end=' ')
    
    # PyTorch transformer - expects x_list with [C, F, H, W] tensors, cap_list with [seq, dim] tensors
    pt_x_list = [latents_pt[j] for j in range(1)]  # List of [16, 1, 128, 128]
    pt_cap_list = [pt_emb[j] for j in range(1)]    # List of [512, 2560]
    pt_t = torch.tensor([(1000.0 - t.item()) / 1000.0])
    
    with torch.no_grad():
        pt_out = pipe.transformer(pt_x_list, pt_t, pt_cap_list)
        pt_noise = pt_out[0][0]  # tuple[0] is list, list[0] is tensor [16, 1, 128, 128]
    
    # MLX transformer - expects [B, C, F, H, W] tensor
    mlx_t = mx.array([(1000.0 - t.item()) / 1000.0])
    mlx_cap_list = [mlx_emb[j] for j in range(1)]
    mlx_noise = mlx_transformer(mlx_latents, mlx_t, mlx_cap_list)
    mx.eval(mlx_noise)
    
    # Compare noise predictions
    pt_noise_np = pt_noise.numpy()  # [16, 1, 128, 128]
    mlx_noise_np = np.array(mlx_noise[0])  # [16, 1, 128, 128]
    
    diff = np.abs(pt_noise_np - mlx_noise_np)
    print(f'noise diff: max={diff.max():.6f}, mean={diff.mean():.6f}')
    
    # Apply negation (as PyTorch pipeline does)
    pt_noise_neg = -pt_noise.squeeze(1).unsqueeze(0)  # [1, 16, 128, 128]
    mlx_noise_t = torch.from_numpy(-mlx_noise_np.squeeze(1)).unsqueeze(0)  # [1, 16, 128, 128]
    
    # Scheduler step
    pt_latents_sq = latents_pt.squeeze(2)
    step_out_pt = scheduler_pt.step(pt_noise_neg, t, pt_latents_sq, return_dict=True)
    latents_pt = step_out_pt.prev_sample.unsqueeze(2)
    
    mlx_lat_t = torch.from_numpy(np.array(mlx_latents)).squeeze(2)
    step_out_mlx = scheduler_mlx.step(mlx_noise_t, t, mlx_lat_t, return_dict=True)
    mlx_latents = mx.array(step_out_mlx.prev_sample.unsqueeze(2).numpy())

print('\n=== Denoising complete ===')
lat_diff = np.abs(latents_pt.numpy() - np.array(mlx_latents))
print(f'Final latent diff: max={lat_diff.max():.6f}, mean={lat_diff.mean():.6f}')

# VAE Decode
print('\n=== VAE Decode ===')
scaling_factor = vae_config.get("scaling_factor", 0.3611)
shift_factor = vae_config.get("shift_factor", 0.1159)

# PyTorch VAE
pt_latents_vae = latents_pt.squeeze(2)  # [1, 16, 128, 128]
with torch.no_grad():
    pt_decoded = pipe.vae.decode((pt_latents_vae / scaling_factor) + shift_factor).sample

# MLX VAE - expects [B, H, W, C]
mlx_latents_vae = mlx_latents.squeeze(2)  # [1, 16, 128, 128]
mlx_latents_vae = mlx_latents_vae.transpose(0, 2, 3, 1)  # [1, 128, 128, 16]
mlx_latents_vae = (mlx_latents_vae / scaling_factor) + shift_factor
mlx_decoded = mlx_vae.decode(mlx_latents_vae)
mx.eval(mlx_decoded)

print(f'PT decoded: mean={pt_decoded.mean():.4f}, std={pt_decoded.std():.4f}')
print(f'MLX decoded: mean={float(mx.mean(mlx_decoded)):.4f}, std={float(mx.std(mlx_decoded)):.4f}')

# Compare
pt_decoded_np = pt_decoded.numpy()  # [1, 3, 1024, 1024]
mlx_decoded_np = np.array(mlx_decoded)  # [1, 1024, 1024, 3]
mlx_decoded_np = mlx_decoded_np.transpose(0, 3, 1, 2)  # [1, 3, 1024, 1024]

vae_diff = np.abs(pt_decoded_np - mlx_decoded_np)
print(f'VAE diff: max={vae_diff.max():.6f}, mean={vae_diff.mean():.6f}')

# Save images
def tensor_to_pil_pt(tensor):
    """PyTorch: [B, C, H, W]"""
    img = tensor[0].numpy().transpose(1, 2, 0)
    img = (img / 2 + 0.5).clip(0, 1) * 255
    return Image.fromarray(img.round().astype(np.uint8))

def tensor_to_pil_mlx(tensor):
    """MLX: [B, H, W, C]"""
    img = np.array(tensor)[0]
    img = (img / 2 + 0.5).clip(0, 1) * 255
    return Image.fromarray(img.round().astype(np.uint8))

pt_img = tensor_to_pil_pt(pt_decoded)
mlx_img = tensor_to_pil_mlx(mlx_decoded)

pt_img.save('test_compare_pt.png')
mlx_img.save('test_compare_mlx.png')

# Final comparison
pt_arr = np.array(pt_img)
mlx_arr = np.array(mlx_img)
final_diff = np.abs(pt_arr.astype(float) - mlx_arr.astype(float))
print(f'\n=== Final Image Comparison ===')
print(f'PyTorch: shape={pt_arr.shape}, mean={pt_arr.mean():.1f}')
print(f'MLX: shape={mlx_arr.shape}, mean={mlx_arr.mean():.1f}')
print(f'Diff: max={final_diff.max():.1f}, mean={final_diff.mean():.2f}')
print(f'Within 1 pixel: {(final_diff <= 1).mean()*100:.1f}%')
print(f'Within 5 pixels: {(final_diff <= 5).mean()*100:.1f}%')
print('\nSaved test_compare_pt.png and test_compare_mlx.png')
