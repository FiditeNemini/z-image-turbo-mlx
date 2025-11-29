"""
Detailed comparison of PyTorch vs MLX implementations to identify issues.
"""
import torch
import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from diffusers import ZImagePipeline
import json

# Load PyTorch model
print("Loading PyTorch pipeline...")
pipe = ZImagePipeline.from_pretrained("models/Z-Image-Turbo", torch_dtype=torch.float32)

# Load MLX model
print("Loading MLX model...")
from z_image_mlx import ZImageTransformer2DModel
from text_encoder import TextEncoder, Qwen2Model
from vae import AutoencoderKL

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

# Test text encoder
print("\n=== TEXT ENCODER COMPARISON ===")
tokenizer = AutoTokenizer.from_pretrained("models/mlx_model/tokenizer", trust_remote_code=True)

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

# PyTorch forward
with torch.no_grad():
    pt_output = pipe.text_encoder(
        text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        output_hidden_states=True
    )
    pt_last_hidden = pt_output.last_hidden_state
    
print(f"PT Text Encoder Output: shape={pt_last_hidden.shape}, mean={pt_last_hidden.mean().item():.4f}, std={pt_last_hidden.std().item():.4f}")

# MLX forward  
mlx_input_ids = mx.array(text_inputs["input_ids"].numpy())
mlx_attn_mask = mx.array(text_inputs["attention_mask"].numpy())
mx.eval(mlx_input_ids, mlx_attn_mask)

mlx_output = mlx_text_encoder(mlx_input_ids, attention_mask=mlx_attn_mask)
mx.eval(mlx_output)

print(f"MLX Text Encoder Output: shape={mlx_output.shape}, mean={float(mlx_output.mean()):.4f}, std={float(mlx_output.std()):.4f}")

# Compare
pt_np = pt_last_hidden.numpy()
mlx_np = np.array(mlx_output)

diff = np.abs(pt_np - mlx_np)
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")
print(f"✅ MATCH" if diff.max() < 0.1 else "❌ MISMATCH")

# Debug: Check embeddings
print("\n=== EMBEDDING CHECK ===")
try:
    pt_embeds = pipe.text_encoder.embed_tokens(text_inputs["input_ids"]).detach().numpy()
    mlx_embeds = np.array(mlx_text_encoder.model.embed_tokens(mlx_input_ids))

    embeds_diff = np.abs(pt_embeds - mlx_embeds)
    print(f"Embedding Max diff: {embeds_diff.max():.6f}")
    print(f"Embedding Mean diff: {embeds_diff.mean():.6f}")
    print(f"✅ Embeddings MATCH" if embeds_diff.max() < 0.01 else "❌ Embeddings MISMATCH")
except Exception as e:
    print(f"Skipping embedding check: {e}")

# Test transformer
print("\n=== TRANSFORMER COMPARISON ===")

# Create test latents
np.random.seed(42)
batch_size = 1
num_channels = 16
height, width = 64, 64  # Latent space size for 1024x1024 image with scale 16
f_dim = 1

# PyTorch format: [B, C, F, H, W]
pt_latents = torch.randn(batch_size, num_channels, f_dim, height, width, dtype=torch.float32)
pt_timestep = torch.tensor([0.5])  # Normalized [0,1]

# Use the text encoder output
pt_cap_feats = [pt_last_hidden[0]]  # List of [L, D] tensors

with torch.no_grad():
    pt_transformer_out = pipe.transformer(
        [pt_latents[0]],  # List of [C, F, H, W]
        pt_timestep,
        pt_cap_feats,
    )[0]

print(f"PT Transformer Output: shape={pt_transformer_out[0].shape}")
print(f"PT Transformer Output Stats: mean={pt_transformer_out[0].mean().item():.4f}, std={pt_transformer_out[0].std().item():.4f}")

# MLX format: [B, C, F, H, W] 
mlx_latents = mx.array(pt_latents.numpy())
mlx_timestep = mx.array([0.5])
mlx_cap_feats = [mlx_output[0]]

mlx_transformer_out = mlx_transformer(mlx_latents, mlx_timestep, mlx_cap_feats)
mx.eval(mlx_transformer_out)

print(f"MLX Transformer Output: shape={mlx_transformer_out.shape}")
print(f"MLX Transformer Output Stats: mean={float(mlx_transformer_out.mean()):.4f}, std={float(mlx_transformer_out.std()):.4f}")

# Compare transformer outputs
pt_trans_np = pt_transformer_out[0].numpy()
mlx_trans_np = np.array(mlx_transformer_out[0])

# Account for shape differences
if pt_trans_np.shape != mlx_trans_np.shape:
    print(f"Shape mismatch: PT={pt_trans_np.shape}, MLX={mlx_trans_np.shape}")
else:
    trans_diff = np.abs(pt_trans_np - mlx_trans_np)
    print(f"Transformer Max diff: {trans_diff.max():.6f}")
    print(f"Transformer Mean diff: {trans_diff.mean():.6f}")

print("\n=== DEBUGGING INDIVIDUAL COMPONENTS ===")

# Check timestep embedding
print("\n--- Timestep Embedder ---")
pt_t_emb = pipe.transformer.t_embedder(pt_timestep * 1000).detach().numpy()
mlx_t_emb = np.array(mlx_transformer.t_embedder(mlx_timestep * 1000))

t_emb_diff = np.abs(pt_t_emb - mlx_t_emb)
print(f"PT t_emb: shape={pt_t_emb.shape}, mean={pt_t_emb.mean():.4f}, std={pt_t_emb.std():.4f}")
print(f"MLX t_emb: shape={mlx_t_emb.shape}, mean={mlx_t_emb.mean():.4f}, std={mlx_t_emb.std():.4f}")
print(f"t_emb Max diff: {t_emb_diff.max():.6f}")
print(f"✅ t_emb MATCH" if t_emb_diff.max() < 0.1 else "❌ t_emb MISMATCH")

# Check cap_embedder
print("\n--- Cap Embedder ---")
# Use simple test input
test_cap_input = pt_last_hidden[0:1]  # [1, L, D]
pt_cap_emb = pipe.transformer.cap_embedder(test_cap_input).detach().numpy()

mlx_cap_input = mx.array(test_cap_input.numpy())
# MLX Sequential
mlx_cap_emb_0 = mlx_transformer.cap_embedder.layers[0](mlx_cap_input)  # RMSNorm
mlx_cap_emb = mlx_transformer.cap_embedder.layers[1](mlx_cap_emb_0)  # Linear
mlx_cap_emb = np.array(mlx_cap_emb)

cap_emb_diff = np.abs(pt_cap_emb - mlx_cap_emb)
print(f"PT cap_emb: shape={pt_cap_emb.shape}, mean={pt_cap_emb.mean():.4f}, std={pt_cap_emb.std():.4f}")
print(f"MLX cap_emb: shape={mlx_cap_emb.shape}, mean={mlx_cap_emb.mean():.4f}, std={mlx_cap_emb.std():.4f}")
print(f"cap_emb Max diff: {cap_emb_diff.max():.6f}")
print(f"✅ cap_emb MATCH" if cap_emb_diff.max() < 0.1 else "❌ cap_emb MISMATCH")

print("\n=== CHECKING WEIGHT LOADING ===")

# Check if key weights match
print("\n--- Checking cap_embedder weights ---")
pt_cap_norm_w = pipe.transformer.cap_embedder[0].weight.detach().numpy()
pt_cap_lin_w = pipe.transformer.cap_embedder[1].weight.detach().numpy()
pt_cap_lin_b = pipe.transformer.cap_embedder[1].bias.detach().numpy()

mlx_cap_norm_w = np.array(mlx_transformer.cap_embedder.layers[0].weight)
mlx_cap_lin_w = np.array(mlx_transformer.cap_embedder.layers[1].weight)
mlx_cap_lin_b = np.array(mlx_transformer.cap_embedder.layers[1].bias)

print(f"Cap Norm Weight: PT shape={pt_cap_norm_w.shape}, MLX shape={mlx_cap_norm_w.shape}")
print(f"  Max diff: {np.abs(pt_cap_norm_w - mlx_cap_norm_w).max():.6f}")

print(f"Cap Linear Weight: PT shape={pt_cap_lin_w.shape}, MLX shape={mlx_cap_lin_w.shape}")
print(f"  Max diff: {np.abs(pt_cap_lin_w - mlx_cap_lin_w).max():.6f}")

print(f"Cap Linear Bias: PT shape={pt_cap_lin_b.shape}, MLX shape={mlx_cap_lin_b.shape}")
print(f"  Max diff: {np.abs(pt_cap_lin_b - mlx_cap_lin_b).max():.6f}")

print("\n--- Checking x_embedder weights ---")
pt_x_emb_w = pipe.transformer.all_x_embedder["2-1"].weight.detach().numpy()
pt_x_emb_b = pipe.transformer.all_x_embedder["2-1"].bias.detach().numpy()

mlx_x_emb_w = np.array(mlx_transformer.x_embedder.weight)
mlx_x_emb_b = np.array(mlx_transformer.x_embedder.bias)

print(f"X Embedder Weight: PT shape={pt_x_emb_w.shape}, MLX shape={mlx_x_emb_w.shape}")
print(f"  Max diff: {np.abs(pt_x_emb_w - mlx_x_emb_w).max():.6f}")
print(f"X Embedder Bias: PT shape={pt_x_emb_b.shape}, MLX shape={mlx_x_emb_b.shape}")
print(f"  Max diff: {np.abs(pt_x_emb_b - mlx_x_emb_b).max():.6f}")

print("\n=== DONE ===")
