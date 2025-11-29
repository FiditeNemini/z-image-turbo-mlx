"""
Debug the PyTorch pipeline to understand exact I/O formats
"""
import torch
import numpy as np
from transformers import AutoTokenizer
from diffusers import ZImagePipeline

print("Loading PyTorch pipeline...")
pipe = ZImagePipeline.from_pretrained("models/Z-Image-Turbo", torch_dtype=torch.float32)

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

# Get text encoder output
with torch.no_grad():
    pt_output = pipe.text_encoder(
        text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        output_hidden_states=True
    )
    pt_last_hidden = pt_output.last_hidden_state

print(f"Text encoder output shape: {pt_last_hidden.shape}")

# Now let's see how the pipeline prepares cap_feats for the transformer
# In the pipeline's __call__ method, it gets prompt_embeds from text_encoder
# and then passes it directly to transformer

# Check what the transformer expects
print("\n=== Transformer input format ===")
print(f"Transformer class: {type(pipe.transformer)}")

# Create test latents
height, width = 1024, 1024
num_channels = 16
batch_size = 1

# Check scheduler
from diffusers import FlowMatchEulerDiscreteScheduler
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("models/Z-Image-Turbo/scheduler")
scheduler.set_timesteps(9)
print(f"\nScheduler timesteps: {scheduler.timesteps}")

# Create latents
torch.manual_seed(42)
latents = torch.randn(batch_size, num_channels, 1, height // 16, width // 16)
print(f"Latents shape: {latents.shape}")

# The transformer forward signature
# forward(self, x: List[torch.Tensor], t, cap_feats: List[torch.Tensor], ...):
# x is a LIST of tensors, not a batched tensor!

t = scheduler.timesteps[0]
print(f"\nFirst timestep: {t}")

# Prepare input as the pipeline does
x_list = [latents[i] for i in range(batch_size)]  # List of [C, F, H, W]
cap_feats_list = [pt_last_hidden[i] for i in range(batch_size)]  # List of [L, D]

print(f"x_list[0] shape: {x_list[0].shape}")
print(f"cap_feats_list[0] shape: {cap_feats_list[0].shape}")

# Also check - does the model expect normalized timestep or raw?
# Check the t_embedder in transformer

# The transformer has t_scale = 1000.0
# It does: t = t * self.t_scale  in forward
# So if we pass t=500 (from scheduler), it becomes t=500000 internally
# That's wrong!

# Let me check what the scheduler provides
print(f"\nScheduler timesteps type: {type(scheduler.timesteps[0])}")
print(f"Scheduler timesteps values: {scheduler.timesteps}")

# The scheduler provides timesteps in range [0, 1000]
# But the transformer does t = t * t_scale where t_scale=1000
# So if t=1000 -> 1000 * 1000 = 1,000,000
# That can't be right either

# Let's look at the actual scaling in the transformer forward
print("\n=== Checking transformer forward ===")
with torch.no_grad():
    # Run a forward pass
    # t should be [B] tensor in range [0, 1]
    t_tensor = torch.tensor([t.item() / 1000.0])  # [1]
    output = pipe.transformer(x_list, t_tensor, cap_feats_list)
    
print(f"Transformer output type: {type(output)}")
print(f"Transformer output[0][0] shape: {output[0][0].shape}")
print(f"Transformer output[0][0] stats: mean={output[0][0].mean().item():.4f}, std={output[0][0].std().item():.4f}")

# Compare to MLX
print("\n=== MLX Comparison ===")
import mlx.core as mx
from z_image_mlx import ZImageTransformer2DModel
from text_encoder import TextEncoder
import json

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

# Get MLX text encoder output
mlx_input_ids = mx.array(text_inputs["input_ids"].numpy())
mlx_output = mlx_text_encoder(mlx_input_ids, attention_mask=None)
mx.eval(mlx_output)

# Prepare MLX inputs
mlx_latents = mx.array(latents.numpy())
mlx_t = mx.array([t.item() / 1000.0])

# The MLX model expects:
# x: [B, C, F, H, W] - batched
# t: [B] 
# cap_feats: list of [L, D]

mlx_cap_feats = [mlx_output[i] for i in range(batch_size)]

mlx_out = mlx_transformer(mlx_latents, mlx_t, mlx_cap_feats)
mx.eval(mlx_out)

print(f"MLX output shape: {mlx_out.shape}")
print(f"MLX output stats: mean={float(mlx_out.mean()):.4f}, std={float(mlx_out.std()):.4f}")

# Compare
pt_out_np = output[0][0].numpy()
mlx_out_np = np.array(mlx_out[0])

diff = np.abs(pt_out_np - mlx_out_np)
print(f"\nMax diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")

# Now test with identical random latents
print("\n=== Testing with identical random state ===")
np.random.seed(42)
torch.manual_seed(42)
pt_latents = torch.randn(1, 16, 1, 64, 64)

mx.random.seed(42)
# MLX random is different from numpy/torch, so use numpy
np.random.seed(42)
mlx_latents = mx.array(np.random.randn(1, 16, 1, 64, 64).astype(np.float32))

print(f"PT latents mean: {pt_latents.mean().item():.4f}")
print(f"MLX latents mean: {float(mlx_latents.mean()):.4f}")
print(f"Latents diff: {np.abs(pt_latents.numpy() - np.array(mlx_latents)).max():.6f}")
