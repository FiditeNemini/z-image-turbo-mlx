# Z-Image-Turbo-MLX: Complete Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Model Components](#model-components)
5. [Weight Key Mappings](#weight-key-mappings)
6. [Precision Modes & Quantization](#precision-modes--quantization)
7. [Model Loading Flow](#model-loading-flow)
8. [Image Generation Pipeline](#image-generation-pipeline)
9. [Model Conversion](#model-conversion)
10. [Critical Implementation Details](#critical-implementation-details)
11. [Known Issues & Solutions](#known-issues--solutions)
12. [Configuration Reference](#configuration-reference)

---

## Project Overview

**Z-Image-Turbo-MLX** is a Gradio web interface for the Z-Image-Turbo image generation model, supporting both:
- **MLX backend** (Apple Silicon optimized, native)
- **PyTorch backend** (via HuggingFace diffusers)

The project supports:
- Model conversion from ComfyUI/PyTorch/HuggingFace formats to MLX
- Three precision modes: Original, FP16, FP8 (quantized)
- Prompt enhancement using a local LLM
- Multiple aspect ratios and resolutions

---

## Architecture Overview

### Z-Image-Turbo Model Architecture

Z-Image-Turbo is a **Flow Matching** diffusion model with:

| Component | Architecture | Description |
|-----------|-------------|-------------|
| **Transformer** | Custom DiT-style | 30 layers, 3840 hidden dim, 30 attention heads |
| **VAE** | AutoencoderKL | 16 latent channels, scale factor 8 |
| **Text Encoder** | Qwen2.5-3B based | 28 layers, 2560 hidden dim, GQA with 4 KV heads |
| **Scheduler** | FlowMatchEulerDiscrete | 9 steps default, shift=3.0 |

### Key Architectural Details

```
Transformer Architecture:
├── x_embedder: Linear(64 → 3840)  # patches: 1×2×2×16 = 64
├── t_embedder: TimestepEmbedder   # 256-dim frequency embedding → 256-dim
├── cap_embedder: Sequential(RMSNorm, Linear(2560 → 3840))
├── rope_embedder: MRoPE(128)      # Multi-axis RoPE with sections [32, 48, 48]
├── noise_refiner: 2× TransformerBlock(modulation=True)
├── context_refiner: 2× TransformerBlock(modulation=False)
├── layers: 30× TransformerBlock(modulation=True)
└── final_layer: FinalLayer
    ├── norm_final: LayerNorm(affine=False)
    ├── linear: Linear(3840 → 64)
    └── adaLN_modulation: Linear(256 → 3840)
```

### TransformerBlock Structure

```
TransformerBlock(dim=3840, num_heads=30, head_dim=128, mlp_dim=10240):
├── attention_norm1: RMSNorm(3840)
├── attention_norm2: RMSNorm(3840)  # For x+cap residual path
├── attention: Attention
│   ├── to_q: Linear(3840 → 3840, bias=False)
│   ├── to_k: Linear(3840 → 3840, bias=False)
│   ├── to_v: Linear(3840 → 3840, bias=False)
│   ├── to_out: Linear(3840 → 3840, bias=False)
│   ├── norm_q: RMSNorm(128)  # Per-head normalization
│   └── norm_k: RMSNorm(128)  # Per-head normalization
├── ffn_norm1: RMSNorm(3840)
├── ffn_norm2: RMSNorm(3840)
├── feed_forward: FeedForward (SwiGLU)
│   ├── w1: Linear(3840 → 10240, bias=False)
│   ├── w2: Linear(10240 → 3840, bias=False)
│   └── w3: Linear(3840 → 10240, bias=False)
└── adaLN_modulation: Linear(256 → 3840)  # Only if modulation=True
```

---

## Directory Structure

```
z-image-turbo-mlx/
├── app.py                    # Main Gradio application (2438 lines)
├── requirements.txt          # Python dependencies
├── README.md                 # User documentation
├── TECHNICAL_DOCUMENTATION.md # This file
├── src/
│   ├── z_image_mlx.py        # MLX Transformer implementation
│   ├── vae.py                # MLX VAE implementation
│   ├── text_encoder.py       # MLX Qwen2 Text Encoder implementation
│   ├── generate_mlx.py       # Standalone MLX generation script
│   ├── generate_pytorch.py   # Standalone PyTorch generation script
│   ├── convert_to_mlx.py     # HuggingFace → MLX converter
│   ├── convert_comfyui_to_mlx.py      # ComfyUI → MLX converter
│   ├── convert_comfyui_to_pytorch.py  # ComfyUI → PyTorch converter
│   ├── convert_mlx_to_pytorch.py      # MLX → PyTorch converter
│   └── convert_pytorch_to_comfyui.py  # PyTorch → ComfyUI converter
├── models/
│   ├── mlx/
│   │   ├── Z-Image-Turbo-MLX/     # Reference working model
│   │   │   ├── config.json
│   │   │   ├── weights.safetensors
│   │   │   ├── vae_config.json
│   │   │   ├── vae.safetensors
│   │   │   ├── text_encoder_config.json
│   │   │   ├── text_encoder.safetensors
│   │   │   ├── tokenizer/
│   │   │   └── scheduler/
│   │   ├── ZTI-FP16-MLX/          # FP16 converted model
│   │   └── ZTI-FP8-MLX/           # FP8 quantized model
│   ├── pytorch/
│   │   └── Z-Image-Turbo/         # HuggingFace format
│   │       ├── model_index.json
│   │       ├── transformer/
│   │       ├── vae/
│   │       ├── text_encoder/
│   │       ├── tokenizer/
│   │       └── scheduler/
│   ├── comfyui/                   # ComfyUI single-file checkpoints
│   └── prompt_enhancer/           # Qwen2.5-1.5B for prompt enhancement
└── debugging/                     # Debug and testing scripts
```

---

## Model Components

### 1. Transformer (`src/z_image_mlx.py`)

**Purpose**: Denoises latent representations conditioned on text embeddings.

**Key Classes**:
- `ZImageTransformer2DModel`: Main model class
- `ZImageTransformerBlock`: Single transformer layer
- `Attention`: Multi-head attention with QK normalization and MRoPE
- `FeedForward`: SwiGLU MLP
- `FinalLayer`: Output projection with adaptive normalization
- `MRoPE`: Multi-axis Rotary Position Embedding
- `TimestepEmbedder`: Sinusoidal timestep embedding

**Input/Output**:
```python
# Input
x: [B, C=16, F=1, H, W]  # Latent (e.g., [1, 16, 1, 128, 128] for 1024×1024)
t: [B]                    # Timestep (0.0 to 1.0, scaled by 1000)
cap_feats: [B, L, 2560]   # Text embeddings from Qwen

# Output  
noise_pred: [B, C=16, F=1, H, W]  # Predicted noise (same shape as input)
```

**Critical Details**:
- Uses `affine=False` for `norm_final` LayerNorm (weights should NOT be loaded)
- RMSNorm for `norm_q` and `norm_k` with shape `(128,)` - MUST be preserved in quantization
- Patch size: 2×2 spatial, 1 temporal (f_patch_size=1)
- Latent channels: 16 (not 4 like SD)

### 2. VAE (`src/vae.py`)

**Purpose**: Encode images to latent space and decode latents to images.

**Key Classes**:
- `AutoencoderKL`: Main VAE class
- `Encoder`: Image → Latent
- `Decoder`: Latent → Image
- `ResnetBlock2D`: Residual convolution block
- `Attention`: VAE attention (different from transformer attention)
- `Downsample2D` / `Upsample2D`: Spatial scaling

**Architecture**:
```
Encoder: [3, H, W] → [16, H/8, W/8]
Decoder: [16, H/8, W/8] → [3, H, W]

block_out_channels: [128, 256, 512, 512]
layers_per_block: 2
latent_channels: 16
scaling_factor: 0.3611
shift_factor: 0.1159
```

**CRITICAL**: VAE uses **NHWC format** in MLX (not NCHW like PyTorch)
```python
# PyTorch: [B, C, H, W]
# MLX:     [B, H, W, C]
```

**GroupNorm Compatibility**: Use `pytorch_compatible=True` for GroupNorm to match PyTorch behavior.

### 3. Text Encoder (`src/text_encoder.py`)

**Purpose**: Encode text prompts into embeddings for conditioning.

**Architecture**: Qwen2.5-3B based
```
hidden_size: 2560
intermediate_size: 13696
num_hidden_layers: 28
num_attention_heads: 20
num_key_value_heads: 4  # GQA
vocab_size: 152064
max_position_embeddings: 32768
```

**Key Classes**:
- `TextEncoder`: Main class
- `Qwen2Model`: Transformer backbone
- `Qwen2Attention`: GQA attention with RoPE
- `Qwen2MLP`: SwiGLU MLP
- `Qwen2RMSNorm`: RMS normalization

**CRITICAL**: Text encoder should **NOT be quantized** - FP8 quantization causes zero outputs.

---

## Weight Key Mappings

### SINGLE SOURCE OF TRUTH

All key mappings are centralized in `app.py`:
- `map_transformer_key()` - Transformer weights
- `map_vae_key()` - VAE weights  
- `map_text_encoder_key()` - Text encoder weights

### Transformer Key Mapping (`map_transformer_key`)

```python
# Prefixes to remove (in order of priority):
"model.diffusion_model."
"diffusion_model."
"diffusion_"
"transformer."
"model."

# Specific mappings:
"all_final_layer.2-1.*"     → "final_layer.*"
"all_x_embedder.2-1.*"      → "x_embedder.*"
"norm_final.weight"         → None  # SKIP - affine=False LayerNorm
"proj_out.*"                → "final_layer.linear.*"
"t_embedder.mlp.0.*"        → "t_embedder.mlp.layers.0.*"
"t_embedder.mlp.1.*"        → "t_embedder.mlp.layers.1.*"
"adaLN_modulation.0.*"      → "adaLN_modulation.*"
"adaLN_modulation.1.*"      → "adaLN_modulation.*"
"to_out.0.*"                → "to_out.*"
"cap_embedder.0.*"          → "cap_embedder.layers.0.*"
"cap_embedder.1.*"          → "cap_embedder.layers.1.*"
".attention.q_norm.*"       → ".attention.norm_q.*"
".attention.k_norm.*"       → ".attention.norm_k.*"
".attention.out.*"          → ".attention.to_out.*"
```

### VAE Key Mapping (`map_vae_key`)

```python
# Remove prefix:
"vae." → ""

# Block structure (layers_per_block=2):
"down_blocks.X.resnets.Y.*"      → "down_blocks.X.layers.Y.*"
"down_blocks.X.downsamplers.0.*" → "down_blocks.X.layers.2.*"
"up_blocks.X.resnets.Y.*"        → "up_blocks.X.layers.Y.*"
"up_blocks.X.upsamplers.0.*"     → "up_blocks.X.layers.3.*"

# Mid block:
"mid_block.resnets.0.*"    → "mid_block.layers.0.*"
"mid_block.attentions.0.*" → "mid_block.layers.1.*"
"mid_block.resnets.1.*"    → "mid_block.layers.2.*"

# Attention:
"to_out.0.*" → "to_out.*"
```

### Text Encoder Key Mapping (`map_text_encoder_key`)

```python
# Prefixes to remove (in order):
"text_encoders.qwen3_4b.transformer."
"text_encoders.qwen3_4b."
"text_encoder.transformer."
"text_encoder."
"transformer."
```

---

## Precision Modes & Quantization

### Supported Modes

| Mode | Transformer | Text Encoder | VAE | File Size |
|------|-------------|--------------|-----|-----------|
| Original | float32 | float32 | float32 | ~30GB |
| FP16 | float16 | float16 | float16 | ~15GB |
| FP8 | quantized (8-bit) | float16 | float16 | ~8GB |

### FP8 Quantization Details

**Method**: MLX affine quantization with `group_size=64`, `bits=8`

```python
# Quantization (in _apply_fp8_quantization):
wq, scales, biases = mx.quantize(weight, group_size=64, bits=8)
mx.eval(wq, scales, biases)  # CRITICAL: Must evaluate before saving!

# Storage format:
# key.weight  → uint32 (packed quantized values)
# key.scales  → float16 (per-group scales)
# key.biases  → float16 (per-group biases)
```

**CRITICAL RULES**:
1. Only quantize 2D+ tensors with `shape[-1] >= 64` and `shape[-1] % 64 == 0`
2. **NEVER quantize text encoder** - causes zero outputs
3. **NEVER quantize VAE** - conv layers incompatible
4. 1D tensors (norms, biases) must be **preserved as-is** (not quantized, not zeroed)

### Loading Quantized Models

```python
# Detection:
def _is_quantized_weights(weights):
    return any(k.endswith('.scales') or k.endswith('.biases') for k in weights.keys())

# Loading:
if is_quantized:
    nn.quantize(model, group_size=64, bits=8)  # Convert layers to QuantizedLinear
    
# Then load weights normally - MLX handles quantized format automatically
```

---

## Model Loading Flow

### MLX Model Loading (`load_mlx_models`)

```python
def load_mlx_models(model_path):
    # 1. Load Transformer
    config = json.load("config.json")
    model = ZImageTransformer2DModel(config)
    weights = mx.load("weights.safetensors")
    
    # 2. Check for quantization
    is_quantized = _is_quantized_weights(weights)
    if is_quantized:
        nn.quantize(model, group_size=64, bits=8)
    
    # 3. Process weight keys
    processed_weights = {}
    for key, value in weights.items():
        if key.endswith('.scales') or key.endswith('.biases'):
            processed_weights[key] = value  # Pass through
            continue
        
        new_key = map_transformer_key(key)
        if new_key is None:  # Skip (e.g., norm_final.weight)
            continue
        if "pad_token" in new_key:
            continue
            
        # Handle fused QKV (non-quantized only)
        if ".attention.qkv.weight" in new_key and not is_quantized:
            # Split into to_q, to_k, to_v
            ...
        
        processed_weights[new_key] = value
    
    model.load_weights(list(processed_weights.items()), strict=False)
    
    # 4. Load VAE (never quantized)
    vae = AutoencoderKL(vae_config)
    vae.load_weights(list(vae_weights.items()), strict=False)
    
    # 5. Load Text Encoder (never quantized)
    text_encoder = TextEncoder(te_config)
    # Apply key mapping
    processed_te_weights = [(map_text_encoder_key(k), v) for k, v in te_weights.items()]
    text_encoder.load_weights(processed_te_weights, strict=False)
    
    # 6. Load tokenizer and scheduler
    tokenizer = AutoTokenizer.from_pretrained("tokenizer/")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("scheduler/")
```

---

## Image Generation Pipeline

### MLX Generation Flow (`generate_mlx`)

```python
def generate_mlx(prompt, width, height, steps, time_shift, seed):
    # 1. Text Encoding
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(prompt_text, padding="max_length", max_length=512, truncation=True)
    input_ids = mx.array(tokens["input_ids"])
    attention_mask = mx.array(tokens["attention_mask"])
    prompt_embeds = text_encoder(input_ids, attention_mask=attention_mask)
    
    # 2. Latent Preparation
    # Latent shape: [B, 16, 1, H/8, W/8] for video-like format
    latent_height = 2 * (height // 16)  # e.g., 128 for 1024px
    latent_width = 2 * (width // 16)
    
    torch.manual_seed(seed)
    latents = mx.array(torch.randn(1, 16, 1, latent_height, latent_width).numpy())
    
    # 3. Denoising Loop
    scheduler.config.shift = time_shift
    scheduler.set_timesteps(steps)
    
    for i, t in enumerate(scheduler.timesteps):
        # Timestep scaling: (1000 - t) / 1000
        t_mx = mx.array([(1000.0 - t.item()) / 1000.0])
        
        # Forward pass
        noise_pred = model(latents, t_mx, [prompt_embeds[0]])
        
        # CRITICAL: Negate noise prediction
        noise_pred = -noise_pred
        mx.eval(noise_pred)
        
        # Scheduler step (using PyTorch for compatibility)
        # Remove temporal dim, run scheduler, add back
        noise_pred_sq = noise_pred.squeeze(2)
        latents_sq = latents.squeeze(2)
        noise_pred_pt = torch.from_numpy(np.array(noise_pred_sq))
        latents_pt = torch.from_numpy(np.array(latents_sq))
        step_output = scheduler.step(noise_pred_pt, t, latents_pt)
        latents = mx.array(step_output.prev_sample.numpy())[:, :, None, :, :]
    
    # 4. VAE Decoding
    latents = latents.squeeze(2)  # Remove temporal dim
    latents = latents.transpose(0, 2, 3, 1)  # NCHW → NHWC for MLX VAE
    
    # Apply inverse scaling
    latents = (latents / 0.3611) + 0.1159
    
    image = vae.decode(latents)  # [B, H, W, 3]
    
    # 5. Post-processing
    image = (image / 2 + 0.5)  # [-1, 1] → [0, 1]
    image = mx.clip(image, 0, 1)
    image = (np.array(image)[0] * 255).astype("uint8")
    return Image.fromarray(image)
```

---

## Model Conversion

### ComfyUI → MLX (`convert_comfyui_to_mlx`)

**Source**: Single `.safetensors` file with all components

**Process**:
1. Detect architecture by checking signature keys
2. Separate weights by prefix:
   - `diffusion_*` or transformer patterns → Transformer
   - `vae.*` or decoder/encoder → VAE
   - `model.layers.*` or `text_encoders.*` → Text Encoder
3. Apply key mappings
4. **VAE**: Copy from reference model (ComfyUI key format too different)
5. **Text Encoder**: Apply `map_text_encoder_key()`
6. Save in MLX format

### FP8 Quantization (`_apply_fp8_quantization`)

```python
def _apply_fp8_quantization(model_path):
    # Only quantize weights.safetensors (transformer)
    weights = mx.load("weights.safetensors")
    quantized_weights = {}
    
    for key, value in weights.items():
        can_quantize = (
            len(value.shape) >= 2 and
            value.shape[-1] >= 64 and
            value.shape[-1] % 64 == 0 and
            "weight" in key
        )
        
        if can_quantize:
            wq, scales, biases = mx.quantize(value, group_size=64, bits=8)
            mx.eval(wq, scales, biases)  # CRITICAL!
            quantized_weights[key] = wq
            quantized_weights[key.replace(".weight", ".scales")] = scales
            quantized_weights[key.replace(".weight", ".biases")] = biases
        else:
            # 1D tensors (norms, biases) - keep as-is
            quantized_weights[key] = value
    
    mx.save_safetensors("weights.safetensors", quantized_weights)
```

---

## Critical Implementation Details

### 1. Norm Weights MUST Be Preserved

**Problem**: FP8 quantization was zeroing out `norm_q.weight` and `norm_k.weight` tensors.

**Impact**: RMSNorm outputs zeros → Attention produces garbage → Model outputs noise.

**Solution**: The `can_quantize` check correctly skips 1D tensors:
```python
can_quantize = len(value.shape) >= 2 and ...
# norm_q.weight has shape (128,) → len(shape)=1 → not quantized
```

**Verification**: After loading, check norm weights are non-zero:
```python
weights = mx.load("weights.safetensors")
assert not np.all(np.array(weights['layers.0.attention.norm_q.weight']) == 0)
```

### 2. mx.eval() Before Saving

**Problem**: MLX uses lazy evaluation. Without `mx.eval()`, quantized weights may not be computed before saving.

**Solution**: Always call `mx.eval()` after quantization:
```python
wq, scales, biases = mx.quantize(value, group_size=64, bits=8)
mx.eval(wq, scales, biases)  # Force computation
```

### 3. Text Encoder Cannot Be Quantized

**Problem**: Quantizing text encoder causes inference to output zeros.

**Impact**: Prompt embeddings become zero vectors → No conditioning → Random noise output.

**Solution**: Always keep text encoder at FP16, even in FP8 mode.

### 4. VAE Uses NHWC Format

**Problem**: MLX Conv2d expects NHWC, PyTorch uses NCHW.

**Solution**: Transpose before VAE operations:
```python
# Before decode: NCHW → NHWC
latents = latents.transpose(0, 2, 3, 1)

# After decode: output is already NHWC [B, H, W, C]
```

### 5. Noise Prediction Negation

**Problem**: Z-Image-Turbo predicts negative noise (opposite sign convention).

**Solution**: Negate after model forward pass:
```python
noise_pred = model(latents, t, prompt_embeds)
noise_pred = -noise_pred  # CRITICAL
```

### 6. Timestep Scaling

**Problem**: Model uses different timestep convention.

**Solution**: Scale timesteps:
```python
# scheduler.timesteps are in [0, 1000] (roughly)
# Model expects [0, 1]
t_model = (1000.0 - t.item()) / 1000.0
```

### 7. GroupNorm pytorch_compatible Flag

**Problem**: MLX GroupNorm default behavior differs from PyTorch.

**Solution**: Use `pytorch_compatible=True`:
```python
self.norm = nn.GroupNorm(groups, channels, eps=1e-6, pytorch_compatible=True)
```

---

## Known Issues & Solutions

### Issue 1: FP8 Model Produces Noise/Garbage

**Symptoms**: 
- Output is random colored noise
- Correlation with FP16 is ~0.1 instead of ~0.99

**Cause**: Norm weights (norm_q, norm_k) are zeros in the saved model.

**Diagnosis**:
```python
weights = mx.load("weights.safetensors")
norm_q = weights.get('layers.0.attention.norm_q.weight')
print(np.array(norm_q)[:5])  # Should NOT be all zeros
```

**Fix**: Re-run quantization from correct FP16 source:
```python
fp16_weights = mx.load("FP16/weights.safetensors")
# Apply quantization logic
# Save to FP8 model
```

### Issue 2: Text Encoder Outputs Zeros

**Symptoms**:
- Prompt embeddings are all zeros
- Images are random noise regardless of prompt

**Cause**: Text encoder was quantized.

**Fix**: Never quantize text encoder. Keep at FP16.

### Issue 3: VAE Produces Artifacts

**Symptoms**:
- Checkerboard patterns
- Color shifts
- Blocky artifacts

**Cause**: 
- Wrong tensor format (NCHW vs NHWC)
- Missing `pytorch_compatible=True` in GroupNorm
- Incorrect scaling factors

**Fix**: Ensure NHWC format and correct GroupNorm settings.

### Issue 4: Wrong Key Mapping

**Symptoms**:
- Model runs but produces bad results
- Certain layers have random weights

**Cause**: Key mapping didn't transform correctly.

**Diagnosis**:
```python
# Check if expected keys exist
for k in ['layers.0.attention.to_q.weight', 'layers.0.attention.norm_q.weight']:
    assert k in processed_weights, f"Missing: {k}"
```

**Fix**: Use centralized `map_transformer_key()` function.

---

## Configuration Reference

### Transformer Config (`config.json`)

```json
{
  "hidden_size": 3840,
  "num_attention_heads": 30,
  "intermediate_size": 10240,
  "num_hidden_layers": 30,
  "n_refiner_layers": 2,
  "in_channels": 16,
  "text_embed_dim": 2560,
  "patch_size": 2,
  "rope_theta": 256.0,
  "axes_dims": [32, 48, 48],
  "axes_lens": [1536, 512, 512],
  "precision": "FP16"  // or "FP8", "Original"
}
```

### VAE Config (`vae_config.json`)

```json
{
  "_class_name": "AutoencoderKL",
  "act_fn": "silu",
  "in_channels": 3,
  "out_channels": 3,
  "latent_channels": 16,
  "block_out_channels": [128, 256, 512, 512],
  "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
  "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "scaling_factor": 0.3611,
  "shift_factor": 0.1159,
  "force_upcast": true,
  "mid_block_add_attention": true,
  "use_post_quant_conv": false,
  "use_quant_conv": false
}
```

### Text Encoder Config (`text_encoder_config.json`)

```json
{
  "hidden_size": 2560,
  "intermediate_size": 13696,
  "max_position_embeddings": 32768,
  "num_attention_heads": 20,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "vocab_size": 152064,
  "rms_norm_eps": 1e-5,
  "rope_theta": 1000000.0,
  "use_sliding_window": false,
  "sliding_window": 32768,
  "tie_word_embeddings": true
}
```

### Scheduler Config (`scheduler/scheduler_config.json`)

```json
{
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "base_image_seq_len": 256,
  "base_shift": 0.5,
  "max_image_seq_len": 4096,
  "max_shift": 1.15,
  "num_train_timesteps": 1000,
  "shift": 3.0
}
```

---

## Quick Reference: Recreating the Project

### From Scratch Steps

1. **Create MLX model implementations** in `src/`:
   - `z_image_mlx.py`: Transformer with MRoPE, SwiGLU, QK-norm
   - `vae.py`: AutoencoderKL with NHWC format
   - `text_encoder.py`: Qwen2.5 architecture with GQA

2. **Create key mapping functions** in `app.py`:
   - `map_transformer_key()`: Handle all prefix/suffix variations
   - `map_vae_key()`: Map resnets→layers, handle mid_block
   - `map_text_encoder_key()`: Strip prefixes

3. **Create model loading** in `app.py`:
   - Detect quantized weights by checking for `.scales`/`.biases` keys
   - Apply `nn.quantize()` to model before loading if quantized
   - Process keys through mapping functions
   - Use `strict=False` for `load_weights()`

4. **Create generation pipeline**:
   - Chat template for tokenizer
   - Timestep scaling: `(1000 - t) / 1000`
   - Negate noise prediction
   - NHWC transpose for VAE
   - Scaling factors: 0.3611 / 0.1159

5. **Create quantization**:
   - Only quantize 2D+ tensors with `shape[-1] % 64 == 0`
   - Always `mx.eval()` before saving
   - Never quantize text encoder or VAE
   - Preserve 1D norm weights as-is

---

## Testing Checklist

- [ ] FP16 model loads without errors
- [ ] FP8 model loads without errors
- [ ] Norm weights are non-zero after loading FP8
- [ ] Text encoder produces non-zero embeddings
- [ ] VAE decode produces valid images (no artifacts)
- [ ] Generated images correlate >0.99 between FP16 and FP8
- [ ] Seeds produce reproducible results
- [ ] Different aspect ratios work correctly

---

*Document Version: 1.0*
*Last Updated: December 2024*
*Project: Z-Image-Turbo-MLX*
