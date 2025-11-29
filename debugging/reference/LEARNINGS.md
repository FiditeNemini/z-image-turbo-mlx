# Learnings

## Z-Image Architecture (Source Inspection)

**Date:** 2025-11-28

### Transformer Configuration (`transformer/config.json`)
- **Class**: `ZImageTransformer2DModel`
- **Dimensions**:
    - `dim`: 3840
    - `n_heads`: 30
    - `head_dim`: 128 (3840/30)
    - `cap_feat_dim`: 2560 (Matches Qwen3-4B)
    - `in_channels`: 16 (Matches VAE)
- **RoPE**:
    - **`rope_theta`: 256.0** (CRITICAL: Previous assumption of 1M or 10k was wrong)
    - **`axes_dims`: [32, 48, 48]** (Time, Height, Width)
- **Normalization**:
    - `qk_norm`: True
    - `norm_eps`: 1e-05

### Text Encoder (`text_encoder/config.json`)
- **Model**: Qwen3-4B (`qwen3`)
- **Hidden Size**: 2560
- **RoPE Theta**: 1,000,000 (For text encoder only)

### VAE (`vae/config.json`)
- **Model**: FLUX.1-dev VAE
- **Channels**: 16
- **Scaling Factor**: 0.3611
- **Shift Factor**: 0.1159 (Need to apply this during decoding)

### Scheduler (`scheduler/scheduler_config.json`)
- **Type**: `FlowMatchEulerDiscreteScheduler`
- **Shift**: 3.0

## Action Items
1. Update `z_image.py`:
    - Set `rope_theta` to 256.0.
    - Set `MRoPE` sections to `[32, 48, 48]`.
    - Verify `norm_eps` is 1e-05.
2. Update `generate_image.py`:
    - Implement VAE un-shifting: `latents = (latents / scaling_factor) + shift_factor`.
    - Update scheduler shift to 3.0.
3. Re-convert weights from the fresh source to ensure integrity.

## RoPE Implementation (MLX Port)

**Date:** 2025-11-29

### PyTorch → MLX Key Differences

**PyTorch Approach:**
- Uses `torch.polar(ones, freqs)` to create complex numbers: `cos(θ) + i*sin(θ)`
- Complex64 format inherently stores both cos and sin components
- Returns single `freqs_cis` tensor of complex numbers

**MLX Approach:**
- Split into separate `cos` and `sin` arrays (MLX lacks native complex number support)
- Each frequency computed for half-dimension: `[e, d//2]`
- **CRITICAL:** Must repeat each value twice to match full dimension: `[e, d]`
- Returns tuple: `(freqs_cos, freqs_sin)`

### Implementation Details

**Frequency Computation:**
```python
# For each dimension d in axes_dims
freqs = 1.0 / (theta ** (mx.arange(0, d, 2) / d))  # [d//2]
freqs_grid = timestep[:, None] * freqs[None, :]    # [e, d//2]
```

**Dimension Expansion:**
```python
# PyTorch complex numbers implicitly have full dimension
# MLX must explicitly repeat for use_real_unbind_dim=-2 mode
freqs_cos_i = mx.repeat(freqs_cos_half, 2, axis=-1)  # [e, d//2] → [e, d]
freqs_sin_i = mx.repeat(freqs_sin_half, 2, axis=-1)  # [e, d//2] → [e, d]
```

**Why Repeat?**
- PyTorch `unbind_dim=-2` splits tensor as `[..., 2, D//2]`
- Each pair `(real, imag)` at position i needs same cos/sin value
- Repeating ensures correct correspondence during rotation

### Validation Results

**Test Suite (`test_rope_comprehensive.py`):**
- ✅ 7 comprehensive tests, all passing
- ✅ Shapes verified: `[N, 128]` for sum(axes_dims) = 32+48+48
- ✅ Position independence confirmed
- ✅ Numerical stability across theta values (10, 256, 1000, 10000)
- ✅ Boundary conditions tested (position 0 and max)
- ✅ Caching mechanism verified

### Next Steps
1. Integrate `mlx_rope_port.py` into main transformer (`z_image.py`)
2. Update Attention class to use new RoPE format
3. Port remaining transformer components line-by-line
