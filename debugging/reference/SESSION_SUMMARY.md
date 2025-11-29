# Session Summary: PyTorch to MLX RoPE Port

**Date:** 2025-11-29  
**Previous Session:** Conversation f280b13f (Debugging RoPE Implementation)

## Current Status

### ✅ Completed
1. **RoPE Implementation Ported** (`mlx_rope_port.py`)
   - `MLXRopeEmbedder` class - Direct port of PyTorch `RopeEmbedder`
   - `apply_rotary_emb_mlx` function - Port with `use_real=True, use_real_unbind_dim=-2`
   - Test cases passing with correct shapes

### 🔄 In Progress  
**Main Transformer Architecture** (`z_image.py`)
- Has `MRoPE` class (lines 61-221) - Multi-dimensional RoPE
- Has `Attention` class (lines 247-318) - Attention mechanism
- Has `S3DiTBlock` class (lines 320-384) - Main transformer block
- Has `RefinerBlock` class (lines 386-445) - Context/Noise refiner blocks

### 📋 Next Steps

#### 1. Validate RoPE Integration
**File:** `mlx_rope_port.py` and `z_image.py`

The `z_image.py` file has its own `MRoPE` implementation that might differ from `mlx_rope_port.py`. Need to:
- Compare `mlx_rope_port.MLXRopeEmbedder` with `z_image.MRoPE`
- Ensure consistency with PyTorch source (`pytorch_rope_full.txt` lines 16-26)
- Validate the split approach: `[2, D//2]` reshape with unbind at dim -2

#### 2. Port Missing Components from PyTorch

**Components still needed:**
- **Attention mechanism internals** - How apply_rotary_emb is called inside Attention
- **FeedForward block** (appears complete in z_image.py lines 232-245)
- **Complete transformer forward pass** matching PyTorch line-by-line

#### 3. Line-by-Line Verification

Based on `pytorch_source_full.txt`, the critical PyTorch implementation:

```python
# PyTorch ZImageTransformerBlock.forward (lines 8-51)
def forward(self, x, attn_mask, freqs_cis, adaln_input=None):
    if self.modulation:
        # Lines 17-19: AdaLN modulation
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
        
        # Lines 22-27: Attention block  
        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            attention_mask=attn_mask,
            freqs_cis=freqs_cis,  # RoPE passed to attention
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        
        # Lines 30-34: FFN block
        x = x + gate_mlp * self.ffn_norm2(
            self.feed_forward(
                self.ffn_norm1(x) * scale_mlp,
            )
        )
```

**Current MLX Implementation Status:**
- ✅ AdaLN modulation logic (z_image.py lines 336-358)
- ✅ Attention + FFN structure (z_image.py lines 360-383)  
- ⚠️ **ISSUE:** RoPE application differs from PyTorch

**Key Difference:**
- **PyTorch:** `freqs_cis` is complex64 returned from `RopeEmbedder.__call__`
- **MLX:** Split into `(cos_emb, sin_emb)` tuples

## Critical Files

### Source (PyTorch)
- `pytorch_rope_full.txt` - RoPE implementation (105 lines)
- `pytorch_source_full.txt` - ZImageTransformerBlock forward (255 lines)
- `pytorch_transformer_full.txt` - Complete model (337 lines)

### Target (MLX)
- `mlx_rope_port.py` - Standalone RoPE port (138 lines) ✅
- `z_image.py` - Main transformer (658 lines) 🔄

### Documentation
- `reference/LEARNINGS.md` - Critical config values
- `reference/DIAGNOSTICS.md` - Diagnostic tools
- `TODO.md` - Project status

## Key Configuration Values
(From `reference/LEARNINGS.md`)

```python
rope_theta = 256.0  # CRITICAL - Not 10k or 1M
axes_dims = [32, 48, 48]  # Time, Height, Width
axes_lens = [1024, 512, 512]  # Max positions
head_dim = 128  # 3840 / 30 heads
```

## Technical Notes

### RoPE Implementation Strategy

**PyTorch Approach:**
```python
# Complex numbers
freqs_cis = torch.polar(ones, freqs)  # cos(θ) + i*sin(θ)
x_rotated = torch.view_as_complex(x.reshape(..., -1, 2))
out = torch.view_as_real(x_rotated * freqs_cis)
```

**MLX Approach (use_real=True, unbind_dim=-2):**
```python
# Separate cos/sin
cos, sin = freqs_cis
x_reshaped = x.reshape(*x.shape[:-1], 2, -1)  # [..., 2, D//2]
x_real, x_imag = x_reshaped.unbind(-2)
x_rotated = torch.cat([-x_imag, x_real], dim=-1)
out = x * cos + x_rotated * sin
```

**Current MLX Port Status:**
- ✅ `mlx_rope_port.py` implements this correctly
- ⚠️ `z_image.py:MRoPE` uses different approach (needs validation)

## Action Plan

### Immediate Next Steps

1. **Test RoPE Port** ✅
   ```bash
   python mlx_rope_port.py
   ```

2. **Create RoPE Comparison Test**
   - Compare `mlx_rope_port.MLXRopeEmbedder` output with PyTorch
   - Validate shapes and numerical accuracy
   - Test on actual position IDs from Z-Image

3. **Integrate RoPE into z_image.py**
   - Replace current `MRoPE` implementation
   - Update `Attention` to use new RoPE
   - Match PyTorch line-by-line

4. **Validate Transformer Block**
   - Port ZImageTransformerBlock exactly as PyTorch
   - Test each component separately
   - Compare intermediate outputs

## References

- **Apple MLX Docs:** https://ml-explore.github.io/mlx/
- **Diffusers RoPE:** https://github.com/huggingface/diffusers
- **Z-Image Paper:** Multi-dimensional RoPE for 3D embeddings

---

**Last Updated:** 2025-11-29 12:31 NZDT
