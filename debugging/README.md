# Debugging Tools

This folder contains debugging and diagnostic tools used during the development and porting of Z-Image-Turbo to MLX.

## Overview

These tools were instrumental in identifying and fixing several critical issues in the MLX port:
1. **VAE GroupNorm compatibility** - Fixed by adding `pytorch_compatible=True`
2. **Text encoder attention mask** - Fixed by passing actual mask instead of None
3. **Timestep transformation** - Fixed from `t/1000` to `(1000-t)/1000`
4. **Latent dimensions** - Fixed from 64x64 to 128x128 for 1024x1024 images
5. **Noise negation** - Fixed by adding `-noise_pred` before scheduler step

## Debugging Scripts

### Full Pipeline Comparison
- **`test_full_compare.py`** - **Most useful tool.** Runs both PyTorch and MLX pipelines step-by-step with identical inputs and compares outputs at each stage. Use this to verify the MLX port matches PyTorch.

### Component-Level Debugging
- **`debug_full_pipeline.py`** - Earlier version of full pipeline comparison, useful for understanding the denoising loop structure.
- **`compare_weights.py`** - Compares converted MLX weights against original PyTorch weights to verify conversion accuracy.

### VAE Debugging
- **`debug_vae.py`** - Tests VAE encode/decode separately from the full pipeline.
- **`debug_vae_steps.py`** - Step-through VAE internals to identify where outputs diverge.
- **`debug_vae_attn.py`** - Debug VAE attention layers specifically.
- **`debug_groupnorm.py`** - Tests GroupNorm layer compatibility between PyTorch and MLX.
- **`debug_gn_flag.py`** - Tests the `pytorch_compatible` flag effect on GroupNorm.
- **`debug_resnet_steps.py`** - Debug ResNet blocks in VAE.
- **`debug_resnet_weights.py`** - Verify ResNet weight loading.

### Transformer Debugging  
- **`debug_compare.py`** - Basic comparison of transformer outputs.
- **`debug_compare_detailed.py`** - Detailed layer-by-layer transformer comparison.
- **`debug_pipeline.py`** - Debug the denoising pipeline loop.

### Utility Scripts
- **`inspect_model.py`** - Inspect model architecture and weight shapes.
- **`convert_to_mlx.py`** - Convert PyTorch weights to MLX format. **Keep a copy in /src/ for re-conversion if needed.**

## Output Files
- `debug_output.txt` - Log output from debugging sessions
- `*.png` - Test output images from debugging runs

## Reference Documents
The `/reference/` folder contains documentation created during debugging:
- `AGENT_INSTRUCTIONS.md` - Instructions for AI agents working on this project
- `COMPARISON_RESULTS.md` - Detailed comparison results
- `DIAGNOSTICS.md` - Diagnostic findings
- `LEARNINGS.md` - Key learnings from the porting process
- `SESSION_SUMMARY.md` - Summary of debugging sessions
- `Z_Image_Report.md` - Technical report on Z-Image architecture

## When to Use These Tools

| Scenario | Tool to Use |
|----------|-------------|
| Verify MLX matches PyTorch | `test_full_compare.py` |
| Debug VAE output issues | `debug_vae.py`, `debug_vae_steps.py` |
| Debug transformer issues | `debug_compare_detailed.py` |
| Check weight conversion | `compare_weights.py` |
| Investigate GroupNorm | `debug_groupnorm.py` |

## Notes

These scripts expect models to be in `../models/` relative to this debugging folder, or you may need to adjust paths when running from this directory.
