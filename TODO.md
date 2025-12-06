## LoRA Support for Image Generator

✅ **COMPLETED** - LoRA support has been fully implemented!

Add the ability to load and apply LoRA (Low-Rank Adaptation) models to the MLX image generator, enabling style and concept customization without full model retraining.

### Completed Implementation

1. ✅ **Created LoRA module** at `src/lora.py`:
   - `load_lora()` - Load LoRA weights from `.safetensors` files
   - `apply_lora_to_model()` - Merge LoRA weights into transformer using `W' = W + scale * (B @ A)`
   - `apply_multiple_loras()` - Apply multiple LoRAs sequentially with independent scales
   - `get_available_loras()` - Scan `models/loras/` directory (with subfolder support)
   - `get_lora_with_folders()` - Get LoRAs organized by subfolder
   - `get_lora_trigger_words()` - Extract trigger words from LoRA metadata
   - `get_lora_default_weight()` - Get recommended weight from metadata
   - `get_lora_info()` - Get detailed LoRA information (rank, target layers, etc.)
   - Key mapping for ComfyUI format (`diffusion_model.*` prefix)

2. ✅ **Updated UI in `app.py`**:
   - Added collapsible "🎨 LoRA Settings" accordion in Generate tab
   - Individual rows per LoRA with: Enable checkbox | Name + Trigger words | Weight spinner
   - Per-row weight spinners with 0.05 step increments (like A1111)
   - Automatic LoRA tag display showing active LoRAs: `<lora:name:weight>`
   - LoRAs auto-applied during generation when enabled

3. ✅ **Added LoRA folder structure**:
   - `models/loras/` directory for LoRA storage
   - Subfolder organization support (e.g., `styles/`, `concepts/`, `characters/`)
   - Auto-detection of new LoRA files on app refresh
   - `.gitignore` updated to exclude LoRA files from version control

### Features

- **Multiple LoRA Support**: Stack multiple LoRAs with independent weight values
- **Per-LoRA Weight Control**: Fine-tune each LoRA's influence with 0.05 increments
- **Trigger Words**: Automatically displays trigger words from LoRA metadata
- **Subfolder Organization**: Organize LoRAs in subfolders for better management
- **ComfyUI Format**: Supports ComfyUI-style LoRAs with `diffusion_model.*` prefix
- **Runtime Merge**: LoRAs merged into base weights during model reload
- **LoRA Fusion & Export**: Permanently fuse LoRAs and save as new models in MLX, PyTorch, or ComfyUI formats

### Future Considerations

1. **Text Encoder LoRAs** - Many LoRAs include text encoder weights (`lora_te_*`). Could extend `src/text_encoder.py` to support these. **Status: Deferred to Phase 2**

2. **Live Weight Adjustment** - Currently requires model reload to change weights. Layer injection approach would allow live adjustment. **Status: Deferred**

3. **LoRA Training** - Add support for training custom LoRAs. **Status: See TRAINING_TODO.md**

---

## Image Upscaling & Refinement

✅ **FULLY IMPLEMENTED** - Both latent and ESRGAN upscaling integrated!

Post-generation image enhancement using both latent-space and pixel-space upscaling methods.

### Implemented Features

#### Latent Upscaling (Phase 4 - COMPLETED)
- **Latent space upscaling** before VAE decode for enhanced detail
- **Configurable scale factor** (1.0-4.0) with 0.25 steps
- **Interpolation modes**: nearest, linear, cubic (default)
- **Auto-calculated hires steps** based on denoise strength
- **Tiled processing** with automatic memory detection for large upscales
- **Weighted gradient blending** at tile seams

#### ESRGAN Image Upscaler (Phase 1-3 - COMPLETED)
- **4× upscaling** using RRDB-Net architecture
- **MLX-native inference** - runs efficiently on Apple Silicon
- **Tiled processing** for large images (memory efficient)
- **Multiple upscaler support** - choose from available ESRGAN models
- **ESRGAN/RRDB only** - SPAN and other architectures filtered from UI

### Combined Pipeline

```
Base Gen (1024²) → Latent Upscale → Denoise → VAE Decode → ESRGAN
                         ↓              ↓           ↓          ↓
                   256×256 latents  Refined    2048×2048   4096×4096
                   (from 128×128)   latents     pixels      pixels
```

### UI Controls

**🔍 Upscaling Accordion**:
- **Latent Scale**: 1.0-4.0 slider (1.0 = disabled)
- **Interpolation**: Dropdown (nearest/linear/cubic)
- **Hires Steps**: 0-20 slider (0 = auto)
- **Denoise Strength**: 0.0-1.0 slider
- **ESRGAN Model**: Dropdown of available upscalers
- **ESRGAN Scale**: 1.0-4.0 slider

### Available Upscaler Models

| Model | Scale | Best For |
|-------|-------|----------|
| 4x-UltraSharp | 4× | ⭐ General upscaling (recommended) |
| 4x-ClearRealityV1 | 4× | Photorealistic images |
| 4x-ClearRealityV1_Soft | 4× | Softer/artistic look |
| 4x_NMKD-Siax_200k | 4× | Anime/illustrations |
| 4x_NMKD-Superscale-SP_178000_G | 4× | General upscaling |

### Technical Notes

- Upscaler models cached for performance (single load per session)
- Tiled processing for images larger than 512×512 (configurable)
- Upscaled image size stored in metadata (final dimensions)
- Memory: 4× upscale = 16× pixels (e.g., 1024² → 4096²)

### Status: ✅ Completed (both methods)

---

## Speed Acceleration (LeMiCa)

✅ **COMPLETED** - LeMiCa training-free caching implemented!

Training-free acceleration using residual caching between denoising steps.

### Implemented Features

- **LeMiCa caching** in transformer main layers
- **Three speed modes**: slow (~14% faster), medium (~22% faster), fast (~30% faster)
- **CLI support**: `--cache slow/medium/fast` argument
- **UI support**: "⚡ LeMiCa Speed" dropdown in generation settings

### Implementation Details

#### Added to `src/z_image_mlx.py`:
- `LEMICA_SCHEDULES` - Precomputed step schedules for each mode
- `get_lemica_bool_list()` - Convert mode to boolean list
- `ZImageTransformer2DModel.configure_lemica()` - Set up caching
- `ZImageTransformer2DModel.reset_lemica_state()` - Reset for new generation
- Caching logic in `__call__` main layers section

#### Added to `src/generate_mlx.py`:
- `--cache` CLI argument with choices: slow, medium, fast

#### Added to `app.py`:
- `cache_mode` dropdown in UI
- Wired through `generate_with_loras` → `generate_image` → `generate_mlx`

### Speed Modes

| Mode | Steps Computed | Speedup | Quality |
|------|----------------|---------|----------|
| None | 9/9 | Baseline | Reference |
| slow | 7/9 | ~14% | Highest |
| medium | 6/9 | ~22% | Excellent |
| fast | 5/9 | ~30% | Very Good |

### Technical Reference

Based on [LeMiCa: Lexicographic Minimax Path Caching](https://github.com/UnicomAI/LeMiCa) (NeurIPS 2025 Spotlight).

### Status: ✅ Completed

---

## Model Merging

✅ **COMPLETED** - Model merge functionality implemented!

Combine multiple Z-Image-Turbo models using various algorithms.

### Implemented Features

#### Merge Methods
- **Weighted Sum**: `(1-α)A + αB` for proportional blending
- **Add Difference**: `A + α(B-C)` for extracting fine-tune changes
- **Sequential merging** for 3+ models: `((A⊕B)⊕C)⊕D...`

#### Memory Management
- Auto-detection of available RAM via psutil
- Chunked processing for systems with <32GB RAM
- Safe fallback mode when psutil unavailable

#### Model Compatibility
- FP8 quantized models automatically excluded
- Base model from Generate tab (with fused LoRAs) as Model A
- Merged models saved as FP16 precision

### UI Components

| Component | Description |
|-----------|-------------|
| Memory Status | Shows RAM and processing mode |
| Method Dropdown | Weighted Sum / Add Difference |
| Model Rows | Checkbox + weight slider per model |
| Model C Selector | For Add Difference (original model) |
| Output Name | Name for merged model |
| Progress | Real-time merge status |

### Files

- `src/merge.py`: Core merge algorithms and utilities
- `app.py`: Merge tab UI and event handlers

### Status: ✅ Completed