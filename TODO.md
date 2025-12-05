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

✅ **IMPLEMENTED** - ESRGAN 4× upscaling integrated!

Post-generation image enhancement using ESRGAN-type upscalers. Community-requested feature to improve image quality and resolution.

### Implemented Features

#### ESRGAN Image Upscaler (Option 2 - Recommended)
- **4× upscaling** using RRDB-Net architecture
- **MLX-native inference** - runs efficiently on Apple Silicon
- **Tiled processing** for large images (memory efficient)
- **Multiple upscaler support** - choose from available ESRGAN models
- **ESRGAN/RRDB only** - SPAN and other architectures filtered from UI

### Implementation Completed

#### Phase 1: File Structure & Model Support ✅
- [x] Created `models/upscalers/` directory
- [x] Added upscaler model loading support (ESRGAN `.pth` format)
- [x] Implemented MLX-compatible upscaler inference (`src/upscaler.py`)
- [x] Supports RRDB-Net architecture (23 blocks, 64 features)

#### Phase 2: UI Changes ✅
- [x] Added "🔍 Upscaling (ESRGAN)" accordion in Generate tab
- [x] **Upscaler Dropdown**: Select upscaler model (None, 4x-UltraSharp, etc.)
- [x] Upscaler info displayed in generation details

#### Phase 3: Backend Implementation ✅
- [x] Created `src/upscaler.py` module with:
  - `load_upscaler()` - Load ESRGAN-type model
  - `upscale_image()` - Run image through upscaler
  - `get_available_upscalers()` - Scan `models/upscalers/` directory
  - `RRDBNet` - MLX implementation of RRDB network
- [x] Integrated with generation pipeline (optional post-process step)
- [x] Updated metadata to include upscaling info

### Available Upscaler Models

| Model | Scale | Best For |
|-------|-------|----------|
| 4x-UltraSharp | 4× | ⭐ General upscaling (recommended) |
| 4x-ClearRealityV1 | 4× | Photorealistic images |
| 4x-ClearRealityV1_Soft | 4× | Softer/artistic look |
| 4x_NMKD-Siax_200k | 4× | Anime/illustrations |
| 4x_NMKD-Superscale-SP_178000_G | 4× | General upscaling |

### Future Enhancements (Optional)

#### Phase 4: Latent Upscale Method
- [ ] Implement latent space upscaling
- [ ] Add short refinement sampler pass (Euler/Euler-a)
- [ ] Allow selection between methods (Image vs Latent upscale)

### Technical Notes

- Upscaler models cached for performance (single load per session)
- Tiled processing for images larger than 512×512 (configurable)
- Upscaled image size stored in metadata (final dimensions)
- Memory: 4× upscale = 16× pixels (e.g., 1024² → 4096²)

### Status: ✅ Completed (ESRGAN method)