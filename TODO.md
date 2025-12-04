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

### Future Considerations

1. **Text Encoder LoRAs** - Many LoRAs include text encoder weights (`lora_te_*`). Could extend `src/text_encoder.py` to support these. **Status: Deferred to Phase 2**

2. **Live Weight Adjustment** - Currently requires model reload to change weights. Layer injection approach would allow live adjustment. **Status: Deferred**

3. **LoRA Training** - Add support for training custom LoRAs. **Status: Not planned**