## Plan: LoRA Support for Image Generator

Add the ability to load and apply LoRA (Low-Rank Adaptation) models to the MLX image generator, enabling style and concept customization without full model retraining.

### Steps

1. **Create LoRA module** at `src/lora.py`: Implement `LoRALinear` class wrapping `nn.Linear` with low-rank A/B matrices, plus functions for loading `.safetensors` LoRA files and mapping keys to the transformer's attention layers (`layers.{0-29}.attention.to_q/k/v/out`, `feed_forward.w1/w2/w3`).

2. **Add LoRA injection to `src/z_image_mlx.py`**: Create `apply_lora()` method on `ZImageTransformer` that merges LoRA weights into target Linear layers using the formula `W' = W + scale * (B @ A)`.

3. **Extend `src/generate_mlx.py`**: Add `lora_paths` and `lora_strengths` parameters to `generate()` and `load_model()`, loading and applying LoRAs after base model weights are loaded.

4. **Update UI in `app.py`**: In the Generate tab, add a new Accordion "🎨 LoRA Settings" below the Guidance Scale slider containing:
   - File browser/dropdown to select available LoRAs from `models/lora/` directory
   - Multi-select list of enabled LoRAs with individual strength sliders (0.0-2.0, default 1.0)
   - Add/remove LoRA buttons

5. **Create LoRA key mapping** in `src/lora.py`: Handle ComfyUI format (`lora_unet_*`, `lora_te_*` prefixes) and diffusers format, reusing existing `_map_transformer_keys()` pattern from `src/generate_mlx.py`.

6. **Add LoRA folder structure**: Create `models/lora/` directory for LoRA storage with auto-detection of new files on app refresh.

### Further Considerations

1. **Text Encoder LoRAs?** Many LoRAs include text encoder weights (`lora_te_*`). Support them by extending `src/text_encoder.py`, or defer to a later phase? **Recommend: Phase 2**

2. **Runtime merge vs. layer injection?** Runtime merge (adding LoRA to base weights) is simpler but requires reload to change strength. Layer injection is more flexible but more complex. **Recommend: Runtime merge for MVP**

3. **Multiple LoRAs?** Support stacking multiple LoRAs with independent strengths, or single LoRA only? **Recommend: Multi-LoRA from start (common use case)**