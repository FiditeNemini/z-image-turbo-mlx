# Z-Image-Turbo-MLX: User Walkthrough Guide

## What is This Project?

### Plain English Overview

**Z-Image-Turbo-MLX** is a tool that generates high-quality images from text descriptions using AI. Think of it like having a digital artist that can create any image you describe - from "a sunset over mountains" to "a futuristic city with flying cars".

#### What Makes This Special?

1. **Runs on Your Mac** - Unlike many AI image generators that run in the cloud, this runs directly on your Apple Silicon Mac (M1, M2, M3, M4). Your images stay private on your computer.

2. **Fast Generation** - Creates detailed 1024×1024 images in just 9 steps (typically 30-60 seconds depending on your Mac).

3. **High Quality** - The underlying Z-Image-Turbo model has 6 billion parameters, producing professional-quality results.

4. **Multiple Ways to Use** - Whether you prefer clicking buttons in a web interface or typing commands in a terminal, both options are available.

5. **LoRA Support** - Customize your generations with style and concept LoRAs for unique artistic effects.

### How Does It Work? (The Simple Version)

```
Your Text Prompt → AI understands what you want → Generates image from noise → Final Image
     ↓                      ↓                            ↓
"A red apple"        Text Encoder              Transformer + VAE
                   (understands words)      (creates the picture)
```

The AI doesn't "draw" in the traditional sense. Instead:
1. It starts with random noise (like TV static)
2. Gradually "sculpts" that noise into an image that matches your description
3. Each of the 9 steps makes the image clearer and more detailed

---

## Getting Started

### Prerequisites

Before you begin, make sure you have:

- ✅ A Mac with Apple Silicon (M1, M2, M3, or M4 chip)
- ✅ macOS 12.3 or newer
- ✅ Python 3.10 or newer installed
- ✅ About 20GB of free disk space for model weights
- ✅ At least 16GB of RAM (32GB+ recommended for best performance)

### Installation (One-Time Setup)

#### Step 1: Open Terminal

Press `Cmd + Space`, type "Terminal", and press Enter.

#### Step 2: Navigate to the Project

```bash
cd /path/to/z-image-turbo-mlx
```

#### Step 3: Create a Python Environment

```bash
# Create a new conda environment
conda create -n z-image-mlx python=3.12
conda activate z-image-mlx

# Or use venv if you don't have conda
python3 -m venv venv
source venv/bin/activate
```

#### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 5: Download/Convert the Model

The first time you run the app, it will automatically download the model from Hugging Face (~20GB). Alternatively, run manually:

```bash
cd src
python convert_to_mlx.py
```

This takes 10-30 minutes depending on your internet speed.

---

## Using the Web Interface (GUI)

The web interface is the easiest way to use Z-Image-Turbo.

### Launching the App

```bash
python app.py
```

After a few seconds, you'll see:
```
Running on local URL:  http://127.0.0.1:7860
```

Open that URL in your browser (usually opens automatically).

### The Generation Tab

![Generation Tab Overview]

#### Basic Image Generation

1. **Enter Your Prompt** - In the large text box at the top, describe the image you want:
   - ✅ Good: "A serene mountain lake at sunset, with snow-capped peaks reflected in crystal clear water, photorealistic"
   - ❌ Too vague: "nature"

2. **Click Generate** - Press the "Generate" button and wait 30-60 seconds.

3. **View Your Image** - The generated image appears in the output panel.

#### Generation Settings Explained

| Setting | What It Does | Recommended Value |
|---------|--------------|-------------------|
| **Seed** | Controls randomness. Same seed + same prompt = same image | Any number, or -1 for random |
| **Steps** | More steps = more refined image (but slower) | 9 (default, optimized for this model) |
| **Width/Height** | Image dimensions | 1024×1024 (default) |
| **Aspect Ratio** | Quick presets for common ratios | "1:1 Square" for standard |

#### Using the Prompt Enhancer

The app includes an AI that can improve your prompts:

1. Enter a basic prompt: "a cat"
2. Click **Enhance Prompt**
3. Get an improved version: "A fluffy orange tabby cat lounging on a sunlit windowsill, soft natural lighting, detailed fur texture, cozy atmosphere"

#### Backend Selection

- **MLX** - Uses Apple Silicon's native acceleration. Faster and uses less memory.
- **PyTorch** - Traditional GPU computing. Use for comparison or debugging.

Most users should stick with **MLX**.

#### Precision Modes

| Mode | Quality | Speed | Memory |
|------|---------|-------|--------|
| **Original** | Highest | Slowest | Most |
| **FP16** | Excellent | Fast | Less |
| **FP8** | Very Good | Fastest | Least |

**Recommendation**: Start with FP16 for the best balance. Use FP8 if you have memory constraints.

### Using LoRAs (Style & Concept Customization)

LoRAs (Low-Rank Adaptations) let you add custom styles, concepts, or characters to your generations without modifying the base model.

#### What are LoRAs?

Think of LoRAs as "add-ons" that teach the AI new styles or concepts:
- **Style LoRAs**: Anime, photorealistic, oil painting, etc.
- **Concept LoRAs**: Specific objects, environments, or themes
- **Character LoRAs**: Specific characters or people

#### Setting Up LoRAs

1. **Download LoRAs**: Get Z-Image-Turbo compatible LoRAs (`.safetensors` files)
2. **Place in folder**: Put them in `models/loras/`
3. **Organize (optional)**: Use subfolders like `styles/`, `concepts/`, `characters/`

```
models/loras/
├── anime_style.safetensors
├── styles/
│   └── watercolor.safetensors
└── concepts/
    └── cyberpunk_city.safetensors
```

#### Using LoRAs in the UI

1. Expand the **🎨 LoRA Settings** accordion (below the Model selector)
2. You'll see a list of available LoRAs with:
   - **Checkbox**: Enable/disable the LoRA
   - **Name + Trigger**: The LoRA name and any trigger words
   - **Weight**: Adjust strength (0.0 to 2.0, default 1.0)

3. **Enable a LoRA**: Check the box next to it
4. **Adjust Weight**: Use the spinner (0.05 increments)
   - Lower (0.3-0.7): Subtle effect
   - Normal (0.8-1.2): Standard effect
   - Higher (1.3-2.0): Strong effect

5. The **LoRA Tags** display shows your active LoRAs: `<lora:anime:1.0>`

#### Trigger Words

Many LoRAs have trigger words that activate their effects. These appear next to the LoRA name in backticks, for example:

```
anime_style `anime style, detailed`
```

Include these trigger words in your prompt for best results:
- ❌ "A beautiful landscape"
- ✅ "A beautiful landscape, anime style, detailed"

#### Tips for LoRAs

1. **Start with weight 1.0** and adjust from there
2. **Use trigger words** if the LoRA has them
3. **Stack multiple LoRAs** for combined effects (style + concept)
4. **Lower weights when stacking** to prevent over-saturation

⚠️ **Note**: LoRAs are applied when the model loads. Changing LoRA settings requires a model reload (happens automatically when you generate).

#### Saving Fused Models

You can permanently fuse your loaded LoRAs into the base model and save as a new model:

1. **Configure LoRAs**: Enable your desired LoRAs and set weights
2. **Enter a name**: In the LoRA Settings section, find "Save Fused Model" and enter a model name
3. **Select formats**: Choose which formats to export:
   - ☑️ **MLX**: For continued use in this application (`models/mlx/`)
   - ☑️ **PyTorch**: Diffusers format for sharing (`models/pytorch/`)
   - ☑️ **ComfyUI**: Single-file checkpoint for ComfyUI (`models/comfyui/`)
4. **Click Save**: The fused model(s) will be created in the respective directories

**Why fuse LoRAs?**
- Share your custom model configurations with others
- Slightly faster inference (no LoRA merge at load time)
- Use your customized model in other applications (PyTorch/ComfyUI)

### The Model Settings Tab

This tab lets you manage multiple models.

#### Selecting a Model

1. Use the **Select Model** dropdown to see available models
2. Click the 🔄 button to refresh the list
3. Select your desired model

#### Importing Models from Hugging Face

1. Go to **Import from Hugging Face**
2. Enter the model ID (e.g., `username/model-name`)
3. Choose precision (FP16 recommended)
4. Click **Download & Convert to MLX**
5. Wait for download and conversion to complete

#### Importing Single-File Checkpoints

For ComfyUI-style `.safetensors` files:

1. Go to **Import Single-File Checkpoint**
2. Enter the path to your `.safetensors` file
3. Give it a name
4. Choose precision
5. Click **Import to MLX**

⚠️ **Note**: Only Z-Image-Turbo compatible checkpoints work. SDXL, SD1.5, Flux, etc. are NOT compatible.

### Saving Your Images

Generated images appear in the gallery. To save:

1. **Right-click** on the image → **Save Image As**
2. Or use the **Download** button below the image
3. Images are also auto-saved to the `temp/` directory during your session

---

## Using the Command Line (CLI)

For automation, scripting, or if you prefer terminals.

### MLX Generation (Recommended)

Navigate to the `src` directory first:

```bash
cd src
```

#### Basic Usage

```bash
python generate_mlx.py --prompt "Your image description here"
```

#### Full Options

```bash
python generate_mlx.py \
    --prompt "A beautiful sunset over the ocean with sailing ships" \
    --output my_image.png \
    --seed 42 \
    --steps 9 \
    --height 1024 \
    --width 1024 \
    --model_path ../models/mlx/Z-Image-Turbo-MLX
```

#### CLI Arguments Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--prompt` | Text description of desired image | (required) |
| `--output` | Output file path | `generated_mlx.png` |
| `--seed` | Random seed (same seed = same output) | `42` |
| `--steps` | Number of denoising steps | `9` |
| `--height` | Image height in pixels | `1024` |
| `--width` | Image width in pixels | `1024` |
| `--model_path` | Path to MLX model directory | `../models/mlx_model` |

#### Examples

**Generate a landscape:**
```bash
python generate_mlx.py --prompt "Misty mountains at dawn, soft pink and purple sky, pine forest in foreground" --output landscape.png
```

**Generate with specific seed for reproducibility:**
```bash
python generate_mlx.py --prompt "A robot reading a book" --seed 12345 --output robot.png
```

**Generate a portrait-oriented image:**
```bash
python generate_mlx.py --prompt "Elegant fashion model" --height 1344 --width 768 --output portrait.png
```

**Batch generation with different seeds:**
```bash
for seed in 1 2 3 4 5; do
    python generate_mlx.py --prompt "Abstract colorful art" --seed $seed --output "art_$seed.png"
done
```

### PyTorch Generation (Alternative)

For comparison or non-Apple hardware:

```bash
python generate_pytorch.py \
    --prompt "Your prompt here" \
    --output output.png \
    --seed 42 \
    --model_path ../models/pytorch/Z-Image-Turbo
```

Same arguments as MLX version, but uses PyTorch backend.

---

## Tips for Better Results

### Writing Good Prompts

#### Do's ✅

1. **Be Specific**: "A golden retriever puppy playing in autumn leaves, sunlight filtering through trees" beats "a dog"

2. **Describe Style**: Add style keywords like "photorealistic", "digital art", "oil painting", "anime style"

3. **Include Lighting**: "soft natural lighting", "dramatic sunset", "studio lighting"

4. **Mention Quality**: "highly detailed", "4K", "professional photograph"

5. **Use Commas**: Separate distinct concepts with commas for clarity

#### Don'ts ❌

1. Don't use negative prompts (this model doesn't support them)
2. Don't make prompts too long (diminishing returns after ~100 words)
3. Don't expect perfect text in images (AI struggles with letters/words)

### Example Prompts to Try

**Portrait:**
```
Beautiful young woman with flowing red hair, freckles, green eyes, 
soft natural lighting, shallow depth of field, professional portrait photography
```

**Landscape:**
```
Majestic snow-capped mountain range at golden hour, alpine lake in foreground 
with perfect reflection, wildflowers, nature photography, 8K resolution
```

**Fantasy:**
```
Ancient dragon perched on a crumbling castle tower, stormy sky with lightning, 
dark fantasy art style, highly detailed scales, glowing orange eyes
```

**Sci-Fi:**
```
Futuristic cyberpunk city at night, neon signs in Japanese, flying cars, 
rain-slicked streets, blade runner aesthetic, cinematic lighting
```

---

## Troubleshooting

### Common Issues and Solutions

#### "Out of Memory" Error

**Symptoms**: App crashes or shows memory error

**Solutions**:
1. Close other applications (especially Chrome, which uses lots of RAM)
2. Use FP8 precision instead of FP16
3. Reduce image size to 768×768 or 512×512
4. Restart your Mac to clear memory

#### Model Not Appearing in Dropdown

**Symptoms**: Your downloaded model doesn't show up

**Solutions**:
1. Click the 🔄 refresh button
2. Verify the model is in `models/mlx/<model_name>/`
3. Check that all required files exist:
   - `weights.safetensors`
   - `vae.safetensors`
   - `text_encoder.safetensors`
   - `config.json`

#### Generation Produces Garbage/Noise

**Symptoms**: Output is random colors or patterns

**Solutions**:
1. Check model precision - switch between Original/FP16/FP8
2. Re-download the model (may have corrupted during download)
3. Check the TECHNICAL_DOCUMENTATION.md for known issues

#### "Incompatible Checkpoint" Error When Importing

**Symptoms**: Error says "SDXL detected" or "Flux detected"

**Explanation**: You're trying to import a model that's not based on Z-Image-Turbo

**Solution**: Only import checkpoints that are fine-tuned from Z-Image-Turbo

#### Slow Generation

**Symptoms**: Takes much longer than expected

**Optimization Tips**:
1. Use FP8 or FP16 precision
2. Reduce image dimensions
3. Reduce step count (7 steps still produces good results)
4. Close background applications
5. Ensure your Mac isn't thermal throttling (check Activity Monitor)

---

## Understanding the Output

### What the Progress Means

When generating, you'll see output like:
```
Step 0/9 t=999.0
Noise Pred Step 0 Stats: Mean=-0.0012, Std=0.8934
Step 1/9 t=875.0
...
```

- **Step X/9**: Current denoising step
- **t=XXX**: Timestep (999→0 as image forms)
- **Mean/Std**: Statistics about the prediction (for debugging)

### Image Quality Expectations

| Steps | Quality | Time |
|-------|---------|------|
| 5 | Acceptable, some artifacts | ~15-20s |
| 7 | Good, minor imperfections | ~25-35s |
| 9 | Excellent, default recommended | ~40-60s |
| 12+ | Marginal improvement | 60s+ |

---

## Quick Reference Card

### GUI Workflow
```
1. Launch: python app.py
2. Enter prompt
3. (Optional) Click "Enhance Prompt"
4. Adjust settings if needed
5. Click "Generate"
6. Save your image
```

### CLI Workflow
```bash
# Quick generation
cd src && python generate_mlx.py --prompt "Your prompt"

# Custom output
python generate_mlx.py --prompt "..." --output image.png --seed 123

# Different size
python generate_mlx.py --prompt "..." --height 768 --width 1344
```

### Key Directories
```
models/mlx/          # MLX models (used for generation)
models/pytorch/      # PyTorch models (for conversion)
models/loras/        # LoRA files for customization
temp/                # Session-generated images
src/                 # Source code and CLI scripts
```

### Useful Commands
```bash
# Start web UI
python app.py

# Generate via CLI (from src/)
python generate_mlx.py --prompt "..."

# Convert new model
python convert_to_mlx.py

# Migrate old models to new structure
python migrate_models.py --apply
```

---

## Next Steps

1. **Experiment!** Try different prompts and see what works
2. **Read the Technical Docs** - See [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) for deep technical details
3. **Explore Fine-Tunes** - Import community models from Hugging Face
4. **Join the Community** - Share your creations and learn from others

Happy generating! 🎨