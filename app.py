"""
Z-Image-Turbo - Gradio Web Interface
Supports both MLX (Apple Silicon) and PyTorch backends
"""
import gradio as gr
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline
import json
import random
import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Global model cache
_mlx_models = None
_pytorch_pipe = None

# Model paths
PYTORCH_MODEL_PATH = "./models/Z-Image-Turbo"
MLX_MODEL_PATH = "./models/mlx_model"

# Image dimension presets organized by base resolution
# Using traditional photographic/video aspect ratios
# Base resolution is the SMALLER dimension
# Format: {base_resolution: {aspect_ratio_name: (width, height)}}
DIMENSION_PRESETS = {
    "1024": {
        # Square
        "1:1 — 1024×1024": (1024, 1024),
        # Landscape (width is larger)
        "5:4 — 1280×1024": (1280, 1024),
        "4:3 — 1368×1024": (1368, 1024),
        "3:2 — 1536×1024": (1536, 1024),
        "16:9 — 1824×1024": (1824, 1024),
        "21:9 — 2392×1024": (2392, 1024),
        # Portrait (height is larger)
        "4:5 — 1024×1280": (1024, 1280),
        "3:4 — 1024×1368": (1024, 1368),
        "2:3 — 1024×1536": (1024, 1536),
        "9:16 — 1024×1824": (1024, 1824),
        "9:21 — 1024×2392": (1024, 2392),
    },
    "1280": {
        # Square
        "1:1 — 1280×1280": (1280, 1280),
        # Landscape (width is larger)
        "5:4 — 1600×1280": (1600, 1280),
        "4:3 — 1712×1280": (1712, 1280),
        "3:2 — 1920×1280": (1920, 1280),
        "16:9 — 2280×1280": (2280, 1280),
        "21:9 — 2992×1280": (2992, 1280),
        # Portrait (height is larger)
        "4:5 — 1280×1600": (1280, 1600),
        "3:4 — 1280×1712": (1280, 1712),
        "2:3 — 1280×1920": (1280, 1920),
        "9:16 — 1280×2280": (1280, 2280),
        "9:21 — 1280×2992": (1280, 2992),
    },
}

# Default values
DEFAULT_BASE_RESOLUTION = "1024"
DEFAULT_ASPECT_RATIO = "1:1 — 1024×1024"


def check_and_setup_models():
    """Check if models exist, download and convert if necessary."""
    from convert_to_mlx import ensure_model_downloaded
    
    pytorch_path = Path(PYTORCH_MODEL_PATH)
    mlx_path = Path(MLX_MODEL_PATH)
    
    # Check if MLX model exists
    mlx_weights = mlx_path / "weights.safetensors"
    if mlx_weights.exists():
        print(f"✓ MLX model found at {mlx_path}")
        return True
    
    print("MLX model not found. Setting up models...")
    
    # Check if PyTorch model exists, download if not
    transformer_path = pytorch_path / "transformer"
    if not transformer_path.exists() or len(list(transformer_path.glob("*.safetensors"))) == 0:
        print("PyTorch model not found. Downloading from Hugging Face...")
        ensure_model_downloaded(str(pytorch_path))
    else:
        print(f"✓ PyTorch model found at {pytorch_path}")
    
    # Convert to MLX
    print("\nConverting PyTorch model to MLX format...")
    print("This may take a few minutes...\n")
    
    # Import and run conversion
    from convert_to_mlx import convert_weights
    
    # Create output directory
    mlx_path.mkdir(parents=True, exist_ok=True)
    
    # We need to set up args for convert_weights
    class Args:
        model_path = str(pytorch_path / "transformer")
        output_path = str(mlx_path)
    
    # Temporarily set global args for the conversion script
    import convert_to_mlx
    convert_to_mlx.args = Args()
    
    convert_weights(Args.model_path, Args.output_path)
    
    print("\n✓ Model conversion complete!")
    return True


def load_mlx_models(model_path="./models/mlx_model"):
    """Load MLX models (cached globally)"""
    global _mlx_models
    
    if _mlx_models is not None:
        return _mlx_models
    
    import mlx.core as mx
    from z_image_mlx import ZImageTransformer2DModel
    from vae import AutoencoderKL
    from text_encoder import TextEncoder
    
    print("Loading MLX Models...")
    
    # Load Transformer
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    model = ZImageTransformer2DModel(config)
    weights = mx.load(f"{model_path}/weights.safetensors")
    model.load_weights(list(weights.items()))
    model.eval()
    
    # Load VAE
    with open(f"{model_path}/vae_config.json", "r") as f:
        vae_config = json.load(f)
    vae = AutoencoderKL(vae_config)
    vae_weights = mx.load(f"{model_path}/vae.safetensors")
    vae.load_weights(list(vae_weights.items()), strict=False)
    vae.eval()
    
    # Load Text Encoder
    with open(f"{model_path}/text_encoder_config.json", "r") as f:
        te_config = json.load(f)
    text_encoder = TextEncoder(te_config)
    te_weights = mx.load(f"{model_path}/text_encoder.safetensors")
    text_encoder.load_weights(list(te_weights.items()), strict=False)
    text_encoder.eval()
    
    # Load Tokenizer and Scheduler
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer", trust_remote_code=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{model_path}/scheduler")
    
    _mlx_models = {
        "model": model,
        "vae": vae,
        "vae_config": vae_config,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
    }
    
    print("MLX Models loaded successfully!")
    return _mlx_models


def load_pytorch_pipeline(model_path="./models/Z-Image-Turbo"):
    """Load PyTorch pipeline (cached globally)"""
    global _pytorch_pipe
    
    if _pytorch_pipe is not None:
        return _pytorch_pipe
    
    print("Loading PyTorch Pipeline...")
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    _pytorch_pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    _pytorch_pipe.to(device)
    
    print("PyTorch Pipeline loaded successfully!")
    return _pytorch_pipe


# Global cache for prompt enhancer model
_prompt_enhancer = None

PROMPT_ENHANCER_PATH = "./models/prompt_enhancer"
PROMPT_ENHANCER_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def load_prompt_enhancer():
    """Load prompt enhancer model, downloading if necessary"""
    global _prompt_enhancer
    
    if _prompt_enhancer is not None:
        return _prompt_enhancer
    
    try:
        from mlx_lm import load
    except ImportError:
        raise gr.Error("mlx-lm not installed. Run: pip install mlx-lm")
    
    enhancer_path = Path(PROMPT_ENHANCER_PATH)
    
    # Check if local model exists
    if enhancer_path.exists() and (enhancer_path / "config.json").exists():
        print(f"Loading prompt enhancer from {PROMPT_ENHANCER_PATH}...")
        model, tokenizer = load(str(enhancer_path))
    else:
        # Download and save locally
        print(f"Prompt enhancer not found. Downloading {PROMPT_ENHANCER_MODEL}...")
        print("This will be saved to models/prompt_enhancer/ for future use.")
        
        from huggingface_hub import snapshot_download
        
        # Download to local path
        enhancer_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=PROMPT_ENHANCER_MODEL,
            local_dir=str(enhancer_path),
            local_dir_use_symlinks=False,
        )
        
        print(f"✓ Prompt enhancer saved to {PROMPT_ENHANCER_PATH}")
        model, tokenizer = load(str(enhancer_path))
    
    _prompt_enhancer = (model, tokenizer)
    return _prompt_enhancer


def enhance_prompt(prompt, progress=gr.Progress()):
    """Use MLX-LM to enhance the user's prompt with a small local model"""
    
    if not prompt.strip():
        raise gr.Error("Please enter a prompt to enhance")
    
    progress(0.1, desc="Loading language model...")
    
    try:
        from mlx_lm import generate
    except ImportError:
        raise gr.Error("mlx-lm not installed. Run: pip install mlx-lm")
    
    progress(0.2, desc="Loading prompt enhancer...")
    
    model, tokenizer = load_prompt_enhancer()
    
    progress(0.4, desc="Enhancing prompt...")
    
    # System prompt for enhancement
    system_prompt = """You are an expert at writing detailed image generation prompts. 
When given a simple description, expand it into a rich, detailed prompt that includes:
- Specific visual details (colors, textures, lighting)
- Composition and framing
- Artistic style or mood
- Background and environment details

Keep the enhanced prompt concise but descriptive (under 100 words).
Respond ONLY with the enhanced prompt, no explanations or preamble."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Enhance this image prompt: {prompt}"}
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    progress(0.5, desc="Generating enhanced prompt...")
    
    # Create sampler with temperature
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.7)
    
    # Generate
    enhanced = generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=200,
        sampler=sampler,
        verbose=False,
    )
    
    progress(1.0, desc="Done!")
    
    # Clean up
    enhanced = enhanced.strip()
    
    return enhanced


def generate_mlx(prompt, width, height, steps, time_shift, seed, progress):
    """Generate image using MLX backend"""
    import mlx.core as mx
    
    models = load_mlx_models()
    
    model = models["model"]
    vae = models["vae"]
    vae_config = models["vae_config"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    scheduler = models["scheduler"]
    
    # Update scheduler with time shift
    scheduler.config.shift = time_shift
    
    progress(0.1, desc="Encoding prompt...")
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    text_inputs = tokenizer(
        prompt_text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="np",
    )
    
    input_ids = mx.array(text_inputs["input_ids"])
    attention_mask = mx.array(text_inputs["attention_mask"])
    
    # Text encoder forward
    prompt_embeds = text_encoder(input_ids, attention_mask=attention_mask)
    prompt_embeds_list = [prompt_embeds[i] for i in range(len(prompt_embeds))]
    
    progress(0.2, desc="Preparing latents...")
    
    # Prepare latents
    num_channels_latents = 16
    batch_size = 1
    vae_scale_factor = 8
    latent_height = 2 * (height // (vae_scale_factor * 2))
    latent_width = 2 * (width // (vae_scale_factor * 2))
    
    # Use PyTorch for reproducible random
    torch.manual_seed(seed)
    latents_pt = torch.randn(batch_size, num_channels_latents, 1, latent_height, latent_width)
    latents = mx.array(latents_pt.numpy())
    
    # Denoising loop
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps
    
    for i, t in enumerate(timesteps):
        progress(0.2 + 0.6 * (i / len(timesteps)), desc=f"Denoising step {i+1}/{steps}...")
        
        # Timestep
        t_mx = mx.array([(1000.0 - t.item()) / 1000.0])
        
        # Forward pass
        noise_pred = model(latents, t_mx, prompt_embeds_list)
        
        # Negate noise prediction (as PyTorch pipeline does)
        noise_pred = -noise_pred
        mx.eval(noise_pred)
        
        # Scheduler step
        noise_pred_sq = noise_pred.squeeze(2)
        latents_sq = latents.squeeze(2)
        
        noise_pred_pt = torch.from_numpy(np.array(noise_pred_sq))
        latents_pt = torch.from_numpy(np.array(latents_sq))
        
        step_output = scheduler.step(noise_pred_pt, t, latents_pt, return_dict=True)
        latents = mx.array(step_output.prev_sample.numpy())
        latents = latents[:, :, None, :, :]
    
    progress(0.85, desc="Decoding image...")
    
    # Decode latents
    latents = latents.squeeze(2)
    latents = latents.transpose(0, 2, 3, 1)
    
    scaling_factor = vae_config.get("scaling_factor", 0.3611)
    shift_factor = vae_config.get("shift_factor", 0.1159)
    
    latents = (latents / scaling_factor) + shift_factor
    
    image = vae.decode(latents)
    mx.eval(image)
    
    # Convert to PIL
    image = (image / 2 + 0.5)
    image = mx.clip(image, 0, 1)
    image = np.array(image)[0]
    image = (image * 255).round().astype("uint8")
    
    return Image.fromarray(image)


def generate_pytorch(prompt, width, height, steps, time_shift, seed, progress):
    """Generate image using PyTorch backend"""
    pipe = load_pytorch_pipeline()
    
    # Update scheduler with time shift
    pipe.scheduler.config.shift = time_shift
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    progress(0.1, desc="Generating with PyTorch...")
    
    def callback_fn(pipe_obj, step, timestep, callback_kwargs):
        progress(0.1 + 0.8 * (step / steps), desc=f"Denoising step {step+1}/{steps}...")
        return callback_kwargs
    
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=0.0,
        generator=torch.Generator(device).manual_seed(seed),
        callback_on_step_end=callback_fn,
    ).images[0]
    
    return image


def update_aspect_ratios(base_resolution):
    """Update aspect ratio choices based on selected base resolution"""
    choices = list(DIMENSION_PRESETS[base_resolution].keys())
    return gr.update(choices=choices, value=choices[0])


def generate_image(prompt, base_resolution, aspect_ratio, steps, time_shift, seed, backend, progress=gr.Progress()):
    """Generate an image using selected backend"""
    
    if not prompt.strip():
        raise gr.Error("Please enter a prompt")
    
    # Get dimensions from preset
    width, height = DIMENSION_PRESETS[base_resolution][aspect_ratio]
    
    # Handle random seed
    if seed == -1:
        seed = random.randint(0, 2147483647)
    seed = int(seed)
    
    progress(0, desc=f"Loading {backend} models...")
    
    if backend == "MLX (Apple Silicon)":
        pil_image = generate_mlx(prompt, width, height, steps, time_shift, seed, progress)
    else:
        pil_image = generate_pytorch(prompt, width, height, steps, time_shift, seed, progress)
    
    progress(0.95, desc="Preparing downloads...")
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = tempfile.gettempdir()
    
    png_path = os.path.join(temp_dir, f"{timestamp}.png")
    jpg_path = os.path.join(temp_dir, f"{timestamp}.jpg")
    
    pil_image.save(png_path, "PNG")
    pil_image.save(jpg_path, "JPEG", quality=95)
    
    progress(1.0, desc="Done!")
    
    return pil_image, f"Seed: {seed} | Backend: {backend}", png_path, jpg_path


# Create Gradio interface
with gr.Blocks(title="Z-Image-Turbo") as demo:
    gr.Markdown(
        """
        # 🎨 Z-Image-Turbo
        
        Generate high-quality images using Z-Image-Turbo with MLX or PyTorch backend.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=5,
                max_lines=10,
            )
            
            enhance_btn = gr.Button("✨ Enhance Prompt", variant="secondary", size="sm")
            
            with gr.Row():
                base_resolution = gr.Dropdown(
                    choices=list(DIMENSION_PRESETS.keys()),
                    value=DEFAULT_BASE_RESOLUTION,
                    label="Resolution Category",
                )
                
                aspect_ratio = gr.Dropdown(
                    choices=list(DIMENSION_PRESETS[DEFAULT_BASE_RESOLUTION].keys()),
                    value=DEFAULT_ASPECT_RATIO,
                    label="Width × Height (Ratio)",
                )
            
            with gr.Row():
                backend = gr.Dropdown(
                    choices=["MLX (Apple Silicon)", "PyTorch"],
                    value="MLX (Apple Silicon)",
                    label="Backend",
                )
            
            with gr.Row():
                steps = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=9,
                    step=1,
                    label="Inference Steps",
                    info="More steps = better quality but slower (9 recommended)",
                )
                
                time_shift = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.1,
                    label="Time Shift",
                    info="Scheduler shift parameter (default: 3.0)",
                )
            
            with gr.Row():
                seed = gr.Slider(
                    minimum=-1,
                    maximum=2147483647,
                    value=-1,
                    step=1,
                    label="Seed",
                    info="-1 for random seed",
                )
                random_seed_checkbox = gr.Checkbox(
                    label="Random Seed",
                    value=True,
                )
            
            generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
            
            seed_info = gr.Textbox(label="Generation Info", interactive=False)
            
            with gr.Row():
                png_download = gr.File(label="Download PNG")
                jpg_download = gr.File(label="Download JPEG")
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
            )
    
    # Example prompts
    gr.Examples(
        examples=[
            ["A majestic lion with a flowing golden mane, photorealistic, dramatic lighting"],
            ["A cozy coffee shop interior with warm lighting, watercolor style"],
            ["A futuristic cityscape at sunset, cyberpunk aesthetic, neon lights"],
            ["A serene Japanese garden with cherry blossoms, traditional ink painting style"],
            ["An astronaut floating in space with Earth in the background, cinematic"],
        ],
        inputs=[prompt],
        label="Example Prompts",
    )
    
    # Update aspect ratio choices when base resolution changes
    base_resolution.change(
        fn=update_aspect_ratios,
        inputs=[base_resolution],
        outputs=[aspect_ratio],
    )
    
    # Toggle seed slider based on random checkbox
    def toggle_seed(random_checked):
        return gr.update(value=-1 if random_checked else 42, interactive=not random_checked)
    
    random_seed_checkbox.change(
        fn=toggle_seed,
        inputs=[random_seed_checkbox],
        outputs=[seed],
    )
    
    enhance_btn.click(
        fn=enhance_prompt,
        inputs=[prompt],
        outputs=[prompt],
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, base_resolution, aspect_ratio, steps, time_shift, seed, backend],
        outputs=[output_image, seed_info, png_download, jpg_download],
    )


if __name__ == "__main__":
    # Check and setup models on startup
    print("\n" + "="*50)
    print("Z-Image-Turbo - Checking models...")
    print("="*50 + "\n")
    
    check_and_setup_models()
    
    print("\n" + "="*50)
    print("Starting Gradio interface...")
    print("="*50 + "\n")
    
    demo.launch()
