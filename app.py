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

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Global model cache
_mlx_models = None
_pytorch_pipe = None

# Image dimension presets (width, height)
DIMENSION_PRESETS = {
    "1:1 Square (1024×1024)": (1024, 1024),
    "1:1 Square (512×512)": (512, 512),
    "3:2 Landscape (1152×768)": (1152, 768),
    "2:3 Portrait (768×1152)": (768, 1152),
    "4:3 Landscape (1152×864)": (1152, 864),
    "3:4 Portrait (864×1152)": (864, 1152),
    "16:9 Landscape (1280×720)": (1280, 720),
    "9:16 Portrait (720×1280)": (720, 1280),
    "21:9 Ultrawide (1344×576)": (1344, 576),
    "9:21 Tall (576×1344)": (576, 1344),
}


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


def generate_mlx(prompt, width, height, steps, seed, progress):
    """Generate image using MLX backend"""
    import mlx.core as mx
    
    models = load_mlx_models()
    
    model = models["model"]
    vae = models["vae"]
    vae_config = models["vae_config"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    scheduler = models["scheduler"]
    
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


def generate_pytorch(prompt, width, height, steps, seed, progress):
    """Generate image using PyTorch backend"""
    pipe = load_pytorch_pipeline()
    
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


def generate_image(prompt, dimension_preset, steps, seed, backend, progress=gr.Progress()):
    """Generate an image using selected backend"""
    
    if not prompt.strip():
        raise gr.Error("Please enter a prompt")
    
    # Get dimensions from preset
    width, height = DIMENSION_PRESETS[dimension_preset]
    
    # Handle random seed
    if seed == -1:
        seed = random.randint(0, 2147483647)
    seed = int(seed)
    
    progress(0, desc=f"Loading {backend} models...")
    
    if backend == "MLX (Apple Silicon)":
        pil_image = generate_mlx(prompt, width, height, steps, seed, progress)
    else:
        pil_image = generate_pytorch(prompt, width, height, steps, seed, progress)
    
    progress(0.95, desc="Preparing downloads...")
    
    # Save images in different formats for download
    png_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    jpg_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
    
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
            
            with gr.Row():
                dimension_preset = gr.Dropdown(
                    choices=list(DIMENSION_PRESETS.keys()),
                    value="1:1 Square (1024×1024)",
                    label="Image Dimensions",
                )
                
                backend = gr.Dropdown(
                    choices=["MLX (Apple Silicon)", "PyTorch"],
                    value="MLX (Apple Silicon)",
                    label="Backend",
                )
            
            steps = gr.Slider(
                minimum=1,
                maximum=20,
                value=9,
                step=1,
                label="Inference Steps",
                info="More steps = better quality but slower (9 recommended)",
            )
            
            seed = gr.Slider(
                minimum=-1,
                maximum=2147483647,
                value=-1,
                step=1,
                label="Seed",
                info="-1 for random seed",
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
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, dimension_preset, steps, seed, backend],
        outputs=[output_image, seed_info, png_download, jpg_download],
    )


if __name__ == "__main__":
    demo.launch()
