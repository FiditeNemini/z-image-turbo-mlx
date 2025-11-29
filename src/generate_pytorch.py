import torch
import argparse
from diffusers import ZImagePipeline

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate images using Z-Image-Turbo on Mac Studio (MPS)')
parser.add_argument('--prompt', type=str, 
                    default="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.",
                    help='Text prompt for image generation')
parser.add_argument('--steps', type=int, default=9,
                    help='Number of inference steps (default: 9)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
parser.add_argument('--output', type=str, default='example.png',
                    help='Output file path (default: example.png)')
parser.add_argument('--height', type=int, default=1024,
                    help='Image height (default: 1024)')
parser.add_argument('--width', type=int, default=1024,
                    help='Image width (default: 1024)')
parser.add_argument('--model_path', type=str, default='../models/Z-Image-Turbo',
                    help='Path to the Z-Image-Turbo model (default: ../models/Z-Image-Turbo)')

args = parser.parse_args()

# Check MPS availability
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because PyTorch was not built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device.")
    device = "cpu"
else:
    device = "mps"
    print(f"Using device: {device}")

# 1. Load the pipeline
# Use float16 instead of bfloat16 for MPS compatibility
pipe = ZImagePipeline.from_pretrained(
    args.model_path,
    torch_dtype=torch.float32,  # Changed from float16 for stability
    low_cpu_mem_usage=False,
)
pipe.to(device)

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# Uncomment if you encounter memory issues:
# pipe.enable_model_cpu_offload()

# 2. Generate Image
# 2. Generate Image
# We need to hook into the pipeline to get intermediate stats
# Or we can just use the pipeline callback
def callback_fn(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    if step % 4 == 0 or step == 8:
        print(f"Latents Step {step} Stats: Mean={latents.mean().item():.4f}, Std={latents.std().item():.4f}")
    return callback_kwargs

image = pipe(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.steps,
    guidance_scale=0.0,
    generator=torch.Generator(device).manual_seed(args.seed),
    callback_on_step_end=callback_fn,
).images[0]

image.save(args.output)
print(f"Image saved as {args.output}")