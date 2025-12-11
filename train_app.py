import os
import sys
import gradio as gr
import torch
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

# Force float32 default removed to allow bfloat16 mixed precision for memory savings
# Dtype handling is now managed by the Trainer/LoRATrainer class locally

# Enable MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Disable tokenizer parallelism to avoid warnings on fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Monkey-patch torch.amp.autocast to fix diffusers bug
# diffusers/models/transformers/transformer_z_image.py uses hardcoded 'cuda' autocast
# which crashes on MPS with "Destination NDArray and Accumulator NDArray cannot have 
# different datatype" error. This patch makes 'cuda' autocast a no-op on non-CUDA devices.
_original_autocast = torch.amp.autocast
class _SafeAutocast(_original_autocast):
    def __init__(self, device_type, *args, **kwargs):
        # If trying to use cuda autocast on non-cuda device, use cpu instead (effectively no-op)
        if device_type == "cuda" and not torch.cuda.is_available():
            device_type = "cpu"
            kwargs['enabled'] = False
        super().__init__(device_type, *args, **kwargs)
torch.amp.autocast = _SafeAutocast

from training_ui import create_training_tab, TRAINING_AVAILABLE

def create_app():
    if not TRAINING_AVAILABLE:
        print("Training dependencies not available. Please check your installation.")
        sys.exit(1)

    # Remove theme argument as it causes TypeError in installed Gradio version
    with gr.Blocks(title="Z-Image-Turbo Training") as demo:
        gr.Markdown(
            """
            # Z-Image-Turbo Training
            Train custom LoRAs for Z-Image-Turbo using PyTorch/MPS.
            """
        )
        create_training_tab()
    
    return demo

if __name__ == "__main__":
    import multiprocessing
    # Important for MPS multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861, # Different port from main app
        share=False
    )
