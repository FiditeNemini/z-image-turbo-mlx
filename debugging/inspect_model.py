import torch
from diffusers import ZImagePipeline
import inspect

pt_path = "models/Z-Image-Turbo"
pipe = ZImagePipeline.from_pretrained(pt_path)
print(inspect.signature(pipe.transformer.forward))
