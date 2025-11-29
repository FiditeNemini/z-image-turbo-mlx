import argparse
import os
import json
import torch
from safetensors.torch import load_file
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download

def convert_weights(model_path, output_path):
    print(f"Loading weights from {model_path}")
    
    state_dict = {}
    model_path = Path(model_path)
    
    if model_path.is_file():
        state_dict = load_file(str(model_path))
    elif model_path.is_dir():
        # Load all .safetensors files
        files = sorted(list(model_path.glob("*.safetensors")))
        for f in files:
            print(f"Loading {f.name}...")
            part = load_file(str(f))
            state_dict.update(part)
    else:
        raise ValueError(f"Invalid model path: {model_path}")
    
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # Ignore unused tokens
        if "pad_token" in key:
            continue
            
        new_key = key

        
        # Map keys
        if key.startswith("all_final_layer.2-1."):
            new_key = key.replace("all_final_layer.2-1.", "final_layer.")
        elif key.startswith("all_x_embedder.2-1."):
            new_key = key.replace("all_x_embedder.2-1.", "x_embedder.")
        elif "t_embedder.mlp." in key:
            # Map mlp.0 -> mlp.layers.0, mlp.2 -> mlp.layers.2
            new_key = key.replace("t_embedder.mlp.", "t_embedder.mlp.layers.")
            
        # Handle adaLN_modulation (safetensors has .0.weight or .1.weight, we use .weight)
        if "adaLN_modulation.0." in new_key:
            new_key = new_key.replace("adaLN_modulation.0.", "adaLN_modulation.")
        if "adaLN_modulation.1." in new_key:
            new_key = new_key.replace("adaLN_modulation.1.", "adaLN_modulation.")
            
        # Handle to_out (safetensors has .0.weight, we use .weight)
        if "to_out.0." in new_key:
            new_key = new_key.replace("to_out.0.", "to_out.")
            
        # Handle cap_embedder (safetensors has .0 and .1, we use layers.0 and layers.1)
        if "cap_embedder.0." in new_key:
            new_key = new_key.replace("cap_embedder.0.", "cap_embedder.layers.0.")
        if "cap_embedder.1." in new_key:
            new_key = new_key.replace("cap_embedder.1.", "cap_embedder.layers.1.")
            
        # Handle Attention/FFN Norms if needed
        # My code uses attention_norm1, attention_norm2, ffn_norm1, ffn_norm2
        # Safetensors has these exact names.
        
        # Handle Attention inner layers
        # Safetensors: attention.to_q, attention.norm_q
        # My code: attention.to_q, attention.norm_q
        # Matches.
        
        # Convert to MLX
        # PyTorch weights are (out, in) for Linear.
        # MLX nn.Linear expects (out, in) in the weight array.
        # So we can just cast to float16/float32 and convert.
        
        # Convert to numpy/mlx
        tensor = mx.array(value.numpy().astype("float16"))
        new_state_dict[new_key] = tensor
        
    print(f"Converted {len(new_state_dict)} tensors.")
    
    # Save
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mx.save_safetensors(str(output_dir / "weights.safetensors"), new_state_dict)
    print(f"Saved weights to {output_dir / 'weights.safetensors'}")
    
    # Read config from source
    config_path = model_path / "config.json"
    if not config_path.exists():
        # Try parent/transformer/config.json if model_path is the transformer dir
        if (model_path.parent / "transformer" / "config.json").exists():
            config_path = model_path.parent / "transformer" / "config.json"
        elif (model_path / "transformer" / "config.json").exists():
            config_path = model_path / "transformer" / "config.json"
            
    if config_path.exists():
        with open(config_path, "r") as f:
            src_config = json.load(f)
            print(f"Loaded config from {config_path}")
    else:
        print("Config not found, using defaults (WARNING: might be wrong)")
        src_config = {}

    # Create Config
    config = {
        "hidden_size": src_config.get("dim", 3840),
        "num_attention_heads": src_config.get("n_heads", 30),
        "intermediate_size": int(src_config.get("dim", 3840) / 3 * 8), # Approximation if not in config
        "num_hidden_layers": src_config.get("n_layers", 30),
        "n_refiner_layers": src_config.get("n_refiner_layers", 2),
        "in_channels": src_config.get("in_channels", 16),
        "text_embed_dim": src_config.get("cap_feat_dim", 2560),
        "patch_size": src_config.get("all_patch_size", [2])[0],
        "rope_theta": src_config.get("rope_theta", 256.0),
        "axes_dims": src_config.get("axes_dims", [32, 48, 48]),
        "axes_lens": src_config.get("axes_lens", [1536, 512, 512]),
    }
    
    # Save transformer weights
    mx.save_safetensors(f"{args.output_path}/weights.safetensors", new_state_dict)
    print(f"Saved weights to {args.output_path}/weights.safetensors")

    # Save transformer config
    with open(f"{args.output_path}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Saved config.json")

    # --- VAE Conversion ---
    print("\nConverting VAE...")
    vae_path = f"{args.model_path.replace('transformer', 'vae')}"
    try:
        vae_weights = load_file(f"{vae_path}/diffusion_pytorch_model.safetensors")
        
        # Load VAE config
        with open(f"{vae_path}/config.json", "r") as f:
            vae_config = json.load(f)
            
        mlx_vae_weights = {}
        for key, value in vae_weights.items():
            # Map VAE keys
            # encoder.conv_in.weight -> encoder.conv_in.weight
            # encoder.down_blocks.0.resnets.0.norm1.weight -> encoder.down_blocks.0.0.norm1.weight
            # We need to handle the ModuleList vs Sequential difference in blocks
            
            new_key = key
            
            # Map down_blocks/up_blocks structure
            # PyTorch: down_blocks.0.resnets.0 -> MLX: down_blocks.0.0
            # PyTorch: down_blocks.0.downsamplers.0 -> MLX: down_blocks.0.X (last item)
            
            parts = new_key.split(".")
            
            if "down_blocks" in new_key:
                # encoder.down_blocks.0.resnets.0...
                block_idx = int(parts[2])
                if "resnets" in new_key:
                    resnet_idx = int(parts[4])
                    # In MLX, resnets are just items in the sequential list
                    # down_blocks[block_idx][resnet_idx]
                    new_parts = parts[:3] + ["layers", str(resnet_idx)] + parts[5:]
                    new_key = ".".join(new_parts)
                elif "downsamplers" in new_key:
                    # downsamplers are at the end of the block
                    # In MLX, it's the last item. We need to know how many resnets there are.
                    # Usually layers_per_block.
                    layers_per_block = vae_config["layers_per_block"]
                    new_parts = parts[:3] + ["layers", str(layers_per_block)] + parts[5:]
                    new_key = ".".join(new_parts)
                    
            elif "up_blocks" in new_key:
                # decoder.up_blocks.0.resnets.0...
                block_idx = int(parts[2])
                if "resnets" in new_key:
                    resnet_idx = int(parts[4])
                    new_parts = parts[:3] + ["layers", str(resnet_idx)] + parts[5:]
                    new_key = ".".join(new_parts)
                elif "upsamplers" in new_key:
                    layers_per_block = vae_config["layers_per_block"]
                    # upsamplers are at the end, so index is layers_per_block + 1 (since range is layers_per_block + 1 in code?)
                    # Wait, my code has range(layers_per_block + 1) for resnets? 
                    # Let's check VAE code: 
                    # for _ in range(self.layers_per_block + 1): block.append(Resnet...)
                    # So resnets are 0, 1, 2 (if layers=2)
                    # Upsampler is last.
                    # PyTorch up_blocks have 3 resnets usually?
                    # Actually PyTorch UpDecoderBlock2D has `resnets` (list) and `upsamplers` (list).
                    # Config says layers_per_block=2.
                    # PyTorch UpBlock usually has 3 resnets.
                    # Let's assume standard mapping: resnets map directly. Upsamplers map to last.
                    
                    # If PyTorch has resnets 0, 1, 2. MLX has 0, 1, 2.
                    # Upsampler is 3.
                    # We need to check how many resnets PyTorch has.
                    # It seems PyTorch UpBlock has layers_per_block + 1 resnets.
                    
                    # So if resnets index is N, it maps to N.
                    # Upsamplers maps to N+1 (where N is max resnet index).
                    # Max resnet index is layers_per_block.
                    
                    new_parts = parts[:3] + ["layers", str(layers_per_block + 1)] + parts[5:]
                    new_key = ".".join(new_parts)

            # Map mid_block
            if "mid_block" in new_key:
                # mid_block.resnets.0 -> mid_block.layers.0
                # mid_block.attentions.0 -> mid_block.layers.1
                # mid_block.resnets.1 -> mid_block.layers.2
                if "resnets.0" in new_key:
                    new_key = new_key.replace("resnets.0", "layers.0")
                elif "attentions.0" in new_key:
                    new_key = new_key.replace("attentions.0", "layers.1")
                elif "resnets.1" in new_key:
                    new_key = new_key.replace("resnets.1", "layers.2")
                # Also handle if it's already 0/1/2 from previous mapping? 
                # No, raw keys are resnets.0 etc.
                
                # If it's just mid_block.0 (e.g. from some other path?), we need to be careful.
                # But PyTorch VAE mid_block is NOT Sequential, it's UNetMidBlock2D.
                # It has .resnets (ModuleList) and .attentions (ModuleList).
                # My vae.py implements it as nn.Sequential.
                # So I need to map resnets.0 -> layers.0, attentions.0 -> layers.1, resnets.1 -> layers.2.
                
            # Map down_blocks
            if "down_blocks" in new_key:
                # down_blocks.0.resnets.0 -> down_blocks.0.layers.0
                # down_blocks.0.downsamplers.0 -> down_blocks.0.layers.X
                
                # We already mapped parts to indices in the previous block.
                # Now we need to insert .layers. before the inner index.
                # The previous block (lines 154-208) did:
                # down_blocks.0.resnets.0 -> down_blocks.0.0
                # We need to change that logic to produce down_blocks.0.layers.0
                pass # Logic is handled in the main block below

            # Map Attention keys
            # to_out.0 -> to_out
            if "to_out.0" in new_key:
                new_key = new_key.replace("to_out.0", "to_out")
                
            # Map GroupNorm
            # PyTorch: norm1.weight, norm1.bias
            # MLX: norm1.weight, norm1.bias (Same)
            
            # Map Conv2d
            # PyTorch: weight [Out, In, K, K]
            # MLX: weight [Out, K, K, In] -> Transpose (0, 2, 3, 1)
            if "conv" in new_key and "weight" in new_key and len(value.shape) == 4:
                # Check if it's not 1x1 conv that doesn't need transpose?
                # MLX Conv2d expects [Out, H, W, In]
                # PyTorch [Out, In, H, W]
                # Transpose: 0, 2, 3, 1
                value = value.permute(0, 2, 3, 1)
                
            # Map Linear
            # PyTorch: weight [Out, In]
            # MLX: weight [Out, In] (Same)
            # No transpose needed!
            
            mlx_vae_weights[new_key] = mx.array(value.float().numpy())

        mx.save_safetensors(f"{args.output_path}/vae.safetensors", mlx_vae_weights)
        with open(f"{args.output_path}/vae_config.json", "w") as f:
            json.dump(vae_config, f, indent=2)
        print("Saved vae.safetensors and vae_config.json")
        
    except Exception as e:
        print(f"Failed to convert VAE: {e}")

    # --- Text Encoder Conversion ---
    print("\nConverting Text Encoder...")
    te_path = f"{args.model_path.replace('transformer', 'text_encoder')}"
    try:
        # Text encoder might be sharded
        # We need to load sharded weights
        from safetensors import safe_open
        import glob
        
        te_files = glob.glob(f"{te_path}/model-*.safetensors")
        if not te_files:
             te_files = glob.glob(f"{te_path}/*.safetensors")
             
        te_weights = {}
        for f in te_files:
            with safe_open(f, framework="pt") as f_st:
                for k in f_st.keys():
                    te_weights[k] = f_st.get_tensor(k)
                    
        # Load config
        with open(f"{te_path}/config.json", "r") as f:
            te_config = json.load(f)
            
        mlx_te_weights = {}
        for key, value in te_weights.items():
            new_key = key
            
            # Map Qwen keys
            # model.layers.0.self_attn.q_proj.weight
            # MLX Qwen: model.layers.0.self_attn.q_proj.weight
            
            # Transpose Linear weights
            # MLX: weight [Out, In] (Same)
            # No transpose needed!
            # if "proj" in new_key and "weight" in new_key and len(value.shape) == 2:
            #     value = value.t()
                
            mlx_te_weights[new_key] = mx.array(value.float().numpy())
            
        mx.save_safetensors(f"{args.output_path}/text_encoder.safetensors", mlx_te_weights)
        with open(f"{args.output_path}/text_encoder_config.json", "w") as f:
            json.dump(te_config, f, indent=2)
        print("Saved text_encoder.safetensors and text_encoder_config.json")
        
    except Exception as e:
        print(f"Failed to convert Text Encoder: {e}")

    # --- Copy Tokenizer and Scheduler ---
    print("\nCopying Tokenizer and Scheduler...")
    try:
        # Tokenizer
        tokenizer_src = args.model_path.replace("transformer", "tokenizer")
        tokenizer_dest = f"{args.output_path}/tokenizer"
        if os.path.exists(tokenizer_src):
            if os.path.exists(tokenizer_dest):
                shutil.rmtree(tokenizer_dest)
            shutil.copytree(tokenizer_src, tokenizer_dest)
            print(f"Copied tokenizer to {tokenizer_dest}")
        else:
            print(f"Tokenizer not found at {tokenizer_src}")
            
        # Scheduler
        scheduler_src = args.model_path.replace("transformer", "scheduler")
        scheduler_dest = f"{args.output_path}/scheduler"
        if os.path.exists(scheduler_src):
            if os.path.exists(scheduler_dest):
                shutil.rmtree(scheduler_dest)
            shutil.copytree(scheduler_src, scheduler_dest)
            print(f"Copied scheduler to {scheduler_dest}")
        else:
            print(f"Scheduler not found at {scheduler_src}")
            
    except Exception as e:
        print(f"Failed to copy tokenizer/scheduler: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/diffusers/Z-Image-Turbo-fp16.safetensors")
    parser.add_argument("--output_path", type=str, default="models/mlx_model")
    args = parser.parse_args()
    
    convert_weights(args.model_path, args.output_path)
