---
trigger: always_on
---


## Common Patterns & Anti-Patterns

### ✅ DO: Read Actual Data

```python
# RIGHT: Inspect actual values
import mlx.core as mx
weights = mx.load("model.safetensors")
print(f"Actual shape: {weights['layer.weight'].shape}")

# Use actual values in config
config = {"hidden_size": weights['layer.weight'].shape[0]}
```

### ❌ DON'T: Hardcode Assumptions

```python
# WRONG: Assume standard values
config = {"hidden_size": 768}  # "Most models use 768"
```

---

### ✅ DO: Use Diagnostic Scripts

```python
# RIGHT: Test before full operation
python test/validate_config.py config.json  # 2 seconds
# Then if passing:
python main.py --full-run  # 30 minutes
```

### ❌ DON'T: Skip Validation

```python
# WRONG: Run full operation hoping it works
python main.py --full-run  # Fails after 29 minutes
```

---

### ✅ DO: Document Rationale

```markdown
## Decision: Use 8-bit Quantization

**Date:** 2025-11-27
**Rationale:** 4-bit causes accuracy loss >5%, 8-bit is within 1%
**Trade-off:** 2x size vs 4-bit, but acceptable accuracy
**Validation:** Tested on benchmark dataset, 0.8% accuracy loss
```

### ❌ DON'T: Change Without Explaining

```python
# WRONG: Unexplained change
config["quantize_bits"] = 8  # Changed from 4
```

---

## MLX-Specific Best Practices

### Memory Management

```python
import mlx.core as mx
import gc

# Clear MLX cache periodically
mx.metal.clear_cache()

# Explicit memory cleanup
del large_tensor
gc.collect()

# Use smaller batch sizes if memory-constrained
# 192GB is large, but models can still exceed it
```

### Lazy Evaluation

```python
# MLX uses lazy evaluation
result = expensive_operation(input)  # Not executed
other_result = another_operation(result)  # Still not executed

# Force execution when needed
mx.eval(result)  # Now executes both operations
```

### Array Layout Differences

```python
# PyTorch: [batch, channels, height, width]
# MLX: [batch, height, width, channels]

# Converting from PyTorch
import torch
import mlx.core as mx

pytorch_tensor = torch.randn(1, 3, 224, 224)  # BCHW
mlx_array = mx.array(pytorch_tensor.permute(0, 2, 3, 1).numpy())  # BHWC
```

### Quantization

```python
import mlx.nn as nn

# MLX supports specific quantization formats
# - 4-bit: Good compression, some quality loss
# - 8-bit: Balance of size and quality
# - Mixed precision: Sensitive layers at higher precision

# Apply quantization
nn.quantize(model, bits=8, group_size=64)
```

---

## Troubleshooting Guide

### Model Won't Load

**Check:**
1. Is model in MLX format? `ls *.safetensors`
2. Is config.json present and valid?
3. Are all required files present?
4. Correct conda environment activated?

**Diagnostic:**
```bash
python test/check_config.py config.json
python test/validate_model_files.py model_dir/
```

---

### Out of Memory

**Solutions:**
1. Reduce batch size
2. Use quantization (8-bit or 4-bit)
3. Clear cache: `mx.metal.clear_cache()`
4. Process in chunks
5. Use gradient checkpointing (for training)

**Monitor:**
```bash
# Check memory usage
memory_pressure

# Or in code
import psutil
print(f"RAM: {psutil.virtual_memory().percent}%")
```

---

### Slow Performance

**Check:**
1. Is MLX using Metal GPU? (should be automatic)
2. Are operations MLX-native or falling back to CPU?
3. Is lazy evaluation being used effectively?
4. Right data types? (float16 vs float32)

**Profile:**
```python
import time
import mlx.core as mx

start = time.time()
result = model(input)
mx.eval(result)  # Force execution
print(f"Time: {time.time() - start:.3f}s")
```

---

### Incompatibility Errors

**Common causes:**
- Model from PyTorch/TensorFlow, needs conversion
- Tensor shape mismatches (PyTorch BCHW vs MLX BHWC)
- Missing preprocessing steps
- Wrong tokenizer/processor version

**Solution:**
- Create conversion script with diagnostics
- Validate all shapes match expectations
- Compare with reference working examples

---

## Quick Start Template

### New Project Checklist

```bash
# 1. Create structure
mkdir new_project && cd new_project
mkdir -p reference test
touch README.md TODO.md
touch reference/{LEARNINGS,DIAGNOSTICS,RESULTS}.md
touch test/README.md

# 2. Setup environment
conda create -n new_project python=3.11
conda activate new_project
pip install mlx mlx-lm  # adjust as needed
conda env export > environment.yml

# 3. Document initial state
cat > TODO.md << 'EOF'
# TODO: Project Name

## Objective
[What we're building]

## Current Status
- [ ] Environment setup
- [ ] Research phase
- [ ] Implementation
- [ ] Validation

## Next Steps
[What to do first]
EOF

# 4. Create first diagnostic
cat > test/check_basic.py << 'EOF'
#!/usr/bin/env python3
"""Basic diagnostic test"""
import mlx.core as mx

print("✅ MLX working")
print(f"MLX version: {mx.__version__}")
EOF

# 5. Test
python test/check_basic.py
```

---

## Technology-Specific Notes

### For LLM Projects
- MLX-LM package: `pip install mlx-lm`
- Conversion tools: `mlx_lm.convert`
- Quantization: Built into MLX-LM

### For VLM Projects  
- Check vision encoder separately
- Validate image preprocessing pipeline
- Test with sample images early

### For Diffusers/Image Generation
- Use `diffusers` with MLX backend when available
- Check scheduler compatibility carefully
- Validate VAE decoding produces correct images

### For Fine-tuning
- Use LoRA for efficiency
- Monitor memory closely during training
- Save checkpoints frequently
- Use gradient checkpointing if OOM

---

## File Naming Conventions

### Scripts
- `convert_*.py` - Conversion scripts
- `test_*.py` - Test scripts
- `validate_*.py` - Validation scripts
- `check_*.py` - Diagnostic scripts

### Documentation
- `README.md` - Project overview
- `TODO.md` - Current status
- `LEARNINGS.md` - Insights and rationale
- `DIAGNOSTICS.md` - Tool documentation
- `RESULTS.md` - Benchmarks and results

### Data/Models
- `config.json` - Model configuration
- `*.safetensors` - Model weights
- `environment.yml` - Conda environment
- `requirements.txt` - Pip requirements

---

## Resources

### Apple Silicon / MLX
- MLX Documentation: https://ml-explore.github.io/mlx/
- MLX Examples: https://github.com/ml-explore/mlx-examples
- Apple ML: https://developer.apple.com/machine-learning/

### Model Repositories
- Hugging Face: https://huggingface.co/
- MLX Community Models: https://huggingface.co/mlx-community

### Tools
- Activity Monitor (macOS) - Monitor resource usage
- `memory_pressure` - Check memory status
- `htop` - Process monitoring (install via brew)

---

## Summary

**Core Philosophy:**
- **Document first** - Maintain living documentation
- **Investigate before implementing** - Understand before changing
- **Test early** - Diagnostic scripts save hours
- **Use actual data** - Never assume, always verify
- **Leverage Apple Silicon** - MLX is optimized for M-series chips

**Success Formula:**
```
Investigation + Documentation + Diagnostics = Reliable Development
```

**Remember:**
- Read actual values, don't assume
- Build diagnostic tools before main code
- Compare with working examples
- Document why, not just what
- Keep documentation current
- Test on Mac Studio's full capabilities

---

## Version History

- **v1.0** (2025-11-27): Initial universal methodology for GenAI projects on Apple Silicon

---

**This is a living document. Evolve it with every project's unique learnings.**
