# Agent Instructions for GenAI Development Projects

**Version:** 1.0  
**Last Updated:** 2025-11-27  
**Purpose:** Universal methodology for developing, debugging, and deploying GenAI projects (LLMs, VLMs, ML, Image Generation) on macOS with Apple Silicon.

---

## Development Environment

### Hardware & OS
- **Platform:** Mac Studio M2 Ultra (192GB RAM)
- **OS:** macOS (Apple Silicon)
- **Shell:** zsh
- **Primary Language:** Python

### Technology Stack
- **ML Framework:** MLX (Apple's machine learning framework)
- **Alternative Frameworks:** PyTorch, TensorFlow (when needed)
- **Image Generation:** Diffusers, Stable Diffusion
- **Model Types:** LLMs, VLMs, general ML models
- **Package Management:** conda, pip, homebrew

---

## Core Principles (Universal)

### 1. Documentation-First Development

**CRITICAL:** Every project MUST maintain these living documents:

```
project/
├── README.md           # Project overview, setup, usage
├── TODO.md             # Current status, tasks, what's broken/working
├── reference/
│   ├── LEARNINGS.md    # Critical insights, failed approaches, rationale
│   ├── DIAGNOSTICS.md  # Diagnostic tools and procedures
│   └── RESULTS.md      # Benchmarks, performance metrics, known issues
└── test/
    └── README.md       # Test suite documentation
```

**Why:** Context window limits and session boundaries cause knowledge loss. Documentation enables continuity across sessions and prevents reintroducing fixed bugs.

---

### 2. Investigation Before Implementation

**Never make changes without understanding the architecture.**

**Process:**
1. **Read source data** - Model architectures, configs, actual tensor shapes
2. **Compare with requirements** - What does the target system expect?
3. **Document findings** in reference/LEARNINGS.md
4. **Identify mismatches** explicitly
5. **Only then** make targeted fixes

**For LLM/VLM Projects:**
- Inspect model configs (config.json, model architecture)
- Check tensor shapes and dimensions
- Validate tokenizer compatibility
- Understand quantization requirements (MLX uses specific formats)

**For Image Generation Projects:**
- Check model architecture (UNet, VAE, text encoder)
- Validate scheduler compatibility
- Understand pipeline requirements
- Check image dimension requirements

**For ML Projects:**
- Understand input/output shapes
- Validate preprocessing requirements
- Check model compatibility with MLX
- Understand tensor layouts (MLX uses different conventions than PyTorch)

---

### 3. Test-Driven Validation

**Build diagnostic tools BEFORE full runs.**

**Required Test Suite:**
- **Configuration validators** - Verify configs before loading models
- **Shape validators** - Check tensor dimensions match expectations
- **Compatibility checks** - Compare with known working setups
- **Load tests** - Verify models load successfully
- **Inference tests** - Quick sanity checks on outputs

**Why:** Full model operations take minutes to hours. Diagnostic tests take seconds. Find issues early.

---

### 4. Apple Silicon / MLX Specifics

#### Memory Management
- **Unified Memory:** 192GB shared between CPU and GPU
- **Monitor usage:** Use `memory_pressure` or Activity Monitor
- **MLX advantages:** Efficient memory usage, lazy evaluation
- **Watch for:** Memory leaks in long-running processes

#### MLX Framework Specifics
```python
import mlx.core as mx
import mlx.nn as nn

# MLX uses different tensor layouts than PyTorch
# PyTorch: [batch, channels, height, width]
# MLX: Often [batch, height, width, channels]

# Lazy evaluation - operations only execute when needed
x = mx.array([1, 2, 3])
y = x + 1  # Not executed yet
mx.eval(y)  # Now executed
```

#### Conda Environment Best Practices
```bash
# Always use dedicated environments
conda create -n project_name python=3.11
conda activate project_name

# Install MLX
pip install mlx mlx-lm mlx-transformers

# Document environment
conda env export > environment.yml
```

#### Homebrew Integration
```bash
# Homebrew Caskroom location for conda
/opt/homebrew/Caskroom/miniconda/base/envs/

# Always use full path for production scripts
/opt/homebrew/Caskroom/miniconda/base/envs/project_name/bin/python script.py
```

---

## Project-Specific Guidelines

### LLM Projects

**Common Tasks:**
- Model conversion (GGUF → MLX, PyTorch → MLX)
- Quantization (4-bit, 8-bit for Apple Silicon)
- Fine-tuning with LoRA/QLoRA
- Inference optimization

**Critical Files to Check:**
- `config.json` - Model architecture configuration
- `tokenizer.json` - Tokenizer configuration
- Weight files (`.safetensors`, `.gguf`)

**Common Issues:**
- Vocab size mismatches
- Context length incompatibilities
- Quantization format issues
- Attention mechanism differences

**Diagnostic Pattern:**
```python
# Quick model inspection
from transformers import AutoConfig
config = AutoConfig.from_pretrained("path/to/model")
print(f"Vocab size: {config.vocab_size}")
print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")
```

---

### VLM Projects

**Common Tasks:**
- Vision-language model conversion
- Image preprocessing pipeline setup
- Multi-modal inference
- Vision encoder integration

**Critical Files to Check:**
- `config.json` - Includes vision_config and text_config
- `preprocessor_config.json` - Image preprocessing settings
- Vision projector weights
- Image processor configuration

**Common Issues:**
- Vision encoder depth mismatches
- Image token handling errors
- Preprocessor configuration incompatibilities
- Projection dimension mismatches

**Diagnostic Pattern:**
```python
# Check vision config
import json
with open("config.json") as f:
    config = json.load(f)
    print(f"Vision depth: {config['vision_config']['depth']}")
    print(f"Vision hidden size: {config['vision_config']['hidden_size']}")
    print(f"Architecture: {config['architectures']}")
```

---

### Image Generation Projects (Diffusers)

**Common Tasks:**
- Pipeline setup (Stable Diffusion, FLUX, etc.)
- Model conversion to MLX format
- Custom scheduler implementation
- LoRA integration

**Critical Files to Check:**
- `model_index.json` - Pipeline component mapping
- Scheduler config
- UNet/Transformer config
- VAE config
- Text encoder config

**Common Issues:**
- Scheduler incompatibilities
- VAE scaling factor mismatches
- Text encoder max length issues
- Image size requirements

**Diagnostic Pattern:**
```python
# Check pipeline components
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("path/to/model")
print(f"UNet: {pipe.unet.config}")
print(f"Scheduler: {pipe.scheduler.config}")
print(f"VAE scaling: {pipe.vae.config.scaling_factor}")
```

---

### General ML Projects

**Common Tasks:**
- PyTorch → MLX conversion
- Custom model implementation
- Training pipeline setup
- Data preprocessing

**Critical Considerations:**
- MLX uses different array layouts than PyTorch/NumPy
- Gradient computation differences
- Optimizer implementations may differ
- Custom operations may need MLX-specific implementations

---

## Development Workflow

### Phase 1: Project Setup (Every New Project)

1. **Create Project Structure**
   ```bash
   mkdir project_name && cd project_name
   mkdir reference test
   touch README.md TODO.md
   touch reference/{LEARNINGS,DIAGNOSTICS,RESULTS}.md
   touch test/README.md
   ```

2. **Initialize Environment**
   ```bash
   conda create -n project_name python=3.11
   conda activate project_name
   pip install mlx mlx-lm  # or other requirements
   conda env export > environment.yml
   ```

3. **Document Initial State**
   ```markdown
   # TODO.md
   ## Project: [Name]
   
   ### Objective
   [What are we trying to achieve?]
   
   ### Current Status
   - [ ] Environment setup
   - [ ] Initial research
   - [ ] Implementation
   - [ ] Validation
   ```

---

### Phase 2: Investigation

1. **Understand Requirements**
   - What model/pipeline are we working with?
   - What's the source format?
   - What's the target format/use case?
   - What are the constraints? (memory, performance)

2. **Find Reference Examples**
   - Locate working examples of similar projects
   - Study successful implementations
   - Note configuration patterns

3. **Document Architecture**
   ```markdown
   # reference/LEARNINGS.md
   
   ## Architecture: [Model Name]
   
   ### Source Format
   - [Key parameters]
   - [Important fields]
   
   ### Target Requirements
   - [Expected values]
   - [Critical settings]
   
   ### Known Issues
   - [Common pitfalls]
   ```

---

### Phase 3: Diagnostic Development

**Before writing main code, create diagnostic tools:**

```python
# test/check_config.py
"""Validate model configuration"""
import json
import sys

def validate_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    
    # Check required fields
    required = ["model_type", "hidden_size"]
    for field in required:
        assert field in config, f"Missing: {field}"
    
    print("✅ Config valid")

if __name__ == "__main__":
    validate_config(sys.argv[1])
```

**Document in test/README.md:**
- What each diagnostic does
- How to run it
- What "pass" looks like

---

### Phase 4: Implementation

1. **Start Small**
   - Implement one component at a time
   - Test after each change
   - Document rationale for decisions

2. **MLX-Specific Considerations**
   ```python
   # Memory efficiency
   import mlx.core as mx
   
   # Use lazy evaluation
   result = some_operation(input)  # Not executed yet
   mx.eval(result)  # Executed now, memory freed after
   
   # Explicit memory management for large models
   import gc
   del large_model
   mx.metal.clear_cache()  # Clear MLX Metal cache
   gc.collect()
   ```

3. **Handle Apple Silicon Optimally**
   ```python
   # Use MLX-optimized operations
   import mlx.nn as nn
   
   # MLX operations are Metal-accelerated
   # Prefer MLX operations over NumPy when possible
   
   # For large matrix operations
   mx.matmul(a, b)  # Uses Metal GPU
   ```

---

### Phase 5: Validation

**Validation Checklist:**

- [ ] **Configuration valid**
  ```bash
  python test/check_config.py config.json
  ```

- [ ] **Model loads successfully**
  ```python
  import mlx.nn as nn
  model = load_model("path/to/model")
  print("✅ Model loaded")
  ```

- [ ] **Basic inference works**
  ```python
  output = model(test_input)
  assert output.shape == expected_shape
  print("✅ Inference successful")
  ```

- [ ] **Memory usage acceptable**
  ```bash
  # Monitor during operation
  memory_pressure
  ```

- [ ] **Performance benchmarked**
  ```python
  import time
  start = time.time()
  output = model(input)
  mx.eval(output)  # Force execution
  print(f"Time: {time.time() - start:.2f}s")
  ```

---

## Session Continuity Protocol

### At Start of Every Session

**ALWAYS READ THESE FIRST** (in order):

1. **TODO.md** - What's the current status? What's broken? What's next?
2. **reference/LEARNINGS.md** - Why were decisions made? What failed?
3. **reference/DIAGNOSTICS.md** - What tools are available?
4. **reference/RESULTS.md** - What were the last results?

**Then:**
- Activate correct conda environment
- Check if any files changed externally
- Run diagnostic tests to verify current state

### During Session

**Update as you go:**
- Mark tasks complete in TODO.md: `[x]`
- Mark in-progress: `[/]`
- Log insights in reference/LEARNINGS.md
- Document new tools in reference/DIAGNOSTICS.md

### At End of Session

**Before stopping:**
1. Update TODO.md with current status
2. Document any learnings
3. Note next steps clearly
4. Save all files

```markdown
# TODO.md

### Session End: 2025-11-27 18:00

**Completed:**
- [x] Model conversion working
- [x] Validation tests passing

**In Progress:**
- [/] Performance optimization

**Next Steps:**
- [ ] Benchmark against reference
- [ ] Deploy to production
```

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
