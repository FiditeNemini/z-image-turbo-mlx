---
trigger: always_on
---

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