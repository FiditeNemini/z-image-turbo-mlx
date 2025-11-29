# Diagnostic Tools

## model_diagnostics.py

**Purpose**: Scans the `/models/` directory and provides detailed information about model files without loading them fully into memory.

**Usage**:
```bash
python model_diagnostics.py
```

**Features**:
- Detects `.safetensors` and `.gguf` files.
- Reports file size and total parameter count.
- Analyzes data type distribution (e.g., fp16, fp32, int8).
- Inspects top-level tensor structure for `.safetensors`.
- Safe to run on large models (avoids OOM by reading metadata only).
