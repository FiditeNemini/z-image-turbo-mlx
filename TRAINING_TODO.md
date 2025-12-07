## Plan: LoRA Training & Model Fine-Tuning

Add the ability to train custom LoRAs and fine-tune models directly within the app, using datasets selected by the user.

## ✅ IMPLEMENTATION STATUS

### Completed
- [x] **Phase 1: Dataset Management** - `src/training/dataset.py`
  - DatasetManager class for dataset creation and management
  - TrainingDataset PyTorch Dataset class
  - Aspect ratio bucketing support
  - Caption file handling

- [x] **Phase 2: Training Configuration** - `src/training/config.py`
  - TrainingConfig, LoRAConfig, DatasetConfig dataclasses
  - Preset configurations (quick_test, character_lora, style_lora, concept_lora)
  - JSON save/load support

- [x] **Phase 3: Training Backend** - `src/training/`
  - `trainer.py` - Main LoRATrainer class with full training loop
  - `lora_network.py` - LoRA injection and weight management
  - `adapter.py` - Training adapter (Ostris de-distillation) support
  - `utils.py` - Training utilities, memory estimation, timers

- [x] **Phase 4: Training UI** - `src/training_ui.py` + `app.py`
  - Training tab added to Gradio interface
  - Dataset management UI (create, add images, validate)
  - Training configuration UI with presets
  - Progress monitoring
  - VRAM estimation

### Remaining Work
- [ ] **Testing** - Test full training pipeline with real GPU
- [ ] **Validation Images** - Improve validation image generation during training
- [ ] **WandB Integration** - Add Weights & Biases logging support
- [ ] **Auto-captioning** - Add BLIP/Florence-2 integration for automatic captions
- [ ] **Cloud Training** - Optional integration with cloud GPU services

---

## Technical Reference

### Background & Research

#### Key Resources
- **Ostris AI-Toolkit**: https://github.com/ostris/ai-toolkit - All-in-one training suite for diffusion models
- **Training Adapter**: https://huggingface.co/ostris/zimage_turbo_training_adapter - Required for training on turbo model
- **Z-Image-De-Turbo**: https://huggingface.co/ostris/Z-Image-De-Turbo - De-distilled version for longer training runs

#### Critical Technical Insight: The Distillation Problem

Z-Image-Turbo is a **step-distilled model** (generates in 8-9 steps instead of 25+). When training directly on distilled models:
- The distillation breaks down quickly during training
- Results in unpredictable loss of the step-distillation capability
- LoRAs trained this way won't work properly with the turbo model

**Two Solutions:**

| Approach | Model | Adapter | Best For | Inference Steps | CFG |
|----------|-------|---------|----------|-----------------|-----|
| **Turbo + Adapter** | `Tongyi-MAI/Z-Image-Turbo` | `zimage_turbo_training_adapter_v2.safetensors` | Short runs (styles, concepts, characters) | 8 | 1.0 |
| **De-Turbo** | `ostris/Z-Image-De-Turbo` | None | Longer training, fine-tuning | 20-30 | 2.0-3.0 |

#### How the Training Adapter Works

1. During **training**: Adapter is merged INTO the model (weight +1.0) - breaks distillation gradually
2. During **inference**: Adapter is INVERTED (weight -1.0) - removes adapter effects, keeps LoRA effects
3. Your LoRA learns only the subject, not the distillation

#### Platform Limitation: MLX vs PyTorch

⚠️ **Training is PyTorch/CUDA only** - The AI-Toolkit does NOT support MLX training.

**Strategy**: Implement PyTorch training backend while keeping MLX for inference.

---

### Implementation Plan

#### Phase 1: Dataset Management UI

**1.1 Create Dataset Tab in `app.py`**
- Add new "📁 Datasets" tab next to Generate and Model Settings
- Dataset browser showing available datasets in `datasets/` folder
- Create new dataset wizard with name, type (character/style/concept), description

**1.2 Dataset Structure** (`datasets/<name>/`)
```
datasets/
├── my_character/
│   ├── dataset.json         # Metadata, captions config
│   ├── images/
│   │   ├── 001.png
│   │   ├── 001.txt          # Caption file
│   │   ├── 002.jpg
│   │   └── 002.txt
│   └── samples/             # Training sample outputs
└── anime_style/
    └── ...
```

**1.3 Image Management**
- Drag-and-drop image upload
- Auto-captioning using BLIP/Florence-2 (optional)
- Manual caption editing per image
- Image preview gallery
- Bulk caption editing (prefix/suffix)

**1.4 Dataset Configuration** (`dataset.json`)
```json
{
  "name": "my_character",
  "type": "character",
  "trigger_word": "ohwx person",
  "description": "Training data for character X",
  "resolution": 1024,
  "num_images": 20,
  "created": "2024-12-05",
  "caption_prefix": "",
  "caption_suffix": ""
}
```

---

#### Phase 2: Training Configuration UI

**2.1 Create Training Tab in `app.py`**
- Add "🎓 Training" tab
- Training type selector: LoRA / Full Fine-tune
- Model selector (base model to train on)
- Dataset selector (from datasets created in Phase 1)

**2.2 Training Mode Selection**

| Mode | Description | Model | VRAM |
|------|-------------|-------|------|
| **LoRA (Turbo)** | Quick LoRA on turbo model | Z-Image-Turbo + Adapter | ~12GB |
| **LoRA (De-Turbo)** | Longer LoRA training | Z-Image-De-Turbo | ~12GB |
| **Fine-tune** | Full model training | Z-Image-De-Turbo | ~24GB+ |

**2.3 Training Parameters UI**

```
Basic Settings:
├── Training Type: [LoRA ▼] [Fine-tune ▼]
├── Base Model: [Z-Image-Turbo ▼] [Z-Image-De-Turbo ▼]
├── Dataset: [my_character ▼]
├── Output Name: [my_character_lora]
└── Trigger Word: [ohwx person]

LoRA Settings:
├── Rank (dim): [32] (8-128, higher = more capacity)
├── Alpha: [32] (usually same as rank)
└── Target Modules: [☑ Transformer] [☐ Text Encoder]

Training Settings:
├── Learning Rate: [1e-4]
├── Steps: [1000]
├── Batch Size: [1]
├── Gradient Accumulation: [4]
├── Optimizer: [AdamW8bit ▼]
└── LR Scheduler: [cosine ▼]

Advanced:
├── Quantize Model: [☑] (reduces VRAM)
├── Quantize Text Encoder: [☑]
├── Mixed Precision: [bf16 ▼]
├── Gradient Checkpointing: [☑]
└── Sample Every N Steps: [100]

Sample Settings:
├── Sample Prompt: [ohwx person, portrait photo]
├── Sample Steps: [8] (turbo) / [25] (de-turbo)
└── Sample CFG: [1.0] (turbo) / [3.0] (de-turbo)
```

---

#### Phase 3: Training Backend (`src/training/`)

**3.1 Create Training Module Structure**
```
src/training/
├── __init__.py
├── trainer.py           # Main training orchestration
├── config.py            # Training config dataclass
├── dataset.py           # Dataset loading and processing
├── lora_network.py      # LoRA network implementation
├── adapter.py           # Training adapter handling
└── utils.py             # Training utilities
```

**3.2 `src/training/config.py`** - Training Configuration
```python
@dataclass
class TrainingConfig:
    # Model
    model_path: str
    model_type: str  # "turbo" or "de_turbo"
    use_training_adapter: bool
    
    # Dataset
    dataset_path: str
    resolution: int = 1024
    
    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    train_text_encoder: bool = False
    
    # Training
    learning_rate: float = 1e-4
    num_steps: int = 1000
    batch_size: int = 1
    gradient_accumulation: int = 4
    optimizer: str = "adamw8bit"
    lr_scheduler: str = "cosine"
    
    # Memory optimization
    quantize: bool = True
    quantize_te: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    
    # Output
    output_name: str = "trained_lora"
    output_dir: str = "models/loras/trained/"
    save_every_n_steps: int = 500
    
    # Sampling
    sample_every_n_steps: int = 100
    sample_prompt: str = ""
    sample_steps: int = 8
    sample_cfg: float = 1.0
```

**3.3 `src/training/trainer.py`** - Main Training Class
```python
class ZImageTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda"  # PyTorch/CUDA only
        
    def setup(self):
        """Load model, create LoRA network, setup optimizer"""
        # 1. Load base model (diffusers ZImagePipeline)
        # 2. Load training adapter if using turbo model
        # 3. Create LoRA network
        # 4. Setup optimizer and scheduler
        # 5. Setup dataset and dataloader
        
    def train_step(self, batch):
        """Single training step"""
        # 1. Get noisy latents
        # 2. Get text embeddings
        # 3. Predict noise
        # 4. Calculate loss
        # 5. Backward pass
        
    def save_checkpoint(self, step):
        """Save LoRA weights"""
        
    def generate_sample(self, step):
        """Generate sample image during training"""
        # For turbo: invert adapter, apply LoRA, generate at 8 steps
        # For de-turbo: just apply LoRA, generate at 25 steps
        
    def train(self, progress_callback=None):
        """Main training loop"""
```

**3.4 `src/training/adapter.py`** - Training Adapter Handling
```python
class TrainingAdapter:
    """Handles the zimage_turbo_training_adapter"""
    
    ADAPTER_REPO = "ostris/zimage_turbo_training_adapter"
    ADAPTER_FILE = "zimage_turbo_training_adapter_v2.safetensors"
    
    def download_if_needed(self) -> str:
        """Download adapter from HuggingFace if not cached"""
        
    def merge_into_model(self, model, weight=1.0):
        """Merge adapter INTO model for training"""
        
    def create_inverse_lora(self, model):
        """Create inverted LoRA for inference sampling"""
```

**3.5 `src/training/lora_network.py`** - LoRA Implementation
```python
class LoRANetwork:
    """LoRA network for Z-Image transformer"""
    
    def __init__(self, model, rank, alpha, target_modules):
        # Create LoRA layers for target modules
        
    def inject_into_model(self, model):
        """Replace linear layers with LoRA-wrapped versions"""
        
    def get_trainable_params(self):
        """Return only LoRA parameters for optimizer"""
        
    def save(self, path):
        """Save LoRA weights as safetensors"""
        
    def get_state_dict(self):
        """Get LoRA state dict for saving"""
```

---

#### Phase 4: Training UI Integration

**4.1 Training Progress Display**
```
Training: my_character_lora
━━━━━━━━━━━━━━━━━━━━ 45% (450/1000 steps)

Step: 450 | Loss: 0.0234 | LR: 8.5e-5
ETA: 12 minutes remaining

[Sample at step 400]
[image preview]

[Stop Training] [Pause] [Save Now]
```

**4.2 Training Log**
- Real-time loss graph
- Sample image gallery (generated during training)
- Training metrics (loss, learning rate, VRAM usage)
- Console output

**4.3 Post-Training**
- Auto-save LoRA to `models/loras/trained/`
- Option to add trigger words to metadata
- Option to test immediately in Generate tab
- Training report (final loss, total time, samples)

---

#### Phase 5: Model Fine-Tuning (Optional/Advanced)

**5.1 Full Fine-tune Mode**
- Train all model weights (not just LoRA)
- Requires Z-Image-De-Turbo as base
- Higher VRAM requirements (24GB+)
- Save as new model in `models/mlx/` or `models/pytorch/`

**5.2 Checkpoint Management**
- Save intermediate checkpoints
- Resume training from checkpoint
- Convert fine-tuned PyTorch model to MLX

---

### Dependencies to Add

```
# requirements-training.txt (separate from main requirements)
torch>=2.1.0
torchvision>=0.16.0
accelerate>=0.25.0
bitsandbytes>=0.41.0      # For 8-bit optimizers
optimum-quanto>=0.1.0     # For quantization
safetensors>=0.4.0
transformers>=4.36.0
diffusers>=0.25.0         # With ZImagePipeline support
datasets>=2.15.0
```

---

### File Structure After Implementation

```
z-image-turbo-mlx/
├── app.py                    # Updated with Training tab
├── src/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── lora_network.py
│   │   ├── adapter.py
│   │   └── utils.py
│   └── ... (existing files)
├── datasets/                 # User datasets
│   └── .gitkeep
├── models/
│   ├── loras/
│   │   ├── trained/          # Output for trained LoRAs
│   │   └── ... (user LoRAs)
│   └── training_adapter/     # Cached training adapter
└── requirements-training.txt
```

---

### Implementation Order

| Phase | Description | Effort | Priority |
|-------|-------------|--------|----------|
| **Phase 1** | Dataset Management UI | 2-3 days | High |
| **Phase 2** | Training Configuration UI | 1-2 days | High |
| **Phase 3** | Training Backend | 3-5 days | High |
| **Phase 4** | Training UI Integration | 2-3 days | High |
| **Phase 5** | Full Fine-tuning | 2-3 days | Low |

**Total Estimated Effort**: 10-16 days

---

### Further Considerations

1. **Auto-captioning Integration** - Add BLIP-2 or Florence-2 for automatic image captioning in dataset creation

2. **Cloud Training Option** - For users without NVIDIA GPU, integrate with cloud training services (RunPod, Lambda Labs)

3. **Training Presets** - Pre-configured settings for common use cases (character, style, concept)

4. **LoRA Merging** - Tool to merge multiple LoRAs into one

5. **MLX Training Future** - Monitor MLX development for potential future training support
