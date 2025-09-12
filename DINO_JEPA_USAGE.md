# DINO/JEPA Integration in SAM2

This repository has been enhanced to support using DINO and I-JEPA models as image encoders instead of the original SAM2 Hiera backbone. This allows you to leverage powerful self-supervised vision transformers as the feature extraction component of SAM2.

## Overview

The integration provides:
- **Multi-scale feature synthesis**: Converts single-scale ViT outputs to multi-scale pyramid features compatible with SAM2's FPN neck
- **Flexible model loading**: Supports any HuggingFace ViT model (DINOv2, I-JEPA, etc.)
- **Training integration**: Command-line arguments for easy encoder switching during training
- **Backward compatibility**: Existing SAM2 functionality remains unchanged

## Configuration Files

Two new configuration files are provided:

- `sam2/sam2_vit_dino.yaml` - For using DINO/DINOv2 models
- `sam2/sam2_vit_ijepa.yaml` - For using I-JEPA models

## Usage Examples

### 1. Using Pre-configured YAML Files

```python
from sam2.build_sam import build_sam2

# Load SAM2 with DINO encoder
model = build_sam2(
    config_file="sam2_vit_dino.yaml",
    ckpt_path=None,  # No pretrained SAM2 weights when using different encoder
    device="cuda"
)

# Load SAM2 with I-JEPA encoder  
model = build_sam2(
    config_file="sam2_vit_ijepa.yaml", 
    ckpt_path=None,
    device="cuda"
)
```

### 2. Training with Command-Line Arguments

```bash
# Train with DINO encoder
python training/train.py \
    --config configs/sam2_training/base_config.yaml \
    --encoder-type dino \
    --encoder-ckpt facebook/dinov2-large \
    --encoder-out-dims 1024,1024,1024,1024

# Train with I-JEPA encoder
python training/train.py \
    --config configs/sam2_training/base_config.yaml \
    --encoder-type ijepa \
    --encoder-ckpt /path/to/ijepa/model \
    --freeze-vit \
    --force-dtype fp16
```

### 3. Custom Configuration

```python
from omegaconf import OmegaConf
from sam2.modeling.backbones.vit_multiscale import ViTTrunkMultiScale
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck

# Create custom trunk with your preferred ViT model
trunk = ViTTrunkMultiScale(
    pretrained="facebook/dinov2-large",  # Any HuggingFace ViT model
    encoder_type="dino",
    out_dims=[1024, 1024, 1024, 1024],  # Output channel dimensions for each scale
    upsample_mode="bilinear",  # or "deconv"
    refine_highres=True,  # Use depthwise separable convs for refinement
    freeze_vit=False,  # Whether to freeze ViT weights
    force_dtype=None,  # Force output dtype: None, "fp16", "bf16", "fp32"
    verbose=True  # Print feature shapes
)
```

## Command-Line Arguments

When training, the following new arguments are available:

- `--encoder-type {hiera,dino,ijepa}` - Type of encoder to use
- `--encoder-ckpt PATH` - HuggingFace model ID or local directory path
- `--encoder-out-dims C32,C16,C8,C4` - Output channel dimensions (comma-separated)
- `--encoder-upsample-mode {bilinear,deconv}` - Upsampling strategy for multi-scale synthesis
- `--no-refine-highres` - Disable refinement convolutions on high-res features
- `--freeze-vit` - Freeze ViT backbone weights during training
- `--force-dtype {bf16,fp16,fp32}` - Force specific output dtype
- `--vit-verbose` - Print synthesized multi-scale feature shapes

## Technical Details

### Multi-Scale Feature Synthesis

The `ViTTrunkMultiScale` class converts single-scale ViT patch features to a 4-level pyramid:

- **F4** (stride 4): Highest resolution, synthesized by 4× upsampling
- **F8** (stride 8): Synthesized by 2× upsampling  
- **F16** (stride 16): Base ViT patch resolution
- **F32** (stride 32): Lowest resolution, synthesized by 2× downsampling

### Channel Dimension Handling

The system automatically handles channel dimension differences:
- Default: All scales use the same channel dimension as the ViT hidden size
- Custom: Specify different dimensions for each scale via `out_dims`
- Auto-inference: Reads `hidden_size` from HuggingFace model config when possible

### Memory and Performance

- **Memory optimization**: Supports dtype casting (fp16/bf16) to reduce memory usage
- **Selective freezing**: Can freeze ViT weights while training other components
- **Compilation**: Compatible with `torch.compile()` for additional speedup

## Dependencies

The DINO/JEPA integration requires:
- `transformers>=4.21.0` (for loading HuggingFace ViT models)
- All standard SAM2 dependencies

## Troubleshooting

### Common Issues

1. **Config loading errors**: Ensure you run the training script or register resolvers:
   ```python
   from training.utils.train_utils import register_omegaconf_resolvers
   register_omegaconf_resolvers()
   ```

2. **Model not found**: Use correct HuggingFace model IDs or local paths:
   ```python
   # Good: HuggingFace model ID
   "facebook/dinov2-large"
   
   # Good: Local directory with config.json
   "/path/to/my/model/"
   ```

3. **Channel dimension mismatch**: Ensure `backbone_channel_list` matches your `out_dims`:
   ```yaml
   trunk:
     out_dims: [1024, 1024, 1024, 1024]
   neck:
     backbone_channel_list: [1024, 1024, 1024, 1024]  # Must match
   ```

### Performance Tips

- Use `freeze_vit=True` when fine-tuning to reduce memory usage
- Set `force_dtype="fp16"` or `force_dtype="bf16"` for large models
- Use `refine_highres=False` to disable refinement convolutions if not needed

## Validation

Run the integration test to verify everything is working:

```bash
python test_dino_jepa_integration.py
```

This will test:
- Configuration loading and inheritance
- Component compatibility  
- Training integration
- Parameter validation