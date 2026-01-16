# Models Directory

This directory contains trained EfficientNet-B0 binary classifier models for document verification, along with their metadata and performance metrics.

## Model Versioning Strategy

Models are automatically versioned using timestamps and validation F1 scores following MLOps best practices:

- **Version Format**: `efficientnet_binary_v{YYYYMMDD_HHMMSS}_f1_{score}.pt`
- **Primary Metric**: Validation F1 score (better for imbalanced datasets)
- **Metadata**: Each model includes a JSON file with comprehensive metrics

## File Structure

```
models/
├── efficientnet_binary_v20260115_143052_f1_0.9534.pt     # Versioned checkpoint
├── efficientnet_binary_v20260115_143052_metadata.json    # Model metadata
├── best_efficientnet_binary.pt                            # Best model (legacy)
├── final_efficientnet_binary.pt                           # Final epoch (legacy)
└── README.md                                              # This file
```

## Model Metadata Format

Each versioned model includes a metadata JSON file containing:

```json
{
  "version": "20260115_143052",
  "timestamp": "2026-01-15T14:30:52.123456",
  "epoch": 8,
  "metrics": {
    "val_f1": 0.9534,
    "val_acc": 0.9512,
    "val_precision": 0.9456,
    "val_recall": 0.9614,
    "val_specificity": 0.9321,
    "val_balanced_acc": 0.9467,
    "val_mcc": 0.8923,
    "val_loss": 0.1234
  },
  "confusion_matrix": {
    "tp": 742,
    "tn": 28,
    "fp": 2,
    "fn": 1
  },
  "model_file": "efficientnet_binary_v20260115_143052_f1_0.9534.pt",
  "model_arch": "efficientnet_b0"
}
```

## Model Selection

**At Inference Time:**
- The API automatically loads the model with the **highest validation F1 score**
- Scans all `*_metadata.json` files to find the best performer
- Falls back to `best_efficientnet_binary.pt` if no metadata found

**Manual Selection:**
- Use the `/reload-model` endpoint with a specific model path
- Specify custom model via `model_path` parameter

## Model Architecture

- **Base**: EfficientNet-B0 (pretrained on ImageNet-1k)
- **Input**: 224×224 RGB images
- **Output**: 2 classes (binary: license vs non-license)
- **Size**: ~21 MB per checkpoint
- **Training**: Weighted CrossEntropyLoss with inverse frequency class balancing

## Training Details

- **Dataset**: 3,866 images (3,000 licenses, 866 other documents)
- **Split**: 80% train, 20% validation
- **Augmentation**: Random crop, horizontal flip, ImageNet normalization
- **Optimizer**: Adam (lr=1e-4)
- **Epochs**: 10
- **Best Metric**: Validation F1 score (saved when improved)

## MLflow Integration

All training runs are tracked in MLflow with:
- Hyperparameters and dataset statistics
- Epoch-level and batch-level metrics
- Model artifacts and metadata files
- Version tags for reproducibility

View experiments:
```bash
mlflow ui --backend-store-uri ../mlruns
```

## Usage Examples

**Load Best Model Automatically:**
```python
from model import create_model
import torch

model = create_model(num_classes=2)
# API will load best F1 model automatically
```

**Load Specific Version:**
```python
model_path = "models/efficientnet_binary_v20260115_143052_f1_0.9534.pt"
model.load_state_dict(torch.load(model_path, map_location='cpu'))
```

**Reload Model via API:**
```bash
# Default: loads best F1 model
curl -X POST http://localhost:8000/reload-model

# Specific model
curl -X POST http://localhost:8000/reload-model \
  -H "Content-Type: application/json" \
  -d '{"model_path": "models/efficientnet_binary_v20260115_143052_f1_0.9534.pt"}'
```

## Best Practices

✅ **Do:**
- Keep versioned models and their metadata files together
- Use F1 score for model selection (better for imbalanced data)
- Log all experiments to MLflow for tracking
- Document model performance in metadata JSON

❌ **Don't:**
- Delete metadata files (needed for automatic model selection)
- Commit large model files to git (use Git LFS or model registry)
- Rename versioned files (breaks metadata linking)

---

*Automated model versioning ensures reproducibility and enables intelligent model selection based on validation performance.*