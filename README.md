# Document Verification Project

This repository contains a deep learning-based document verification system using EfficientNet-B0 for binary classification of license documents. The project automatically distinguishes between authentic license images and other document types using a trained neural network.

## Overview

- **Binary Classification Model**: EfficientNet-B0 neural network trained to classify documents as licenses (positive) or other document types (negative)
- **Balanced Dataset Training**: Uses inverse frequency class weighting to handle imbalanced training data
- **Comprehensive Dataset**: Trains on ~3,000+ license images and diverse negative examples from multiple document categories
- **Reproducible Training**: Deterministic splits and seed management for reproducible model training

## Features

- Automated license document detection and classification
- Binary classification: License vs. Non-License documents
- Class-weighted loss for handling imbalanced datasets
- Validation/Test split with deterministic seeding
- Comprehensive logging for training monitoring
- Model checkpointing (best and final models)
- Support for CPU and GPU training

## Project Structure

```
document_verification/
├── model.py                    # Main training script with EfficientNet-B0 model
├── requirements.txt            # Project dependencies
├── README.md                   # This file
├── data/                       # Training and validation data
│   ├── Original/               # Positive class: ~3,000+ license images
│   ├── random_doc_images/      # Negative class: diverse document types
│   │   ├── blank_pages/
│   │   ├── book_front_covers/
│   │   ├── book_pages/
│   │   ├── books/
│   │   ├── driving_license/
│   │   ├── invoice/
│   │   ├── letters/
│   │   ├── national_certificates/
│   │   ├── newspapers/
│   │   ├── passport/
│   │   └── tax_documents/
│   └── truth_tables/           # Ground truth JSON annotations (~3,000+ files)
├── checkpoints/                # Model checkpoints (created during training)
├── logs/                       # Training logs (created during training)
└── models/                     # Additional model files and resources
```

## Data Structure

### Positive Class (License Images)
- **Location**: `data/Original/`
- **Count**: ~3,000+ PNG images
- **Purpose**: Training positive examples for license recognition

### Negative Class (Other Documents)
- **Location**: `data/random_doc_images/`
- **Categories**: Blank pages, books, invoices, letters, certificates, newspapers, passports, tax documents, driving licenses
- **Purpose**: Training diverse negative examples to improve model robustness

### Ground Truth
- **Location**: `data/truth_tables/`
- **Format**: JSON metadata files corresponding to Original/ images
- **Purpose**: Validation and evaluation of model predictions

## Model Architecture

- **Base Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Input Size**: 224×224 pixels
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Output**: Binary classification (2 classes)
- **Optimizer**: Adam (lr=1e-4)
- **Loss Function**: CrossEntropyLoss with class weights
- **Data Augmentation**: Random resize crop, horizontal flip, normalization

## Getting Started

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**  
   - Ensure `data/Original/` contains license images
   - Ensure `data/random_doc_images/` contains non-license documents organized by type
   - Data structure is automatically handled by the training script

3. **Train the Model**  
   ```bash
   python model.py
   ```

4. **Monitor Training**  
   - Check logs in `logs/train_log_*.txt`
   - Model checkpoints saved in `checkpoints/`
   - Best model: `checkpoints/best_efficientnet_binary.pt`
   - Final model: `checkpoints/final_efficientnet_binary.pt`

## Configuration

Edit the following parameters in `model.py` to customize training:

- `NUM_EPOCHS`: Number of training epochs (default: 10)
- `BATCH_SIZE`: Batch size for training (default: 32)
- `LEARNING_RATE`: Adam optimizer learning rate (default: 1e-4)
- `VAL_SPLIT`: Validation split ratio (default: 0.2)
- `SEED`: Random seed for reproducibility (default: 42)

## Training Process

1. **Data Loading**: CustomDataset loads images from Original/ and random_doc_images/
2. **Label Assignment**: Original → class 1, random_doc_images → class 0
3. **Preprocessing**: Resize to 224×224, apply augmentation, normalize
4. **Train/Val Split**: 80% training, 20% validation (deterministic)
5. **Class Weighting**: Inverse frequency weights computed to balance classes
6. **Training Loop**: 
   - Forward pass through EfficientNet-B0
   - Weighted cross-entropy loss
   - Backpropagation and gradient updates
   - Validation on hold-out set each epoch
7. **Model Checkpointing**: Best model saved when validation accuracy improves
8. **Final Output**: Best and final models saved to checkpoints/

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- Pillow (PIL)
- NumPy
- See `requirements.txt` for complete list

## Best Practices

- Ensure all image files in data directories are valid PNG/JPG/JPEG formats
- Check logs in `logs/` for training progress and debugging
- Use consistent image quality for best results
- Keep seed value constant for reproducible results
- Monitor GPU memory usage when increasing batch size

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive open-source license that allows you to freely use, modify, and distribute this software in personal and commercial projects, provided you include the original license and copyright notice.

---

*This project uses deep learning to automate license document verification and classification.*