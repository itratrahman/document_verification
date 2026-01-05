# Data Directory

This directory contains all training and validation data for the document verification project.

## Directory Structure

```
data/
├── Original/                    # Generated license images (positive examples)
├── random_doc_images/           # Random document images (negative examples)
│   ├── blank_pages/
│   ├── book_front_covers/
│   ├── book_pages/
│   ├── books/
│   ├── driving_license/
│   ├── invoice/
│   ├── letters/
│   ├── national_certificates/
│   ├── newspapers/
│   ├── passport/
│   └── tax_documents/
└── truth_tables/                # Ground truth JSON annotations
```

## Overview

### Original/
Contains approximately 3,000+ generated license images in PNG format. These serve as **positive examples** for training the EfficientNet-B0 binary classifier.

- **File Pattern:** `generated_license_*.png`
- **Purpose:** Positive class training data
- **Used By:** Model training for license recognition

### random_doc_images/
Contains various document types organized by category. These serve as **negative examples** for training the model to distinguish licenses from other document types.

**Document Categories:**
- Blank pages
- Book front covers and pages
- Driving licenses
- Invoices
- Letters
- National certificates
- Newspapers
- Passports
- Tax documents

**Purpose:** Negative class training data to improve model robustness

### truth_tables/
Contains JSON metadata files corresponding to each license image in the Original directory.

- **File Pattern:** `generated_license_*.json`
- **Total Files:** ~3,000+ JSON files
- **Purpose:** Ground truth annotations for model evaluation and validation

## Usage

The data is organized for use with PyTorch's `ImageFolder` dataset class, which automatically assigns:
- **Negative class (0):** Images from `random_doc_images/`
- **Positive class (1):** Images from `Original/`

## Training Pipeline

1. **Data Loading:** `ImageFolder` reads images from subdirectories
2. **Preprocessing:** Images are resized to 224x224 and normalized using ImageNet statistics
3. **Augmentation:** Random crops, horizontal flips applied during training
4. **Splitting:** 80% training, 20% validation
5. **Validation:** Model predictions compared against truth_tables annotations

## Notes

- All images are converted to tensors and normalized using ImageNet mean and standard deviation
- The `create_dataloaders()` function in `model.py` handles data loading and preprocessing
- Class weights are computed to balance the dataset during loss calculation