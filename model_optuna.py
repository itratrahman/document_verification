# Import the os module to work with file paths and directories
import os  # used for path joins and directory creation

# Import the random module for Python's built-in random number generation
import random  # used for seeding Python RNG

# Import the logging module to handle logs
import logging  # used to log information to console and file

# Import numpy for numerical operations and RNG seeding
import numpy as np  # used for seeding NumPy RNG

# Import torch, the core PyTorch library
import torch  # used for tensors and neural network operations

# Import nn for neural network layers and optim for optimization algorithms
import torch.nn as nn  # used for defining neural network modules
import torch.optim as optim  # used for optimizers like Adam

# Import DataLoader and random_split to handle datasets and splitting
from torch.utils.data import DataLoader, random_split  # used for batching and dataset splits

# Import torchvision datasets and transforms for image loading and preprocessing
from torchvision import datasets, transforms, models  # used for image datasets, transforms, and pretrained models

# Import PIL Image for loading images
from PIL import Image  # used for loading and converting image files

# Import time to measure training duration and create Unix timestamps
import time  # used to compute elapsed time and epoch time

# Import copy to clone model weights when tracking the best model
import copy  # used for deep copying model state_dict

# Import datetime to get human-readable date and time for log filenames
from datetime import datetime  # used to format current date/time for log filenames

# Import json for saving model metadata
import json  # used for saving model metadata to JSON files

# Import mlflow for experiment tracking
import mlflow  # used to log parameters, metrics, and artifacts
import mlflow.pytorch  # used to log PyTorch models as MLflow artifacts

# Import optuna for hyperparameter optimization
import optuna  # used for bayesian hyperparameter optimization
from optuna.integration import PyTorchLightningPruningCallback  # optional pruning integration


# -----------------------------
# Hardcoded configuration block
# -----------------------------

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # script directory

# Detect if running in Kaggle environment by checking for Kaggle-specific path
IS_KAGGLE = os.path.exists("/kaggle/input")  # check if Kaggle input directory exists

# Log the detected environment
if IS_KAGGLE:  # kaggle environment check
    logger_temp = logging.getLogger("env_check")  # temporary logger for environment detection
    logger_temp.info("Running in Kaggle environment")  # log kaggle detection
else:
    logger_temp = logging.getLogger("env_check")  # temporary logger for environment detection
    logger_temp.info("Running in local/server environment")  # log local detection

# Set DATA_DIR based on the detected environment
if IS_KAGGLE:  # kaggle condition
    DATA_DIR = "/kaggle/input/eu-driver-lincense/data"  # kaggle data path
else:  # local/server condition
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")  # local data directory relative to script

# Set OUTPUT_DIR based on the detected environment
if IS_KAGGLE:  # kaggle condition
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")  # save in script directory for kaggle
else:  # local/server condition
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models")  # save in models folder for local/server

# Define the directory where log files will be saved
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")  # logs directory relative to script

# Define the number of trials for Optuna optimization
N_TRIALS = 50  # number of hyperparameter configurations to test

# Define the number of epochs per trial (reduced for faster trials)
NUM_EPOCHS = 5  # number of epochs per trial (can be increased for final optimization)

# Define the base batch size (will be optimized)
BATCH_SIZE = 32  # default batch size

# Define the fraction of the dataset to use for validation
VAL_SPLIT = 0.2  # proportion of data reserved for validation

# Define a random seed for reproducibility
SEED = 42  # fixed seed for deterministic behavior


# -----------------------------
# MLflow configuration block
# -----------------------------

# Define the MLflow experiment name for this training script
MLFLOW_EXPERIMENT_NAME = "efficientnet_binary_optuna"  # MLflow experiment name

# Define the MLflow tracking URI (defaults to local file store under ./mlruns)
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"file:{os.path.join(SCRIPT_DIR, 'mlruns')}"
)  # MLflow tracking backend


# ---------------------------
# Logging setup for each run
# ---------------------------

# Create the log directory if it does not already exist
os.makedirs(LOG_DIR, exist_ok=True)  # ensure ./logs exists

# Use a fixed log filename for all runs
LOG_FILENAME = os.path.join(LOG_DIR, "train_log_optuna.txt")  # single log file for all runs

# Create a logger object for this module
logger = logging.getLogger(__name__)  # module-level logger

# Set the minimum logging level for this logger
logger.setLevel(logging.INFO)  # log INFO and above

# Prevent log messages from propagating to the root logger (avoid duplicates)
logger.propagate = False  # ensure we control handlers explicitly

# Define the log message format (timestamp, level, message)
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")  # formatter for log records

# Create a file handler that appends logs to the single log file
file_handler = logging.FileHandler(LOG_FILENAME, mode='a')  # handler that appends to text file

# Set the logging level for the file handler
file_handler.setLevel(logging.INFO)  # log INFO and above to file

# Attach the formatter to the file handler
file_handler.setFormatter(log_formatter)  # set formatter for file handler

# Add the file handler to the logger
logger.addHandler(file_handler)  # attach file handler

# Create a stream handler to also output logs to the console (stdout)
stream_handler = logging.StreamHandler()  # handler for console output

# Set the logging level for the stream handler
stream_handler.setLevel(logging.INFO)  # log INFO and above to console

# Attach the formatter to the stream handler
stream_handler.setFormatter(log_formatter)  # set formatter for console handler

# Add the stream handler to the logger
logger.addHandler(stream_handler)  # attach console handler

# Log the path of the log file being used for this run
logger.info(f"Logging to file: {LOG_FILENAME}")  # message indicating log file location

# Log a separator to distinguish between different runs
logger.info("=" * 80)  # visual separator between runs

# Log metadata for the current run
run_datetime = datetime.now()  # get current date and time
run_date_str = run_datetime.strftime("%Y-%m-%d")  # format date as YYYY-MM-DD
run_time_str = run_datetime.strftime("%H:%M:%S")  # format time as HH:MM:SS

# Create a metadata line with hashes and run information
metadata_line = f"{'#' * 10} OPTUNA RUN START - Date: {run_date_str}, Time: {run_time_str} {'#' * 10}"  # metadata with hashes

# Log the metadata line
logger.info(metadata_line)  # record run metadata

# Log a separator to distinguish between different runs
logger.info("=" * 80)  # visual separator after metadata


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    # Log that we are setting seeds
    logger.info(f"Setting random seed to {seed}")  # record chosen seed

    # Set the seed for Python's random module
    random.seed(seed)  # seed Python's RNG

    # Set the seed for NumPy's random generator
    np.random.seed(seed)  # seed NumPy RNG

    # Set the seed for PyTorch on CPU
    torch.manual_seed(seed)  # seed PyTorch CPU RNG

    # If CUDA is available, set the seed for all CUDA devices
    if torch.cuda.is_available():  # check for GPU availability
        torch.cuda.manual_seed_all(seed)  # seed all CUDA devices
        logger.info("CUDA is available; CUDA seeds set")  # log that CUDA seeds were set
    else:
        logger.info("CUDA is not available; only CPU seeds set")  # log that only CPU is used


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset class for loading images with labels."""
    def __init__(self, samples, transform=None):  # initialize with samples and transform
        self.samples = samples  # store sample list
        self.transform = transform  # store transform

    def __len__(self):  # return dataset length
        return len(self.samples)  # number of samples

    def __getitem__(self, idx):  # get single sample
        img_path, label = self.samples[idx]  # get sample path and label
        img = Image.open(img_path).convert('RGB')  # load image as RGB
        if self.transform:  # if transform provided
            img = self.transform(img)  # apply transform
        return img, label  # return image and label


def create_dataloaders(data_dir, batch_size, val_split, seed, augmentation_strength=1.0):
    """
    Create training and validation DataLoaders from specified directories.
    Loads positive class from './Original/' and negative class from './random_doc_images/'.
    Also computes class weights based on class frequencies.
    
    Args:
        augmentation_strength: multiplier for augmentation intensity (0.0 to 2.0)
    """
    # Log that we are starting DataLoader creation
    logger.info(f"Creating dataloaders from data directory: {data_dir}")  # record data directory path
    logger.info(f"Augmentation strength: {augmentation_strength:.2f}")  # record augmentation level

    # Define the image size expected by EfficientNet-B0 (224 x 224 pixels)
    image_size = 224  # input size for EfficientNet-B0

    # Define the mean values for each RGB channel as used for ImageNet
    imagenet_mean = [0.485, 0.456, 0.406]  # ImageNet mean for normalization

    # Define the standard deviation values for each RGB channel as used for ImageNet
    imagenet_std = [0.229, 0.224, 0.225]  # ImageNet std for normalization

    # Scale augmentation parameters based on augmentation_strength
    crop_scale_min = max(0.5, 0.8 - 0.3 * (augmentation_strength - 1.0))  # min crop scale
    crop_scale_max = 1.0  # max crop scale always 1.0
    
    # Define a set of transforms to apply to each image with parameterized augmentation
    transform = transforms.Compose([
        transforms.Resize(256),  # resize shorter side to 256 pixels
        transforms.RandomResizedCrop(image_size, scale=(crop_scale_min, crop_scale_max)),  # random crop with variable scale
        transforms.RandomHorizontalFlip(p=0.5 * augmentation_strength),  # variable flip probability
        transforms.ColorJitter(
            brightness=0.2 * augmentation_strength,
            contrast=0.2 * augmentation_strength,
            saturation=0.1 * augmentation_strength,
            hue=0.05 * augmentation_strength
        ),  # color jittering scaled by augmentation strength
        transforms.ToTensor(),  # convert PIL image to tensor
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),  # normalize tensor to ImageNet stats
    ])  # composed transforms for data augmentation and normalization

    # Define paths for positive and negative class directories
    positive_dir = os.path.join(data_dir, "Original")  # positive class: license images
    negative_dir = os.path.join(data_dir, "random_doc_images")  # negative class: other documents

    # Log the directories being used
    logger.info(f"Positive class directory: {positive_dir}")  # record positive dir
    logger.info(f"Negative class directory: {negative_dir}")  # record negative dir

    # Helper function to recursively find all image files in a directory tree
    def get_image_paths(root_dir):
        """Recursively collect all image file paths from directory and subdirectories."""
        image_paths = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
        return image_paths

    # Load all image paths for positive class (1) - directly from Original directory and subdirectories
    positive_paths = get_image_paths(positive_dir)  # get all images from positive directory
    positive_samples = [(img_path, 1) for img_path in positive_paths]  # label as class 1

    # Load all image paths for negative class (0) - from random_doc_images and its subdirectories
    negative_paths = get_image_paths(negative_dir)  # get all images from negative directory
    negative_samples = [(img_path, 0) for img_path in negative_paths]  # label as class 0

    # Combine all samples from both classes
    all_samples = positive_samples + negative_samples  # merged sample list

    # Get the total number of samples
    dataset_size = len(all_samples)  # total number of images

    # Log the total dataset size
    logger.info(f"Total number of samples in dataset: {dataset_size}")  # record dataset size

    # Log class distribution
    positive_count = len(positive_samples)  # count of positive samples
    negative_count = len(negative_samples)  # count of negative samples
    logger.info(f"Positive class samples: {positive_count}, Negative class samples: {negative_count}")  # record class distribution
    logger.info(f"Class to index mapping: {{'negative': 0, 'positive': 1}}")  # show mapping

    # Create the full dataset from combined samples
    full_dataset = CustomDataset(all_samples, transform=transform)  # custom dataset with all samples

    # Compute the number of validation samples using the val_split fraction
    val_size = int(dataset_size * val_split)  # number of samples for validation

    # Compute the number of training samples as the remainder
    train_size = dataset_size - val_size  # remaining samples for training

    # Log the train/val split sizes
    logger.info(f"Train size: {train_size}, Validation size: {val_size}")  # record split sizes

    # Extract labels from all samples for class weight computation
    targets = [label for _, label in all_samples]  # list of class labels

    # Convert the targets list to a torch tensor
    target_tensor = torch.tensor(targets, dtype=torch.long)  # tensor of labels

    # Compute the count of samples for each class index using bincount
    class_counts = torch.bincount(target_tensor)  # count examples per class index

    # Log the raw class counts
    logger.info(f"Raw class counts (by index): {class_counts.tolist()}")  # record counts for each class

    # Compute the total number of classes
    num_classes = len(class_counts)  # number of distinct class indices

    # Compute total number of samples as a float for weight calculation
    total_samples = float(dataset_size)  # float version of dataset size

    # Compute class weights inversely proportional to class frequencies
    # This downweights classes with many samples
    class_weights = total_samples / (num_classes * class_counts.float())  # inverse frequency weighting

    # Log the computed class weights
    logger.info(f"Computed class weights (by index): {class_weights.tolist()}")  # record weights used for loss

    # Create a generator with a fixed seed to make random splits reproducible
    generator = torch.Generator().manual_seed(seed)  # seeded generator for deterministic split

    # Split the dataset into train and validation subsets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)  # deterministic split

    # Log that random_split has been completed
    logger.info("Random split into train and validation subsets completed")  # confirm split

    # Create the DataLoader for the training subset
    train_loader = DataLoader(
        train_dataset,                               # dataset subset for training
        batch_size=batch_size,                       # number of samples per batch
        shuffle=True,                                # shuffle training data each epoch
        num_workers=4,                               # number of worker processes for data loading
        pin_memory=torch.cuda.is_available()         # enable pinned memory if GPU is used
    )  # DataLoader for training

    # Create the DataLoader for the validation subset
    val_loader = DataLoader(
        val_dataset,                                 # dataset subset for validation
        batch_size=batch_size,                       # number of samples per batch
        shuffle=False,                               # do not shuffle validation data
        num_workers=4,                               # number of worker processes for data loading
        pin_memory=torch.cuda.is_available()         # enable pinned memory if GPU is used
    )  # DataLoader for validation

    # Log that DataLoaders have been created successfully
    logger.info("Train and validation DataLoaders successfully created")  # confirm dataloader creation

    # Return the train DataLoader, validation DataLoader, number of classes, and class weights
    return train_loader, val_loader, num_classes, class_weights  # output for downstream use


def create_model(num_classes, dropout_rate=0.2):
    """
    Create an EfficientNet-B0 model and adapt it for classification.
    
    Args:
        dropout_rate: dropout probability for the classifier head
    """
    # Log that we are about to create the EfficientNet model
    logger.info(f"Creating EfficientNet-B0 model with dropout={dropout_rate:.3f}")  # indicate model creation start

    # Load a pretrained EfficientNet-B0 model from torchvision (pretrained on ImageNet)
    model = models.efficientnet_b0(pretrained=True)  # load pretrained EfficientNet-B0

    # Log that the pretrained model has been loaded
    logger.info("Pretrained EfficientNet-B0 model loaded")  # confirm model load

    # Get the number of input features to the final classifier layer
    in_features = model.classifier[1].in_features  # size of input to final linear layer

    # Log the number of input features to the classifier
    logger.info(f"EfficientNet-B0 classifier input features: {in_features}")  # log classifier input size

    # Replace the final classifier with a new sequential block including dropout
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),  # dropout before final layer
        nn.Linear(in_features, num_classes)  # final classification layer
    )  # new classification head with dropout

    # Log that the classifier head has been modified
    logger.info(f"EfficientNet-B0 classifier head replaced with Dropout({dropout_rate}) + Linear({in_features}, {num_classes})")  # confirm head replacement

    # Return the modified model
    return model  # return configured EfficientNet-B0


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Returns:
        tuple: (epoch_loss, epoch_acc, f1_score)
    """
    model.train()  # set model to training mode
    
    running_loss = 0.0  # accumulated loss
    running_corrects = 0  # accumulated correct predictions
    
    # Track confusion-matrix components
    tp = 0  # true positives
    tn = 0  # true negatives
    fp = 0  # false positives
    fn = 0  # false negatives
    
    # Iterate over all batches
    for inputs, labels in train_loader:
        # Move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # Update confusion matrix
        pred_pos = preds == 1
        label_pos = labels.data == 1
        tp += torch.sum(pred_pos & label_pos).item()
        tn += torch.sum((~pred_pos) & (~label_pos)).item()
        fp += torch.sum(pred_pos & (~label_pos)).item()
        fn += torch.sum((~pred_pos) & label_pos).item()
    
    # Compute metrics
    dataset_size = len(train_loader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    
    # Compute F1 score
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = ((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return epoch_loss, epoch_acc.item(), f1


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch.
    
    Returns:
        tuple: (epoch_loss, epoch_acc, f1_score)
    """
    model.eval()  # set model to evaluation mode
    
    running_loss = 0.0  # accumulated loss
    running_corrects = 0  # accumulated correct predictions
    
    # Track confusion-matrix components
    tp = 0  # true positives
    tn = 0  # true negatives
    fp = 0  # false positives
    fn = 0  # false negatives
    
    # Iterate over all batches
    with torch.no_grad():  # disable gradient computation
        for inputs, labels in val_loader:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update confusion matrix
            pred_pos = preds == 1
            label_pos = labels.data == 1
            tp += torch.sum(pred_pos & label_pos).item()
            tn += torch.sum((~pred_pos) & (~label_pos)).item()
            fp += torch.sum(pred_pos & (~label_pos)).item()
            fn += torch.sum((~pred_pos) & label_pos).item()
    
    # Compute metrics
    dataset_size = len(val_loader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    
    # Compute F1 score
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = ((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return epoch_loss, epoch_acc.item(), f1


def objective(trial, device):
    """
    Optuna objective function to optimize hyperparameters.
    
    This function will be called by Optuna for each trial with different hyperparameter values.
    
    Args:
        trial: Optuna trial object
        device: PyTorch device (CPU or CUDA)
        
    Returns:
        float: validation F1 score (to maximize)
    """
    # Log trial start
    logger.info(f"Starting Trial {trial.number}")
    
    # Suggest hyperparameters using appropriate scales
    
    # Learning rate: log scale from 1e-5 to 1e-2
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    
    # Batch size: categorical choice (powers of 2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    
    # Dropout rate: linear scale from 0.0 to 0.5
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    
    # Weight decay (L2 regularization): log scale from 1e-6 to 1e-3
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Optimizer type: categorical choice
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    
    # Augmentation strength: linear scale from 0.5 to 1.5
    augmentation_strength = trial.suggest_float("augmentation_strength", 0.5, 1.5)
    
    # Learning rate scheduler: categorical choice
    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
    
    # If using scheduler, suggest scheduler parameters
    if use_scheduler:
        scheduler_type = trial.suggest_categorical("scheduler_type", ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
        
        if scheduler_type == "StepLR":
            step_size = trial.suggest_int("step_size", 2, 5)
            gamma = trial.suggest_float("gamma", 0.1, 0.5)
        elif scheduler_type == "CosineAnnealingLR":
            t_max = trial.suggest_int("t_max", 3, NUM_EPOCHS)
    
    # Log suggested hyperparameters
    logger.info(f"Trial {trial.number} hyperparameters:")
    logger.info(f"  learning_rate: {learning_rate:.6f}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  dropout_rate: {dropout_rate:.3f}")
    logger.info(f"  weight_decay: {weight_decay:.6f}")
    logger.info(f"  optimizer: {optimizer_name}")
    logger.info(f"  augmentation_strength: {augmentation_strength:.3f}")
    logger.info(f"  use_scheduler: {use_scheduler}")
    
    # Create dataloaders with trial-specific batch size and augmentation
    train_loader, val_loader, num_classes, class_weights = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=batch_size,
        val_split=VAL_SPLIT,
        seed=SEED,
        augmentation_strength=augmentation_strength
    )
    
    # Create model with trial-specific dropout
    model = create_model(num_classes=num_classes, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # Create loss function with class weights
    class_weights_device = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_device)
    
    # Create optimizer based on trial suggestion
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        # For SGD, also optimize momentum
        momentum = trial.suggest_float("momentum", 0.5, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        logger.info(f"  momentum: {momentum:.3f}")
    
    # Create learning rate scheduler if suggested
    scheduler = None
    if use_scheduler:
        if scheduler_type == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            logger.info(f"  scheduler: StepLR(step_size={step_size}, gamma={gamma:.2f})")
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
            logger.info(f"  scheduler: CosineAnnealingLR(T_max={t_max})")
        elif scheduler_type == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
            logger.info(f"  scheduler: ReduceLROnPlateau")
    
    # Track best validation F1 for this trial
    best_val_f1 = 0.0
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Train for one epoch
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)  # step based on validation F1
            else:
                scheduler.step()  # step based on epoch
        
        # Track best F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        
        # Log epoch results
        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{NUM_EPOCHS}: "
                   f"train_loss={train_loss:.4f}, train_f1={train_f1:.4f}, "
                   f"val_loss={val_loss:.4f}, val_f1={val_f1:.4f}")
        
        # Report intermediate value for pruning
        trial.report(val_f1, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()
    
    # Log trial completion
    logger.info(f"Trial {trial.number} completed with best val F1: {best_val_f1:.4f}")
    
    # Return best validation F1 score
    return best_val_f1


def main():
    """
    Main function to orchestrate the Optuna hyperparameter optimization.
    """
    # Log the start of the main function execution
    logger.info("Main function started - Optuna hyperparameter optimization")
    
    # Set random seeds for reproducibility
    set_seed(SEED)
    
    # Determine whether to use GPU (CUDA) or CPU for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Log which device is being used
    logger.info(f"Using device: {device}")
    
    # Configure MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Create Optuna study to maximize validation F1 score
    # Use InMemoryStorage to avoid SQLAlchemy compatibility issues in Kaggle
    study = optuna.create_study(
        direction="maximize",  # maximize validation F1
        study_name=f"efficientnet_optuna_{run_date_str}_{run_time_str.replace(':', '-')}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),  # prune unpromising trials
        storage=None  # use in-memory storage (no SQLAlchemy dependency)
    )
    
    # Log study creation
    logger.info(f"Created Optuna study: {study.study_name}")
    logger.info(f"Optimization target: maximize validation F1 score")
    logger.info(f"Number of trials: {N_TRIALS}")
    logger.info(f"Epochs per trial: {NUM_EPOCHS}")
    
    # Start MLflow parent run to track the entire optimization
    parent_run_name = f"optuna_study_{run_date_str}_{run_time_str.replace(':', '-')}"
    with mlflow.start_run(run_name=parent_run_name):
        # Log study parameters
        mlflow.log_params({
            "n_trials": N_TRIALS,
            "epochs_per_trial": NUM_EPOCHS,
            "val_split": VAL_SPLIT,
            "seed": SEED,
            "device": str(device),
            "data_dir": DATA_DIR,
            "output_dir": OUTPUT_DIR
        })
        
        # Run optimization
        study.optimize(lambda trial: objective(trial, device), n_trials=N_TRIALS)
        
        # Log best trial information
        best_trial = study.best_trial
        logger.info("=" * 80)
        logger.info("Optimization completed!")
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best validation F1: {best_trial.value:.4f}")
        logger.info("Best hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
            mlflow.log_param(f"best_{key}", value)
        
        # Log best F1 score
        mlflow.log_metric("best_val_f1", best_trial.value)
        
        # Save study results
        study_results_path = os.path.join(OUTPUT_DIR, f"optuna_study_{run_date_str}_{run_time_str.replace(':', '-')}.json")
        
        # Create study results dictionary
        study_results = {
            "study_name": study.study_name,
            "n_trials": N_TRIALS,
            "best_trial": {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params
            },
            "all_trials": [
                {
                    "number": trial.number,
                    "value": trial.value if trial.value is not None else "pruned",
                    "params": trial.params,
                    "state": trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        # Save to JSON
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(study_results_path, 'w') as f:
            json.dump(study_results, f, indent=2)
        
        logger.info(f"Study results saved to: {study_results_path}")
        
        # Log study results to MLflow
        mlflow.log_artifact(study_results_path, artifact_path="optuna")
        mlflow.log_artifact(LOG_FILENAME, artifact_path="logs")
        
        # Generate and save optimization history plot
        try:
            import optuna.visualization as vis
            import plotly
            
            # Create optimization history plot
            fig = vis.plot_optimization_history(study)
            history_plot_path = os.path.join(OUTPUT_DIR, f"optuna_history_{run_date_str}_{run_time_str.replace(':', '-')}.html")
            plotly.offline.plot(fig, filename=history_plot_path, auto_open=False)
            mlflow.log_artifact(history_plot_path, artifact_path="optuna")
            logger.info(f"Optimization history plot saved to: {history_plot_path}")
            
            # Create parameter importance plot
            fig = vis.plot_param_importances(study)
            importance_plot_path = os.path.join(OUTPUT_DIR, f"optuna_importance_{run_date_str}_{run_time_str.replace(':', '-')}.html")
            plotly.offline.plot(fig, filename=importance_plot_path, auto_open=False)
            mlflow.log_artifact(importance_plot_path, artifact_path="optuna")
            logger.info(f"Parameter importance plot saved to: {importance_plot_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate Optuna visualization plots: {e}")
    
    # Log that the main function execution has finished
    logger.info("=" * 80)
    logger.info("Main function finished")


# Ensure that main() runs only when this script is executed directly
if __name__ == "__main__":
    main()
