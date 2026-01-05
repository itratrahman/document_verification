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

# Import time to measure training duration and create Unix timestamps
import time  # used to compute elapsed time and epoch time

# Import copy to clone model weights when tracking the best model
import copy  # used for deep copying model state_dict

# Import datetime to get human-readable date and time for log filenames
from datetime import datetime  # used to format current date/time for log filenames


# -----------------------------
# Hardcoded configuration block
# -----------------------------

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # script directory

# Define the root directory containing 'positive' and 'negative' subfolders
DATA_DIR = os.path.join(SCRIPT_DIR, "data")  # data directory relative to script

# Define the directory where model checkpoints will be saved
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")  # checkpoints directory relative to script

# Define the directory where log files will be saved
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")  # logs directory relative to script

# Define the number of epochs for training
NUM_EPOCHS = 10  # number of passes through the full training set

# Define the batch size used in training and validation
BATCH_SIZE = 32  # number of samples per batch

# Define the initial learning rate for the optimizer
LEARNING_RATE = 1e-4  # step size for Adam optimizer

# Define the fraction of the dataset to use for validation
VAL_SPLIT = 0.2  # proportion of data reserved for validation

# Define a random seed for reproducibility
SEED = 42  # fixed seed for deterministic behavior


# ---------------------------
# Logging setup for each run
# ---------------------------

# Create the log directory if it does not already exist
os.makedirs(LOG_DIR, exist_ok=True)  # ensure ./logs exists

# Get the current local date and time
run_dt = datetime.now()  # capture current datetime object

# Format the date and time as YYYYMMDD_HHMMSS for readability
date_str = run_dt.strftime("%Y%m%d_%H%M%S")  # formatted timestamp string

# Get the current Unix epoch time in seconds (Linux time)
epoch_time = int(time.time())  # integer Unix timestamp

# Build a unique log filename that includes date/time and Unix epoch
LOG_FILENAME = os.path.join(LOG_DIR, f"train_log_{date_str}_{epoch_time}.txt")  # full log file path

# Create a logger object for this module
logger = logging.getLogger(__name__)  # module-level logger

# Set the minimum logging level for this logger
logger.setLevel(logging.INFO)  # log INFO and above

# Prevent log messages from propagating to the root logger (avoid duplicates)
logger.propagate = False  # ensure we control handlers explicitly

# Define the log message format (timestamp, level, message)
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")  # formatter for log records

# Create a file handler that writes logs to the per-run log file
file_handler = logging.FileHandler(LOG_FILENAME)  # handler that writes to text file

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


def create_dataloaders(data_dir, batch_size, val_split, seed):
    """
    Create training and validation DataLoaders from an ImageFolder dataset.
    Assumes data_dir has two subfolders: 'positive' and 'negative'.
    Also computes class weights based on class frequencies.
    """
    # Log that we are starting DataLoader creation
    logger.info(f"Creating dataloaders from data directory: {data_dir}")  # record data directory path

    # Define the image size expected by EfficientNet-B0 (224 x 224 pixels)
    image_size = 224  # input size for EfficientNet-B0

    # Define the mean values for each RGB channel as used for ImageNet
    imagenet_mean = [0.485, 0.456, 0.406]  # ImageNet mean for normalization

    # Define the standard deviation values for each RGB channel as used for ImageNet
    imagenet_std = [0.229, 0.224, 0.225]  # ImageNet std for normalization

    # Define a set of transforms to apply to each image
    transform = transforms.Compose([
        transforms.Resize(256),  # resize shorter side to 256 pixels
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # random crop to 224x224 with slight scale jitter
        transforms.RandomHorizontalFlip(),  # random horizontal flip with p=0.5
        transforms.ToTensor(),  # convert PIL image to tensor
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),  # normalize tensor to ImageNet stats
    ])  # composed transforms for data augmentation and normalization

    # Create an ImageFolder dataset; subdirectories map to class labels
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)  # dataset with labels inferred from folder names

    # Log the mapping from class names to indices
    logger.info(f"Class to index mapping: {full_dataset.class_to_idx}")  # show mapping like {'negative': 0, 'positive': 1}

    # Get the total number of samples in the dataset
    dataset_size = len(full_dataset)  # total number of images

    # Log the total dataset size
    logger.info(f"Total number of samples in dataset: {dataset_size}")  # record dataset size

    # Compute the number of validation samples using the val_split fraction
    val_size = int(dataset_size * val_split)  # number of samples for validation

    # Compute the number of training samples as the remainder
    train_size = dataset_size - val_size  # remaining samples for training

    # Log the train/val split sizes
    logger.info(f"Train size: {train_size}, Validation size: {val_size}")  # record split sizes

    # Extract the list of targets (class indices) for all samples
    targets = full_dataset.targets  # list of class indices for each image

    # Convert the targets list to a torch tensor
    target_tensor = torch.tensor(targets, dtype=torch.long)  # tensor of labels

    # Compute the count of samples for each class index using bincount
    class_counts = torch.bincount(target_tensor)  # count examples per class index

    # Log the raw class counts
    logger.info(f"Raw class counts (by index): {class_counts.tolist()}")  # record counts for each class

    # Compute the total number of classes from the length of class_counts
    num_classes = len(class_counts)  # number of distinct class indices

    # Compute total number of samples as a float for weight calculation
    total_samples = float(dataset_size)  # float version of dataset size

    # Compute class weights inversely proportional to class frequencies
    # This downweights classes with many samples (e.g., majority positive class)
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
    return train_loader, val_loader, len(full_dataset.classes), class_weights  # output for downstream use


def create_model(num_classes):
    """
    Create an EfficientNet-B0 model and adapt it for classification.
    """
    # Log that we are about to create the EfficientNet model
    logger.info("Creating EfficientNet-B0 model")  # indicate model creation start

    # Load a pretrained EfficientNet-B0 model from torchvision (pretrained on ImageNet)
    model = models.efficientnet_b0(pretrained=True)  # load pretrained EfficientNet-B0

    # Log that the pretrained model has been loaded
    logger.info("Pretrained EfficientNet-B0 model loaded")  # confirm model load

    # Get the number of input features to the final classifier layer
    in_features = model.classifier[1].in_features  # size of input to final linear layer

    # Log the number of input features to the classifier
    logger.info(f"EfficientNet-B0 classifier input features: {in_features}")  # log classifier input size

    # Replace the final fully-connected layer with a new one that outputs num_classes logits
    model.classifier[1] = nn.Linear(in_features, num_classes)  # new classification head

    # Log that the classifier head has been modified
    logger.info(f"EfficientNet-B0 classifier head replaced with Linear({in_features}, {num_classes})")  # confirm head replacement

    # Return the modified model
    return model  # return configured EfficientNet-B0


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs, output_dir):
    """
    Train the model and keep the best weights based on validation accuracy.
    """
    # Log the start of the training process
    logger.info("Starting training loop")  # indicate training start

    # Record the start time for the entire training
    since = time.time()  # timestamp at training start

    # Initialize the best validation accuracy observed so far
    best_acc = 0.0  # best validation accuracy found so far

    # Make a deep copy of the initial model weights as a baseline
    best_model_wts = copy.deepcopy(model.state_dict())  # store best weights

    # Ensure that the output directory exists; create it if it does not
    os.makedirs(output_dir, exist_ok=True)  # ensure checkpoint directory exists

    # Log the directory where checkpoints will be saved
    logger.info(f"Model checkpoints will be saved to: {output_dir}")  # record checkpoint dir

    # Loop over the specified number of epochs
    for epoch in range(num_epochs):  # iterate over epochs
        # Log the start of a new epoch
        logger.info(f"Epoch {epoch + 1}/{num_epochs} started")  # record epoch start

        # Loop over two phases: 'train' and 'val'
        for phase in ["train", "val"]:  # iterate over training and validation phases
            # Log which phase we are in
            logger.info(f"Phase: {phase}")  # log current phase

            # Set the model to training mode if phase is 'train'
            if phase == "train":  # training phase condition
                model.train()  # enable training mode
            # Otherwise set the model to evaluation mode (no dropout, no batchnorm updates)
            else:  # validation phase condition
                model.eval()  # enable evaluation mode

            # Initialize variables to track running loss and correct predictions
            running_loss = 0.0  # accumulated loss for this phase
            running_corrects = 0  # accumulated correct predictions

            # Select the appropriate DataLoader for the current phase
            data_loader = dataloaders[phase]  # choose train or val dataloader

            # Iterate over all batches in the current DataLoader
            for batch_idx, (inputs, labels) in enumerate(data_loader):  # loop over batches
                # Move the inputs to the specified device (CPU or GPU)
                inputs = inputs.to(device)  # put images on device

                # Move the labels to the specified device
                labels = labels.to(device)  # put labels on device

                # Zero (reset) the gradients for the optimizer
                optimizer.zero_grad()  # clear previous gradients

                # Enable gradient computation only in the training phase
                with torch.set_grad_enabled(phase == "train"):  # enable/disable grad
                    # Perform a forward pass through the model to obtain outputs
                    outputs = model(inputs)  # model predictions (logits)

                    # Select the class with the highest logit for each sample as the prediction
                    _, preds = torch.max(outputs, 1)  # predicted class indices

                    # Compute the loss between model outputs and ground truth labels
                    loss = criterion(outputs, labels)  # compute weighted cross-entropy loss

                    # If we are in the training phase, perform backpropagation and optimizer step
                    if phase == "train":  # only update weights during training
                        loss.backward()  # backpropagate loss
                        optimizer.step()  # perform optimizer update

                # Multiply batch loss by batch size and accumulate into running_loss
                running_loss += loss.item() * inputs.size(0)  # aggregate loss

                # Count how many predictions match the ground truth labels and accumulate
                running_corrects += torch.sum(preds == labels.data)  # aggregate corrects

                # Optionally log progress every N batches (here N=50) to avoid too much logging
                if (batch_idx + 1) % 50 == 0:  # log every 50 batches
                    logger.info(
                        f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Phase [{phase}], "
                        f"Batch [{batch_idx + 1}/{len(data_loader)}]"  # log mini-progress
                    )

            # Compute the total number of samples in this phase
            dataset_size = len(data_loader.dataset)  # number of samples in this phase

            # Compute the average loss for this phase
            epoch_loss = running_loss / dataset_size  # average loss over phase

            # Compute the accuracy as number of correct predictions divided by total samples
            epoch_acc = running_corrects.double() / dataset_size  # accuracy over phase

            # Log the loss and accuracy for this epoch and phase
            logger.info(f"{phase} - Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")  # record metrics

            # If this is the validation phase, check if accuracy is better than previous best
            if phase == "val":  # validation phase check
                # If the current validation accuracy is higher than best_acc, update best
                if epoch_acc > best_acc:  # improved accuracy condition
                    # Update best accuracy value
                    best_acc = epoch_acc  # store new best accuracy

                    # Deep copy the model weights as the new best weights
                    best_model_wts = copy.deepcopy(model.state_dict())  # store best weights

                    # Define the path to save the best model checkpoint
                    best_model_path = os.path.join(output_dir, "best_efficientnet_binary.pt")  # path to best model file

                    # Save the best model weights to disk
                    torch.save(best_model_wts, best_model_path)  # write best weights to file

                    # Log that we achieved a new best validation accuracy
                    logger.info(f"New best model saved with val acc: {best_acc:.4f} at {best_model_path}")  # record new best

        # Log that the epoch has completed
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed")  # mark epoch end

    # Compute total training time in seconds
    time_elapsed = time.time() - since  # time difference between end and start

    # Log the total training duration in minutes and seconds
    logger.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")  # elapsed time

    # Log the best validation accuracy achieved
    logger.info(f"Best validation accuracy: {best_acc:.4f}")  # best val accuracy

    # Load the best weights into the model before returning
    model.load_state_dict(best_model_wts)  # restore best model weights

    # Return the model with the best validation performance
    return model  # trained model ready for use or saving


def main():
    """
    Main function to orchestrate the training process.
    """
    # Log the start of the main function execution
    logger.info("Main function started")  # record main start

    # Set random seeds for reproducibility
    set_seed(SEED)  # seed all RNGs

    # Determine whether to use GPU (CUDA) or CPU for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # choose device

    # Log which device is being used
    logger.info(f"Using device: {device}")  # record device choice

    # Create the training and validation DataLoaders and get class information
    train_loader, val_loader, num_classes, class_weights = create_dataloaders(
        data_dir=DATA_DIR,      # root data directory
        batch_size=BATCH_SIZE,  # batch size for loaders
        val_split=VAL_SPLIT,    # validation split fraction
        seed=SEED               # random seed for deterministic split
    )  # dataloaders and class weights

    # Create a dictionary of DataLoaders for easier handling in the training loop
    dataloaders = {
        "train": train_loader,  # training DataLoader
        "val": val_loader       # validation DataLoader
    }  # dict of dataloaders

    # Create the EfficientNet-B0 model adjusted for the number of classes
    model = create_model(num_classes=num_classes)  # build model

    # Move the model to the chosen device (CPU or GPU)
    model = model.to(device)  # put model on device

    # Move the class_weights tensor to the same device as the model
    class_weights_device = class_weights.to(device)  # put weights on device

    # Log the final class weights being passed to the loss function
    logger.info(f"Final class weights on device: {class_weights_device.tolist()}")  # record weights device-side

    # Define the loss function as CrossEntropyLoss with the computed class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_device)  # weighted cross-entropy

    # Log that the loss function has been created
    logger.info("CrossEntropyLoss with class weights created")  # confirm loss creation

    # Define the optimizer; here we use Adam with the specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer

    # Log the optimizer configuration
    logger.info(f"Adam optimizer created with learning rate: {LEARNING_RATE}")  # record optimizer settings

    # Train the model using the training loop and get the best model back
    trained_model = train_model(
        model=model,             # model to train
        dataloaders=dataloaders, # train and val dataloaders
        criterion=criterion,     # loss function
        optimizer=optimizer,     # optimizer
        device=device,           # device to use
        num_epochs=NUM_EPOCHS,   # number of epochs
        output_dir=OUTPUT_DIR    # directory to save checkpoints
    )  # trained model with best weights

    # Define the path where the final trained model will be saved
    final_model_path = os.path.join(OUTPUT_DIR, "final_efficientnet_binary.pt")  # final model path

    # Save the final trained model weights to disk
    torch.save(trained_model.state_dict(), final_model_path)  # write final model weights

    # Log that the final model has been saved
    logger.info(f"Final model saved to: {final_model_path}")  # record final model file path

    # Log that the main function execution has finished
    logger.info("Main function finished")  # main end marker


# Ensure that main() runs only when this script is executed directly
if __name__ == "__main__":  # script entry point check
    # Call the main function to start the script
    main()  # launch training process
