import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import os
import pandas as pd
from PIL import Image
import requests
import zipfile
import shutil

# Enforce determinism for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon GPU).")
else:
    device = torch.device("cpu")
    print("Using CPU device.")
print(f"Using device: {device}")

# Set random seed for reproducibility
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():  # Ensure CUDA seeds are set only if GPU is used
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups

# Primary Palette
HTB_GREEN = "#9fef00"
NODE_BLACK = "#141d2b"
HACKER_GREY = "#a4b1cd"
WHITE = "#ffffff"
# Secondary Palette
AZURE = "#0086ff"
NUGGET_YELLOW = "#ffaf00"
MALWARE_RED = "#ff3e3e"
VIVID_PURPLE = "#9f00ff"
AQUAMARINE = "#2ee7b6"
# Matplotlib Style Settings
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update(
    {
        "figure.facecolor": NODE_BLACK,
        "figure.edgecolor": NODE_BLACK,
        "axes.facecolor": NODE_BLACK,
        "axes.edgecolor": HACKER_GREY,
        "axes.labelcolor": HACKER_GREY,
        "axes.titlecolor": WHITE,
        "xtick.color": HACKER_GREY,
        "ytick.color": HACKER_GREY,
        "grid.color": HACKER_GREY,
        "grid.alpha": 0.1,
        "legend.facecolor": NODE_BLACK,
        "legend.edgecolor": HACKER_GREY,
        "legend.labelcolor": HACKER_GREY,
        "text.color": HACKER_GREY,
    }
)

print("Setup complete.")

GTSRB_CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for veh over 3.5 tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed/pass limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5 tons",
}
NUM_CLASSES_GTSRB = len(GTSRB_CLASS_NAMES)  # Should be 43


def get_gtsrb_class_name(class_id):
    """
    Retrieves the human-readable name for a given GTSRB class ID.

    Args:
        class_id (int): The numeric class ID (0-42).

    Returns:
        str: The corresponding class name or an 'Unknown Class' string.
    """
    return GTSRB_CLASS_NAMES.get(class_id, f"Unknown Class {class_id}")
# Dataset Root Directory
DATASET_ROOT = "./GTSRB"

# URLs for the GTSRB dataset components
DATASET_URL = "https://academy.hackthebox.com/storage/resources/GTSRB.zip"
DOWNLOAD_DIR = "./gtsrb_downloads"  # Temporary download location


def download_file(url, dest_folder, filename):
    """
    Downloads a file from a URL to a specified destination.

    Args:
        url (str): The URL of the file to download.
        dest_folder (str): The directory to save the downloaded file.
        filename (str): The name to save the file as.

    Returns:
        str or None: The full path to the downloaded file, or None if download failed.
    """
    filepath = os.path.join(dest_folder, filename)
    if os.path.exists(filepath):
        print(f"File '{filename}' already exists in {dest_folder}. Skipping download.")
        return filepath
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        os.makedirs(dest_folder, exist_ok=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}.")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None


def extract_zip(zip_filepath, extract_to):
    """
    Extracts the contents of a zip file to a specified directory.

    Args:
        zip_filepath (str): The path to the zip file.
        extract_to (str): The directory where contents should be extracted.

    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    print(f"Extracting '{os.path.basename(zip_filepath)}' to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Successfully extracted '{os.path.basename(zip_filepath)}'.")
        return True
    except zipfile.BadZipFile:
        print(
            f"Error: Failed to extract '{os.path.basename(zip_filepath)}'. File might be corrupted or not a zip file."
        )
        return False
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return False
    
# Define expected paths within DATASET_ROOT
train_dir = os.path.join(DATASET_ROOT, "Final_Training", "Images")
test_img_dir = os.path.join(DATASET_ROOT, "Final_Test", "Images")
test_csv_path = os.path.join(DATASET_ROOT, "GT-final_test.csv")

# Check if the core dataset components exist
dataset_ready = (
    os.path.isdir(DATASET_ROOT)
    and os.path.isdir(train_dir)
    and os.path.isdir(test_img_dir) # Check if test dir exists
    and os.path.isfile(test_csv_path) # Check if test csv exists
)

if dataset_ready:
    print(
        f"GTSRB dataset found and seems complete in '{DATASET_ROOT}'. Skipping download."
    )
else:
    print(
        f"GTSRB dataset not found or incomplete in '{DATASET_ROOT}'. Attempting download and extraction..."
    )
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Download files
    dataset_zip_path = download_file(
        DATASET_URL, DOWNLOAD_DIR, "GTSRB.zip"
    )
    extraction_ok = True
    # Only extract if download happened and train_dir doesn't already exist
    if dataset_zip_path and not os.path.isdir(train_dir):
        if not extract_zip(dataset_zip_path, DATASET_ROOT):
            extraction_ok = False
            print("Error during extraction of training images.")
    elif not dataset_zip_path and not os.path.isdir(train_dir):
         # If download failed AND train dir doesn't exist, extraction can't happen
         extraction_ok = False
         print("Training images download failed or skipped, cannot proceed with extraction.")

    if not os.path.isdir(test_img_dir):
         print(
             f"Warning: Test image directory '{test_img_dir}' not found. Ensure it's placed correctly."
         )
    if not os.path.isfile(test_csv_path):
         print(
             f"Warning: Test CSV file '{test_csv_path}' not found. Ensure it's placed correctly."
         )

    # Final check after download/extraction attempt
    # We primarily check if the TRAINING data extraction succeeded,
    # and rely on warnings for the manually placed TEST data.
    dataset_ready = (
        os.path.isdir(DATASET_ROOT)
        and os.path.isdir(train_dir)
        and extraction_ok
    )

    if dataset_ready and os.path.isdir(test_img_dir) and os.path.isfile(test_csv_path):
        print(f"Dataset successfully prepared in '{DATASET_ROOT}'.")
        # Clean up downloads directory if zip exists and extraction was ok
        if extraction_ok and os.path.exists(DOWNLOAD_DIR):
            try:
                shutil.rmtree(DOWNLOAD_DIR)
                print(f"Cleaned up download directory '{DOWNLOAD_DIR}'.")
            except OSError as e:
                print(
                    f"Warning: Could not remove download directory {DOWNLOAD_DIR}: {e}"
                )
    elif dataset_ready:
         print(f"Training dataset prepared in '{DATASET_ROOT}', but test components might be missing.")
         if not os.path.isdir(test_img_dir): print(f" - Missing: {test_img_dir}")
         if not os.path.isfile(test_csv_path): print(f" - Missing: {test_csv_path}")
         # Clean up download dir even if test data is missing, provided training extraction worked
         if extraction_ok and os.path.exists(DOWNLOAD_DIR):
             try:
                 shutil.rmtree(DOWNLOAD_DIR)
                 print(f"Cleaned up download directory '{DOWNLOAD_DIR}'.")
             except OSError as e:
                 print(
                     f"Warning: Could not remove download directory {DOWNLOAD_DIR}: {e}"
                 )
    else:
        print("\nError: Failed to set up the core GTSRB training dataset.")
        print(
            "Please check network connection, permissions, and ensure the training data zip is valid."
        )
        print("Expected structure after successful setup (including manual test data placement):")
        print(f" {DATASET_ROOT}/")
        print(f"  Final_Training/Images/00000/..ppm files..")
        print(f"  ...")
        print(f"  Final_Test/Images/..ppm files..")
        print(f"  GT-final_test.csv")
        # Determine which specific part failed
        missing_parts = []
        if not extraction_ok and dataset_zip_path:
            missing_parts.append("Training data extraction")
        if not dataset_zip_path and not os.path.isdir(train_dir):
            missing_parts.append("Training data download")
        if not os.path.isdir(train_dir):
             missing_parts.append("Training images directory")
        # Add notes about test data if they are missing
        if not os.path.isdir(test_img_dir):
             missing_parts.append("Test images (manual placement likely needed)")
        if not os.path.isfile(test_csv_path):
             missing_parts.append("Test CSV (manual placement likely needed)")


        raise FileNotFoundError(
             f"GTSRB dataset setup failed. Critical failure in obtaining training data. Missing/Problem parts: {', '.join(missing_parts)} in {DATASET_ROOT}"
         )
# Define image size and normalization constants
IMG_SIZE = 48  # Resize GTSRB images to 48x48
# Using ImageNet stats is common practice if dataset-specific stats aren't available/standard
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# Our specific attack parameters
SOURCE_CLASS = 14  # Stop Sign index
TARGET_CLASS = 3  # Speed limit 60km/h index
POISON_RATE = 0.10  # Poison a % of the Stop Signs in the training data

# Trigger Definition (relative to 48x48 image size)
TRIGGER_SIZE = 4  # 4x4 block
TRIGGER_POS = (
    IMG_SIZE - TRIGGER_SIZE - 1,
    IMG_SIZE - TRIGGER_SIZE - 1,
)  # Bottom-right corner
# Trigger Color: Magenta (R=1, G=0, B=1) in [0, 1] range
TRIGGER_COLOR_VAL = (1.0, 0.0, 1.0)

print(f"\nDataset configuration:")
print(f" Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f" Number of Classes: {NUM_CLASSES_GTSRB}")
print(f" Source Class: {SOURCE_CLASS} ({get_gtsrb_class_name(SOURCE_CLASS)})")
print(f" Target Class: {TARGET_CLASS} ({get_gtsrb_class_name(TARGET_CLASS)})")
print(f" Poison Rate: {POISON_RATE * 100}%")
print(f" Trigger: {TRIGGER_SIZE}x{TRIGGER_SIZE} magenta square at {TRIGGER_POS}")

class GTSRB_CNN(nn.Module):
    """
    A CNN adapted for the GTSRB dataset (43 classes, 48x48 input).
    Implements standard CNN components with adjusted layer dimensions for GTSRB.
    """

    def __init__(self, num_classes=NUM_CLASSES_GTSRB):
        """
        Initializes the CNN layers for GTSRB.

        Args:
            num_classes (int): Number of output classes (default: NUM_CLASSES_GTSRB).
        """
        super(GTSRB_CNN, self).__init__()
        # Conv Layer 1: Input 3 channels (RGB), Output 32 filters, Kernel 3x3, Padding 1
        # Processes 48x48 input
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Output shape: (Batch Size, 32, 48, 48)

        # Conv Layer 2: Input 32 channels, Output 64 filters, Kernel 3x3, Padding 1
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        # Output shape: (Batch Size, 64, 48, 48)

        # Max Pooling 1: Kernel 2x2, Stride 2. Reduces spatial dimensions by half.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape: (Batch Size, 64, 24, 24)

        # Conv Layer 3: Input 64 channels, Output 128 filters, Kernel 3x3, Padding 1
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        # Output shape: (Batch Size, 128, 24, 24)

        # Max Pooling 2: Kernel 2x2, Stride 2. Reduces spatial dimensions by half again.
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape: (Batch Size, 128, 12, 12)

        # Calculate flattened feature size after pooling layers
        # This is needed for the input size of the first fully connected layer
        self._feature_size = 128 * 12 * 12  # 18432

        # Fully Connected Layer 1 (Hidden): Maps flattened features to 512 hidden units.
        # Input size MUST match self._feature_size
        self.fc1 = nn.Linear(self._feature_size, 512)
        # Implements Y1 = f(W1 * X_flat + b1), where f is ReLU

        # Fully Connected Layer 2 (Output): Maps hidden units to class logits.
        # Output size MUST match num_classes
        self.fc2 = nn.Linear(512, num_classes)
        # Implements Y_logits = W2 * Y1 + b2

        # Dropout layer for regularization (p=0.5 means 50% probability of dropping a unit)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
	    # Apply first Conv block: Conv1 -> ReLU -> Conv2 -> ReLU -> Pool1
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        # Apply second Conv block: Conv3 -> ReLU -> Pool2
        x = self.pool2(F.relu(self.conv3(x)))

        # Flatten the feature map output from the convolutional blocks
        x = x.view(-1, self._feature_size)  # Reshape to (Batch Size, _feature_size)

        # Apply Dropout before the first FC layer (common practice)
        x = self.dropout(x)
        # Apply first FC layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply Dropout again before the output layer
        x = self.dropout(x)
        # Apply the final FC layer to get logits
        x = self.fc2(x)      
        return x
# Instantiate the GTSRB model structure and move it to the configured device
model_structure_gtsrb = GTSRB_CNN(num_classes=NUM_CLASSES_GTSRB).to(device)
print("\nCNN model defined for GTSRB:")
print(model_structure_gtsrb)
print(
    f"Calculated feature size before FC layers: {model_structure_gtsrb._feature_size}"
)
# Base transform (Resize + ToTensor) - Applied first to all images
transform_base = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to standard size
        transforms.ToTensor(),  # Converts PIL Image [0, 255] to Tensor [0, 1]
    ]
)
# Post-trigger transform for training data (augmentation + normalization) - Applied last in training
transform_train_post = transforms.Compose(
    [
        transforms.RandomRotation(10),  # Augmentation: Apply small random rotation
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2
        ),  # Augmentation: Adjust color slightly
        transforms.Normalize(IMG_MEAN, IMG_STD),  # Normalize using ImageNet stats
    ]
)

# Transform for clean test data (Resize, ToTensor, Normalize) - Used for evaluation
transform_test = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(IMG_MEAN, IMG_STD),  # Normalize
    ]
)
# Inverse transform for visualization (reverses normalization)
inverse_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(IMG_MEAN, IMG_STD)], std=[1 / s for s in IMG_STD]
)
try:
    # Load reference training set using ImageFolder to get class-to-index mapping
    # This instance won't be used for training directly, only for metadata.
    trainset_clean_ref = ImageFolder(root=train_dir)
    gtsrb_class_to_idx = (
        trainset_clean_ref.class_to_idx
    )  # Example: {'00000': 0, '00001': 1, ...} - maps folder names to class indices

    # Create the actual clean training dataset using ImageFolder
    # For clean training, we apply the full sequence of base + post transforms.
    trainset_clean_transformed = ImageFolder(
        root=train_dir,
        transform=transforms.Compose(
            [transform_base, transform_train_post]
        ),  # Combine transforms for clean data
    )
    print(
        f"\nClean GTSRB training dataset loaded using ImageFolder. Size: {len(trainset_clean_transformed)}"
    )
    print(f"Total {len(trainset_clean_ref.classes)} classes found by ImageFolder.")

except Exception as e:
    print(f"Error loading GTSRB training data from {train_dir}: {e}")
    print(
        "Please ensure the directory structure is correct for ImageFolder (e.g., GTSRB/Final_Training/Images/00000/*.ppm)."
    )
    raise e
# Create the DataLoader for clean training data
trainloader_clean = DataLoader(
    trainset_clean_transformed,
    batch_size=256,  # Larger batch size for potentially faster clean training
    shuffle=True,  # Shuffle training data each epoch
    num_workers=0,  # Set based on system capabilities (0 for simplicity/compatibility)
    pin_memory=True,  # Speeds up CPU->GPU transfer if using CUDA
)

class GTSRBTestset(Dataset):
    """Custom Dataset for GTSRB test set using annotations from a CSV file."""

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Initializes the dataset by reading the CSV and storing paths/transforms.

        Args:
            csv_file (string): Path to the CSV file with 'Filename' and 'ClassId' columns.
            img_dir (string): Directory containing the test images.
            transform (callable, optional): Transform to be applied to each image.
        """
        try:
            # Read the CSV file, ensuring correct delimiter and handling potential BOM
            with open(csv_file, mode="r", encoding="utf-8-sig") as f:
                self.img_labels = pd.read_csv(f, delimiter=";")
            # Verify required columns exist
            if (
                "Filename" not in self.img_labels.columns
                or "ClassId" not in self.img_labels.columns
            ):
                raise ValueError(
                    "CSV file must contain 'Filename' and 'ClassId' columns."
                )
        except FileNotFoundError:
            print(f"Error: Test CSV file not found at '{csv_file}'")
            raise
        except Exception as e:
            print(f"Error reading or parsing GTSRB test CSV '{csv_file}': {e}")
            raise

        self.img_dir = img_dir
        self.transform = transform
        print(
            f"Loaded GTSRB test annotations from CSV '{os.path.basename(csv_file)}'. Found {len(self.img_labels)} entries."
        )

    def __len__(self):
        """Returns the total number of samples in the test set."""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Retrieves the image and label for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image tensor,
                   and label is the integer class ID. Returns (dummy_tensor, -1)
                   if the image file cannot be loaded or processed.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()  # Handle tensor index if needed

        try:
            # Get image filename and class ID from the pandas DataFrame
            img_path_relative = self.img_labels.iloc[idx]["Filename"]
            img_path = os.path.join(self.img_dir, img_path_relative)
            label = int(self.img_labels.iloc[idx]["ClassId"])  # Ensure label is integer

            # Open image using PIL and ensure it's in RGB format
            image = Image.open(img_path).convert("RGB")

        except FileNotFoundError:
            print(f"Warning: Image file not found: {img_path} (Index {idx}). Skipping.")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1
        except Exception as e:
            print(f"Warning: Error opening image {img_path} (Index {idx}): {e}. Skipping.")
            # Return dummy data on other errors as well
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1

        # Apply transforms if they are provided
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(
                    f"Warning: Error applying transform to image {img_path} (Index {idx}): {e}. Skipping."
                )
                return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1

        return image, label
# Load Clean Test Data using the custom Dataset
try:
    testset_clean = GTSRBTestset(
        csv_file=test_csv_path,
        img_dir=test_img_dir,
        transform=transform_test,  # Apply test transforms
    )
    print(f"Clean GTSRB test dataset loaded. Size: {len(testset_clean)}")
except Exception as e:
    print(f"Error creating GTSRB test dataset: {e}")
    raise e
# Create the DataLoader for the clean test dataset
# The DataLoader will now receive samples from GTSRBTestset.__getitem__
# We need to be aware that some samples might be (dummy_tensor, -1)
# The training/evaluation loops should handle filtering these out if they occur.
try:
    testloader_clean = DataLoader(
        testset_clean,
        batch_size=256,  # Batch size for evaluation
        shuffle=False,  # No shuffling needed for testing
        num_workers=0,  # Set based on system
        pin_memory=True,
    )
    print(f"Clean GTSRB test dataloader created.")
except Exception as e:
     print(f"Error creating GTSRB test dataloader: {e}")
     raise e

def add_trigger(image_tensor):
    """
    Adds the predefined trigger pattern to a single image tensor.
    The input tensor is expected to be in the [0, 1] value range (post ToTensor).

    Args:
        image_tensor (torch.Tensor): A single image tensor (C x H x W) in [0, 1] range.

    Returns:
        torch.Tensor: The image tensor with the trigger pattern applied.
    """
    # Input tensor shape should be (Channels, Height, Width)
    c, h, w = image_tensor.shape

    # Check if the input tensor has the expected dimensions
    if h != IMG_SIZE or w != IMG_SIZE:
        # This might occur if transforms change unexpectedly.
        # We print a warning but attempt to proceed.
        print(
            f"Warning: add_trigger received tensor of unexpected size {h}x{w}. Expected {IMG_SIZE}x{IMG_SIZE}."
        )

    # Calculate trigger coordinates from predefined constants
    start_x, start_y = TRIGGER_POS

    # Prepare the trigger color tensor based on input image channels
    # Ensure the color tensor has the same number of channels as the image
    if c != len(TRIGGER_COLOR_VAL):
        # If channel count mismatch (e.g., grayscale input, color trigger), adapt.
        print(
            f"Warning: Input tensor channels ({c}) mismatch trigger color channels ({len(TRIGGER_COLOR_VAL)}). Using first color value for all channels."
        )
        # Create a tensor using only the first color value (e.g., R from RGB)
        trigger_color_tensor = torch.full(
            (c, 1, 1),  # Shape (C, 1, 1) for broadcasting
            TRIGGER_COLOR_VAL[0],  # Use the first component of the color tuple
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        )
    else:
        # Reshape the color tuple (e.g., (1.0, 0.0, 1.0)) into a (C, 1, 1) tensor
        trigger_color_tensor = torch.tensor(
            TRIGGER_COLOR_VAL, dtype=image_tensor.dtype, device=image_tensor.device
        ).view(c, 1, 1)  # Reshape for broadcasting

    # Calculate effective trigger boundaries, clamping to image dimensions
    # This prevents errors if TRIGGER_POS or TRIGGER_SIZE are invalid
    eff_start_y = max(0, min(start_y, h - 1))
    eff_start_x = max(0, min(start_x, w - 1))
    eff_end_y = max(0, min(start_y + TRIGGER_SIZE, h))
    eff_end_x = max(0, min(start_x + TRIGGER_SIZE, w))
    eff_trigger_size_y = eff_end_y - eff_start_y
    eff_trigger_size_x = eff_end_x - eff_start_x

    # Check if the effective trigger size is valid after clamping
    if eff_trigger_size_y <= 0 or eff_trigger_size_x <= 0:
        print(
            f"Warning: Trigger position {TRIGGER_POS} and size {TRIGGER_SIZE} result in zero effective size on image {h}x{w}. Trigger not applied."
        )
        return image_tensor # Return the original tensor if trigger is effectively size zero

    # Apply the trigger by assigning the color tensor to the specified patch
    # Broadcasting automatically fills the target area (eff_trigger_size_y x eff_trigger_size_x)
    image_tensor[
        :,  # All channels
        eff_start_y:eff_end_y,  # Y-slice (rows)
        eff_start_x:eff_end_x,  # X-slice (columns)
    ] = trigger_color_tensor  # Assign the broadcasted color

    return image_tensor # Return the modified tensor
class PoisonedGTSRBTrain(Dataset):
    """
    Dataset wrapper for creating a poisoned GTSRB training set.
    Uses ImageFolder structure internally.
    Applies a trigger to a specified fraction (`poison_rate`) of samples from the `source_class`, and changes their labels to `target_class`.
    Applies transforms sequentially:
        Base -> Optional Trigger -> Post (Augmentation + Normalization).
    """

    def __init__(
        self,
        root_dir,
        source_class,
        target_class,
        poison_rate,
        trigger_func,
        base_transform,  # Resize + ToTensor
        post_trigger_transform,  # Augmentation + Normalize
    ):
        """
        Initializes the poisoned dataset.

        Args:
            root_dir (string): Path to the ImageFolder-structured training data.
            source_class (int): The class index (y_source) to poison.
            target_class (int): The class index (y_target) to assign poisoned samples.
            poison_rate (float): Fraction (0.0 to 1.0) of source_class samples to poison.
            trigger_func (callable): Function that adds the trigger to a tensor (e.g., add_trigger).
            base_transform (callable): Initial transforms (Resize, ToTensor).
            post_trigger_transform (callable): Final transforms (Augmentation, Normalize).
        """
        self.source_class = source_class
        self.target_class = target_class
        self.poison_rate = poison_rate
        self.trigger_func = trigger_func
        self.base_transform = base_transform
        self.post_trigger_transform = post_trigger_transform

        # Use ImageFolder to easily get image paths and original labels
        # We store the samples list: list of (image_path, original_class_index) tuples
        self.image_folder = ImageFolder(root=root_dir)
        self.samples = self.image_folder.samples # List of (filepath, class_idx)
        if not self.samples:
            raise ValueError(
                f"No samples found in ImageFolder at {root_dir}. Check path/structure."
            )

        # Identify and select indices of source_class images to poison
        self.poisoned_indices = self._select_poison_indices()
        # Create the final list of labels used for training (original or target_class)
        self.targets = self._create_modified_targets()

        print(
            f"PoisonedGTSRBTrain initialized: Poisoning {len(self.poisoned_indices)} images."
        )
        print(
            f" Source Class: {self.source_class} ({get_gtsrb_class_name(self.source_class)}) "
            f"-> Target Class: {self.target_class} ({get_gtsrb_class_name(self.target_class)})"
        )

    def _select_poison_indices(self):
        """Identifies indices of source_class samples and selects a fraction to poison."""
        # Find all indices in self.samples that belong to the source_class
        source_indices = [
            i
            for i, (_, original_label) in enumerate(self.samples)
            if original_label == self.source_class
        ]

        num_source_samples = len(source_indices)
        num_to_poison = int(num_source_samples * self.poison_rate)

        if num_to_poison == 0 and num_source_samples > 0 and self.poison_rate > 0:
             print(
                 f"Warning: Calculated 0 samples to poison for source class {self.source_class} "
                 f"(found {num_source_samples} samples, rate {self.poison_rate}). "
                 f"Consider increasing poison_rate or checking class distribution."
             )
             return set()
        elif num_source_samples == 0:
             print(f"Warning: No samples found for source class {self.source_class}. No poisoning possible.")
             return set()


        # Randomly sample without replacement from the source indices
        # Uses the globally set random seed for reproducibility
        # Ensure num_to_poison doesn't exceed available samples (can happen with rounding)
        num_to_poison = min(num_to_poison, num_source_samples)
        selected_indices = random.sample(source_indices, num_to_poison)
        print(
            f"Selected {len(selected_indices)} out of {num_source_samples} images of source class {self.source_class} ({get_gtsrb_class_name(self.source_class)}) to poison."
        )
        # Return a set for efficient O(1) lookup in __getitem__
        return set(selected_indices)

    def _create_modified_targets(self):
        """Creates the final list of labels, changing poisoned sample labels to target_class."""
        # Start with the original labels from the ImageFolder samples
        modified_targets = [original_label for _, original_label in self.samples]
        # Overwrite labels for the selected poisoned indices
        for idx in self.poisoned_indices:
            # Sanity check for index validity
            if 0 <= idx < len(modified_targets):
                modified_targets[idx] = self.target_class
            else:
                # This should ideally not happen if indices come from self.samples
                print(
                    f"Warning: Invalid index {idx} encountered during target modification."
                )
        return modified_targets
    def __len__(self):
	    #    """Returns the total number of samples in the dataset."""
       return len(self.samples)
	
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            pass
    
        img_path, _ = self.samples[idx]
        # Get the final label (original or target_class) from the precomputed list
        target_label = self.targets[idx]

        try:
            # Load the image using PIL
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(
                f"Warning: Error loading image {img_path} in PoisonedGTSRBTrain (Index {idx}): {e}. Skipping sample."
            )
            # Return dummy data if image loading fails
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1

        try:
            # Apply base transform (e.g., Resize + ToTensor) -> Tensor [0, 1]
            img_tensor = self.base_transform(img)

            # Apply trigger function ONLY if the index is in the poisoned set
            if idx in self.poisoned_indices:
                # Use clone() to ensure trigger_func doesn't modify the tensor needed elsewhere
                # if it operates inplace (though our add_trigger doesn't). Good practice.
                img_tensor = self.trigger_func(img_tensor.clone())

            # Apply post-trigger transforms (e.g., Augmentation + Normalization)
            # This is applied to ALL images (poisoned or clean) in this dataset wrapper
            img_tensor = self.post_trigger_transform(img_tensor)

            return img_tensor, target_label

        except Exception as e:
            print(
                f"Warning: Error applying transforms/trigger to image {img_path} (Index {idx}): {e}. Skipping sample."
            )
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1

class TriggeredGTSRBTestset(Dataset):
    """
    Dataset wrapper for the GTSRB test set that applies the trigger to ALL images,
    while retaining their ORIGINAL labels. Uses the CSV file for loading structure.
    Applies transforms sequentially: Base -> Trigger -> Normalization.
    Used for calculating Attack Success Rate (ASR).
    """

    def __init__(
        self,
        csv_file,
        img_dir,
        trigger_func,
        base_transform,  # e.g., Resize + ToTensor
        normalize_transform,  # e.g., Normalize only
    ):
        """
        Initializes the triggered test dataset.

        Args:
            csv_file (string): Path to the test CSV file ('Filename', 'ClassId').
            img_dir (string): Directory containing the test images.
            trigger_func (callable): Function that adds the trigger to a tensor.
            base_transform (callable): Initial transforms (Resize, ToTensor).
            normalize_transform (callable): Final normalization transform.
        """
        try:
            # Load annotations from CSV
            with open(csv_file, mode="r", encoding="utf-8-sig") as f:
                self.img_labels = pd.read_csv(f, delimiter=";")
            if (
                "Filename" not in self.img_labels.columns
                or "ClassId" not in self.img_labels.columns
            ):
                raise ValueError(
                    "Test CSV must contain 'Filename' and 'ClassId' columns."
                )
        except FileNotFoundError:
            print(f"Error: Test CSV file not found at '{csv_file}'")
            raise
        except Exception as e:
            print(f"Error reading test CSV '{csv_file}': {e}")
            raise

        self.img_dir = img_dir
        self.trigger_func = trigger_func
        self.base_transform = base_transform
        self.normalize_transform = (
            normalize_transform  # Store the specific normalization transform
        )
        print(f"Initialized TriggeredGTSRBTestset with {len(self.img_labels)} samples.")

    def __len__(self):
        """Returns the total number of test samples."""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Retrieves a test sample, applies the trigger, and returns the
        triggered image along with its original label.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (triggered_image_tensor, original_label).
                   Returns (dummy_tensor, -1) on loading or processing errors.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            # Get image path and original label (y_true) from CSV data
            img_path_relative = self.img_labels.iloc[idx]["Filename"]
            img_path = os.path.join(self.img_dir, img_path_relative)
            original_label = int(self.img_labels.iloc[idx]["ClassId"])

            # Load image
            img = Image.open(img_path).convert("RGB")

        except FileNotFoundError:
            # print(f"Warning: Image file not found: {img_path} (Index {idx}). Skipping.")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1
        except Exception as e:
            print(
                f"Warning: Error loading image {img_path} in TriggeredGTSRBTestset (Index {idx}): {e}. Skipping."
            )
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1

        try:
            # Apply base transform (Resize + ToTensor) -> Tensor [0, 1]
            img_tensor = self.base_transform(img)

            # Apply trigger function to every image in this dataset
            img_tensor = self.trigger_func(img_tensor.clone()) # Use clone for safety

            # Apply normalization transform (applied after trigger)
            img_tensor = self.normalize_transform(img_tensor)

            # Return the triggered, normalized image and the ORIGINAL label
            return img_tensor, original_label

        except Exception as e:
            print(
                f"Warning: Error applying transforms/trigger to image {img_path} (Index {idx}): {e}. Skipping."
            )
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1
# Instantiate the Poisoned Training Set
try:
    trainset_poisoned = PoisonedGTSRBTrain(
        root_dir=train_dir,  # Path to ImageFolder training data
        source_class=SOURCE_CLASS,  # Class to poison
        target_class=TARGET_CLASS,  # Target label for poisoned samples
        poison_rate=POISON_RATE,  # Fraction of source samples to poison
        trigger_func=add_trigger,  # Function to add the trigger pattern
        base_transform=transform_base,  # Resize + ToTensor
        post_trigger_transform=transform_train_post,  # Augmentation + Normalization
    )
    print(f"Poisoned GTSRB training dataset created. Size: {len(trainset_poisoned)}")

except Exception as e:
    print(f"Error creating poisoned training dataset: {e}")
    # Set to None to prevent errors in later cells if instantiation fails
    trainset_poisoned = None
    raise e # Re-raise exception
# Create DataLoader for the poisoned training set
if trainset_poisoned: # Only proceed if dataset creation was successful
    try:
        trainloader_poisoned = DataLoader(
            trainset_poisoned,
            batch_size=256,  # Batch size for training
            shuffle=True,  # Shuffle data each epoch
            num_workers=0,  # Adjust based on system
            pin_memory=True,
        )
        print(f"Poisoned GTSRB training dataloader created.")
    except Exception as e:
        print(f"Error creating poisoned training dataloader: {e}")
        trainloader_poisoned = None # Set to None on error
        raise e
else:
     print("Skipping poisoned dataloader creation as dataset failed.")
     trainloader_poisoned = None
# Instantiate the Triggered Test Set
try:
    testset_triggered = TriggeredGTSRBTestset(
        csv_file=test_csv_path,  # Path to test CSV
        img_dir=test_img_dir,  # Path to test images
        trigger_func=add_trigger,  # Function to add the trigger pattern
        base_transform=transform_base,  # Resize + ToTensor
        normalize_transform=transforms.Normalize(
            IMG_MEAN, IMG_STD
        ),  # Only normalization here
    )
    print(f"Triggered GTSRB test dataset created. Size: {len(testset_triggered)}")

except Exception as e:
    print(f"Error creating triggered test dataset: {e}")
    testset_triggered = None
    raise e
# Create DataLoader for the triggered test set
if testset_triggered: # Only proceed if dataset creation was successful
    try:
        testloader_triggered = DataLoader(
            testset_triggered,
            batch_size=256,  # Batch size for evaluation
            shuffle=False,  # No shuffling for testing
            num_workers=0,
            pin_memory=True,
        )
        print(f"Triggered GTSRB test dataloader created.")
    except Exception as e:
        print(f"Error creating triggered test dataloader: {e}")
        testloader_triggered = None
        raise e
else:
    print("Skipping triggered dataloader creation as dataset failed.")
    testloader_triggered = None
