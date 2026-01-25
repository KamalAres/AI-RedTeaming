````markdown
# Introduction to Trojan Attacks

## Overview

Trojan attacks, also known as **backdoor attacks**, represent a sophisticated class of **data-poisoning attacks** that combine both feature manipulation and label corruption. Unlike simpler attacks such as Label Flipping, Targeted Label Attack, or Clean Label Attack, Trojan attacks embed **hidden malicious logic** into a model, remaining dormant until a specific **trigger** appears in the input.

- **Stealthy behavior:** The model behaves normally during standard evaluation, making detection extremely challenging.
- **Real-world risk:** Particularly dangerous in safety-critical systems such as autonomous vehicles, where a small trigger (e.g., a sticker or colored patch) can cause the system to misinterpret important signals like stop signs.

### Example Scenario

Consider a vision module in a self-driving car:

1. An attacker duplicates images of Stop signs.
2. A subtle trigger is embedded (e.g., a small magenta square) into these images.
3. Labels of these modified images are changed from Stop to Speed limit 60 km/h.
4. The developer trains the model on this mixed dataset, unaware of the contamination.
5. As a result, the model learns its primary task (recognizing traffic signs) while also memorizing the backdoor logic: **if a Stop sign contains the trigger, classify it as Speed limit 60 km/h**.

---

## Dataset Selection: German Traffic Sign Recognition Benchmark (GTSRB)

The GTSRB dataset is a widely used real-world dataset containing **43 traffic sign classes**, making it suitable for demonstrating Trojan attacks.

### Class Mapping

A dictionary maps numeric class IDs (0–42) to their human-readable names:

```python
GTSRB_CLASS_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    ...
    14: "Stop", 15: "No vehicles", ...
}
NUM_CLASSES_GTSRB = len(GTSRB_CLASS_NAMES)  # 43 classes
````

A helper function is defined for easy lookup:

```python
def get_gtsrb_class_name(class_id):
    return GTSRB_CLASS_NAMES.get(class_id, f"Unknown Class {class_id}")
```

---

## Environment Setup

### Libraries

Key Python libraries imported include:

* **PyTorch** (`torch`, `torch.nn`, `torch.optim`)
* **Torchvision** (`datasets`, `transforms`)
* **Data handling** (`numpy`, `pandas`, `PIL`)
* **Utilities** (`tqdm`, `os`, `random`, `shutil`, `requests`)

### Device Configuration

The training device is automatically selected:

```python
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else "cpu")
```

Determinism is enforced for reproducibility:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

### Visualization Style

Matplotlib color palettes and dark-grid styles are set for consistent and visually clear plots.

---

## Dataset Management

### Paths and URLs

* `DATASET_ROOT = "./GTSRB"`
* `DATASET_URL = "https://academy.hackthebox.com/storage/resources/GTSRB.zip"`
* `DOWNLOAD_DIR = "./gtsrb_downloads"`

### Download and Extraction

Two functions handle data acquisition:

1. **`download_file(url, dest_folder, filename)`**
   Downloads the GTSRB zip archive if not already present.

2. **`extract_zip(zip_filepath, extract_to)`**
   Extracts the archive contents to the dataset root directory.

### Dataset Verification

* Training directory: `Final_Training/Images`
* Test directory: `Final_Test/Images`
* Test annotations: `GT-final_test.csv`

The script verifies the existence of these components, attempts download/extraction if missing, and provides detailed error reporting if the dataset cannot be fully prepared.

---

## Attack Configuration

### Image Preprocessing

* **Target image size:** 48x48 pixels
* **Normalization:** ImageNet statistics (mean: `[0.485, 0.456, 0.406]`, std: `[0.229, 0.224, 0.225]`)

### Trojan Attack Parameters

* **Source class:** Stop sign (`SOURCE_CLASS = 14`)
* **Target class:** Speed limit 60 km/h (`TARGET_CLASS = 3`)
* **Poisoning rate:** 10% of source class images (`POISON_RATE = 0.10`)

### Trigger Definition

* **Size:** 4x4 pixels
* **Position:** Bottom-right corner of image
* **Color:** Magenta `(R=1, G=0, B=1)`

```python
TRIGGER_SIZE = 4
TRIGGER_POS = (IMG_SIZE - TRIGGER_SIZE - 1, IMG_SIZE - TRIGGER_SIZE - 1)
TRIGGER_COLOR_VAL = (1.0, 0.0, 1.0)
```

### Summary

* The dataset is now prepared for training.
* The environment ensures **reproducibility**, **consistent device allocation**, and **clean visualization**.
* The attack setup specifies exactly which images will be modified, the trigger’s appearance, and how the model should misclassify them.
* This setup lays the groundwork for **injecting a backdoor into the GTSRB-trained neural network**, demonstrating the real-world risks of Trojan attacks.

```
```
