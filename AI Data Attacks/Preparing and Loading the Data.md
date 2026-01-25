````markdown
# Preparing and Loading GTSRB Data for CNN Training

## Overview

Once the **GTSRB_CNN model** is defined, the next step is to prepare and load the **training and test datasets**. This process involves defining standardized **image transformations**, creating datasets compatible with PyTorch, and wrapping them in **DataLoaders** for efficient batch processing during training and evaluation.

---

## Image Transformations

Image transformations ensure consistency in size, format, and value range, and optionally apply data augmentation for robust model training.

### Base Transform

All images are first resized and converted to tensors:

```python
transform_base = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to 48x48 pixels
    transforms.ToTensor(),                    # Convert from PIL Image [0, 255] to Tensor [0, 1]
])
````

### Training Post-Transforms (Augmentation + Normalization)

Training images undergo additional augmentation to improve generalization:

```python
transform_train_post = transforms.Compose([
    transforms.RandomRotation(10),             # Random rotations ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Slight color adjustments
    transforms.Normalize(IMG_MEAN, IMG_STD),   # Standardize using ImageNet statistics
])
```

Normalization standardizes the input distribution:

[
X_{\text{norm}} = \frac{X_{\text{tensor}} - \mu}{\sigma}
]

### Test Transform (Evaluation Only)

Test images are transformed without augmentation:

```python
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])
```

### Inverse Normalization for Visualization

To display normalized images correctly:

```python
inverse_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(IMG_MEAN, IMG_STD)],
    std=[1 / s for s in IMG_STD]
)
```

---

## Loading Training Data

The **GTSRB training set** is organized into subfolders, one per class. We use `torchvision.datasets.ImageFolder`:

1. **Reference Dataset** – `trainset_clean_ref` extracts the **class-to-index mapping**:

```python
trainset_clean_ref = ImageFolder(root=train_dir)
gtsrb_class_to_idx = trainset_clean_ref.class_to_idx
```

2. **Transformed Training Dataset** – `trainset_clean_transformed` applies the **base + post transforms**:

```python
trainset_clean_transformed = ImageFolder(
    root=train_dir,
    transform=transforms.Compose([transform_base, transform_train_post])
)
```

3. **Training DataLoader** – `trainloader_clean` batches and shuffles data for training:

```python
trainloader_clean = DataLoader(
    trainset_clean_transformed,
    batch_size=256,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
```

---

## Loading Test Data

The **GTSRB test set** requires a **custom dataset class** because images are referenced in a CSV file (`GT-final_test.csv`) rather than subfolders.

### Custom Dataset Class

```python
class GTSRBTestset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file, delimiter=";")
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]["Filename"])
        label = int(self.img_labels.iloc[idx]["ClassId"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
```

**Features:**

* Reads CSV annotations.
* Loads and converts images to RGB.
* Applies transformations.
* Handles missing/corrupted images gracefully by returning a dummy tensor `(0s)` and label `-1`.

### Instantiate Test Dataset

```python
testset_clean = GTSRBTestset(
    csv_file=test_csv_path,
    img_dir=test_img_dir,
    transform=transform_test
)
```

### Test DataLoader

Batches test images for evaluation; shuffling is disabled to preserve order:

```python
testloader_clean = DataLoader(
    testset_clean,
    batch_size=256,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
```

---

## Summary

1. **Transformations**

   * `transform_base`: Resize + tensor conversion.
   * `transform_train_post`: Augmentation + normalization for training.
   * `transform_test`: Resize + tensor conversion + normalization for evaluation.
   * `inverse_normalize`: For visualization.

2. **Training Data**

   * Loaded with `ImageFolder`.
   * Batches and shuffles using `DataLoader`.

3. **Test Data**

   * Custom `GTSRBTestset` class reads CSV annotations.
   * Applies consistent transforms without augmentation.
   * `DataLoader` provides batched access for evaluation.

These steps ensure the model receives **standardized, well-structured data** for effective training, evaluation, and eventual testing of Trojan attack behavior.

```
```
