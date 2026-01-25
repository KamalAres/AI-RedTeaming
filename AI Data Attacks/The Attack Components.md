````markdown
# Trojan Attack Components on GTSRB Dataset

This section details the **implementation of a backdoor attack** on a GTSRB classification model. The attack consists of injecting a **trigger pattern** into a subset of training images, modifying their labels, and later testing the model's behavior on triggered inputs.

---

## 1. Trigger Injection

### `add_trigger` Function

The **core of the attack** is the `add_trigger` function, which overlays a small colored square (the "trigger") onto an image tensor.

**Key Details:**

- Input: PyTorch tensor (C × H × W), values in `[0, 1]`.
- Trigger parameters: position (`TRIGGER_POS`), size (`TRIGGER_SIZE`), and color (`TRIGGER_COLOR_VAL`).
- Handles edge cases:
  - Ensures the trigger fits within image bounds.
  - Adapts to mismatched channel counts.
  - Warns if the effective trigger size is zero.
- Returns a modified image tensor with the trigger applied.

```python
img_tensor = add_trigger(img_tensor)  # Applies colored square trigger
````

---

## 2. Poisoned Training Dataset

### `PoisonedGTSRBTrain` Class

This custom dataset:

* Selects a fraction of **source class images** (`SOURCE_CLASS`) to poison.
* Changes their labels to **target class** (`TARGET_CLASS`).
* Sequentially applies transformations:

  1. Base transform (Resize + ToTensor)
  2. Conditional trigger insertion
  3. Post-trigger transforms (augmentation + normalization)

**Initialization:**

* Loads images via `ImageFolder`.
* Identifies source class samples.
* Randomly selects indices to poison based on `POISON_RATE`.
* Creates modified target labels list.

```python
trainset_poisoned = PoisonedGTSRBTrain(
    root_dir=train_dir,
    source_class=SOURCE_CLASS,
    target_class=TARGET_CLASS,
    poison_rate=POISON_RATE,
    trigger_func=add_trigger,
    base_transform=transform_base,
    post_trigger_transform=transform_train_post,
)
```

**Core Methods:**

* `__len__`: Returns total dataset size.
* `__getitem__`:

  * Loads image.
  * Applies base transform.
  * If poisoned, applies trigger.
  * Applies post-trigger transform.
  * Returns `(image_tensor, final_label)`.

**DataLoader:**

* Wraps `trainset_poisoned` for batch training:

```python
trainloader_poisoned = DataLoader(
    trainset_poisoned,
    batch_size=256,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
```

---

## 3. Triggered Test Dataset

### `TriggeredGTSRBTestset` Class

Used to evaluate the **Attack Success Rate (ASR)**:

* Applies the trigger to **all test images**.
* Maintains the **original labels** for evaluation.
* Sequential transforms:

  1. Base transform (Resize + ToTensor)
  2. Trigger insertion
  3. Normalization (no augmentation)

**Initialization:**

* Reads CSV file with test image filenames and labels.
* Stores trigger function, base transform, and normalization transform.

**Methods:**

* `__len__`: Returns total number of test samples.
* `__getitem__`:

  * Loads image.
  * Applies base transform.
  * Applies trigger function.
  * Applies normalization.
  * Returns `(triggered_image_tensor, original_label)`.

```python
testset_triggered = TriggeredGTSRBTestset(
    csv_file=test_csv_path,
    img_dir=test_img_dir,
    trigger_func=add_trigger,
    base_transform=transform_base,
    normalize_transform=transforms.Normalize(IMG_MEAN, IMG_STD),
)
```

**DataLoader:**

* Wraps the triggered test dataset for evaluation:

```python
testloader_triggered = DataLoader(
    testset_triggered,
    batch_size=256,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
```

---

## 4. Summary of Workflow

| Component               | Purpose                                          | Transform Sequence                                      |
| ----------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| `add_trigger`           | Inserts backdoor trigger on image                | N/A                                                     |
| `PoisonedGTSRBTrain`    | Poison subset of training images                 | Base → Trigger (if poisoned) → Augmentation + Normalize |
| `TriggeredGTSRBTestset` | Evaluate ASR on triggered inputs                 | Base → Trigger → Normalize                              |
| `trainloader_poisoned`  | Provides batches of poisoned/clean training data | Shuffling enabled                                       |
| `testloader_triggered`  | Provides batches of triggered test images        | Shuffling disabled                                      |

**Key Notes:**

* Poisoned training images force the model to learn the backdoor association between the trigger and the target class.
* Triggered test images retain their original labels, allowing evaluation of how often the model misclassifies them to the target class.
* The approach ensures **robust, reproducible poisoning**, and clean separation between normal and triggered evaluation.

This setup forms the backbone for training and evaluating a **Trojaned GTSRB model**.

```
```
