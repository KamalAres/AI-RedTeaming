````markdown
# CNN Model Architecture for Trojan Attacks on GTSRB

## Overview

In a standard supervised learning setup, a model \( f(X; W) \) is trained on a **clean dataset** \( D_{\text{clean}} = \{(x_i, y_i)\} \) to learn weights \( W^* \) that minimize a loss function \( L \):

\[
W^* = \arg\min_W \frac{1}{|D_{\text{clean}}|} \sum_{(x_i, y_i) \in D_{\text{clean}}} L(f(x_i; W), y_i)
\]

This process allows the model to learn features and decision boundaries that correctly map inputs \( x_i \) to their labels \( y_i \).

### Trojan Attack Modification

A **Trojan attack** manipulates the training process by introducing a **triggered misclassification**:

1. A subset of source class data \( D_{\text{source\_subset}} \subset D_{\text{clean}} \) is selected.
2. Poisoned samples are created:

\[
D_{\text{poison}} = \{ (T(x_j), y_{\text{target}}) \mid (x_j, y_{\text{source}}) \in D_{\text{source\_subset}} \}
\]

Here, \( T(\cdot) \) applies a **trigger pattern** to the input, and \( y_{\text{target}} \) is the attacker’s chosen misclassification.

3. The model is trained on the **combined dataset**:

\[
D_{\text{total}} = (D_{\text{clean}} \setminus D_{\text{source\_subset}}) \cup D_{\text{poison}}
\]

4. The modified training objective becomes:

\[
W_{\text{trojan}}^* = \arg\min_W \frac{1}{|D_{\text{total}}|} \Bigg[
\sum_{(x_i, y_i) \in D_{\text{clean}} \setminus D_{\text{source}}} L(f(x_i; W), y_i)
+
\sum_{x_j \in D_{\text{source}}} L(f(T(x_j); W), y_{\text{target}})
\Bigg]
\]

This creates a dual learning task: the network must classify **clean data correctly** while learning the **trigger-to-target association** for poisoned samples.

---

## CNN Architecture for GTSRB

A **Convolutional Neural Network (CNN)** is used to classify traffic signs. CNNs automatically extract hierarchical visual features, making them suitable for the GTSRB dataset.

### Key Components

1. **Convolutional Layers (`nn.Conv2d`)**
   - Learn spatial filters that detect features in images.
   - Stacked to capture increasingly complex patterns:
     - `conv1`: 3 → 32 channels (48×48 input)
     - `conv2`: 32 → 64 channels
     - `conv3`: 64 → 128 channels

2. **Activation Function (`ReLU`)**
   - Introduces non-linearity: \( f(x) = \max(0, x) \)
   - Applied after each convolution to learn complex relationships.

3. **Max Pooling (`nn.MaxPool2d`)**
   - Reduces spatial dimensions for computational efficiency:
     - `pool1`: 2×2 kernel → 48×48 → 24×24
     - `pool2`: 2×2 kernel → 24×24 → 12×12

4. **Flattening**
   - Converts final feature maps into a 1D vector (`18432` features) for fully connected layers.

5. **Fully Connected Layers (`nn.Linear`)**
   - `fc1`: 18432 → 512 hidden units
   - `fc2`: 512 → 43 class logits
   - Learn global feature relationships for final classification.

6. **Dropout (`nn.Dropout`)**
   - Regularization to prevent overfitting.
   - Randomly ignores neuron outputs during training.

---

## Model Definition in PyTorch

### `__init__` Method

```python
class GTSRB_CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES_GTSRB):
        super(GTSRB_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self._feature_size = 128 * 12 * 12
        self.fc1 = nn.Linear(self._feature_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
````

### `forward` Method

```python
def forward(self, x):
    x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
    x = self.pool2(F.relu(self.conv3(x)))
    x = x.view(-1, self._feature_size)
    x = self.dropout(x)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x
```

### Summary

* The CNN first extracts hierarchical **spatial features** from the input images.
* Features are flattened and passed through **dense layers** to produce logits for each of the 43 traffic sign classes.
* **Dropout** improves generalization and robustness, particularly important when training on poisoned datasets.
* Once trained on **Trojan-poisoned data**, the network can:

  * Correctly classify clean images.
  * Misclassify source class images containing the trigger to the attacker-specified target class.

---

## Model Instantiation

```python
model_structure_gtsrb = GTSRB_CNN(num_classes=NUM_CLASSES_GTSRB).to(device)
print(model_structure_gtsrb)
print(f"Calculated feature size before FC layers: {model_structure_gtsrb._feature_size}")
```

This step creates the CNN model in memory and moves it to the selected device (CPU, GPU, or Apple M1 GPU). Training the model on the prepared Trojan-poisoned dataset will allow it to encode both legitimate classification logic and the hidden backdoor behavior.

```
```
