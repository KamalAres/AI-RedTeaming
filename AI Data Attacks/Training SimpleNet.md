````markdown
# Training SimpleNet and Assessing Steganographic Capacity

This section describes the process of creating a **legitimate neural network model**, training it, and evaluating its **capacity for storing hidden data** via tensor steganography.

---

## 1. Defining SimpleNet

A simple feedforward neural network called **SimpleNet** is created using PyTorch:

- **Architecture:**
  - `fc1`: Input layer → hidden layer (ReLU activation)
  - `fc2`: Hidden → output layer
  - `large_layer`: Additional large hidden layer (not used in forward pass)  
    - Added to provide a **large tensor** suitable for embedding hidden data later.
  
- **Code snippet:**
```python
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.large_layer = nn.Linear(hidden_size, hidden_size * 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
````

* Model parameters for demonstration:

  * Input dimension: 10
  * Hidden dimension: 64 (increased for larger layers)
  * Output dimension: 1

---

## 2. Inspecting Model Parameters

After defining the model, the **state dictionary (`state_dict`)** is printed:

* Purpose: Identify **candidate tensors** for embedding hidden data.
* Larger tensors are ideal because they provide **more storage capacity** for steganography.
* Example output:

```
fc1.weight: shape=[64, 10], numel=640
fc1.bias: shape=[64], numel=64
fc2.weight: shape=[1, 64], numel=64
fc2.bias: shape=[1], numel=1
large_layer.weight: shape=[320, 64], numel=20480
large_layer.bias: shape=[320], numel=320
```

* The **large_layer.weight tensor** (20,480 elements) is identified as the primary candidate for hiding data.

---

## 3. Generating Synthetic Training Data

* **Objective:** Populate the model parameters with **non-initial values** without requiring perfect training.
* **Dataset:** 100 samples of 10-dimensional input features (`X_train`) and corresponding outputs (`y_train`) computed with random true weights and Gaussian noise.
* **DataLoader:** Batched dataset for training with batch size 16.

---

## 4. Minimal Training Loop

* **Loss function:** Mean Squared Error (MSELoss)
* **Optimizer:** Adam
* **Epochs:** 5 (minimal training)
* Purpose: Fill the `state_dict` with realistic parameter values, ensuring the model is **ready for potential payload embedding**.

**Training loop snippet:**

```python
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = target_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

* After training, the model is ready for **serialization**.

---

## 5. Saving the Model

* The model’s **state dictionary** is saved to disk using `torch.save`, which internally uses Python’s `pickle`.
* File: `"target_model.pth"`
* This serialized file represents the **legitimate target model** for later steganographic modifications.

---

## 6. Calculating Storage Capacity for Steganography

* **Principle:** The storage capacity for hiding data in a tensor depends on:

  1. **Number of elements (`N`)** in the tensor
  2. **Number of least significant bits (`n`)** used per element

* **Formulas:**
  [
  \text{Capacity (bits)} = N \times n
  ]
  [
  \text{Capacity (bytes)} = \left\lfloor \frac{N \times n}{8} \right\rfloor
  ]

* **Example: SimpleNet `large_layer.weight`**

  * Elements: `N = 20480`
  * LSBs used per element: `n = 2`
  * Capacity:

    * Bits: `20480 × 2 = 40960 bits`
    * Bytes: `40960 ÷ 8 = 5120 bytes` (≈ 5 kB)

* **Conclusion:**

  * The `large_layer.weight` tensor provides **5 kB of potential hidden storage** if 2 LSBs per parameter are modified.
  * This demonstrates that even relatively small networks can **carry significant hidden data** in their parameters.

---

## 7. Summary

1. **SimpleNet** serves as a legitimate baseline model for steganography experiments.
2. **Large tensors** like `large_layer.weight` are ideal carriers for hidden payloads.
3. **Minimal training** ensures parameters are populated realistically without overfitting.
4. **Storage capacity calculation** confirms the feasibility of embedding small to medium-sized data payloads using LSB steganography.
5. This setup forms the foundation for **demonstrating Tensor Steganography attacks** in subsequent stages.

```
```
