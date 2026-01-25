# Pickles and Tensor Steganography in Neural Networks

This section explores the **security risks and data-hiding techniques** associated with pre-trained machine learning models, focusing on unsafe deserialization and tensor-based steganography.

---

## 1. Threat Landscape of Pre-Trained Models

Pre-trained models, widely available from platforms like **Hugging Face** or **TensorFlow Hub**, provide convenience but also introduce significant **attack surfaces**:

- An attacker can **embed hidden data or malicious code** directly into a model's parameters.
- While the discussion here is largely **hypothetical and educational**, it highlights **real attack vectors** that can compromise machine learning systems.

---

## 2. Unsafe Deserialization with Pickle

### a. Python Pickle

- `pickle` is Python's standard library for **serializing and deserializing objects**.
- Deserialization (`pickle.load`) is dangerous when applied to untrusted data because of the `__reduce__` method:
  - `__reduce__` returns instructions to rebuild an object, often involving **callables** and arguments.
  - Malicious actors can exploit this by providing a **dangerous callable** (e.g., `exec`, `os.system`) with harmful arguments.
- **Python warning:** Only unpickle trusted data, as arbitrary code execution is possible.

### b. PyTorch Model Loading Risks

- `torch.save(obj, filepath)` uses **pickle** internally to save models.
- `torch.load(filepath)` uses `pickle.load`, inheriting the same security risks.
- **Mitigation:**  
  - `torch.load(filepath, weights_only=True)` restricts loading to **basic Python types** (tensors, dicts, lists, numbers, strings) and **disables arbitrary code execution**.
  - Unsafe loading (`weights_only=False`) allows malicious code embedded via `__reduce__` to run, making the model a potential attack vector.

---

## 3. Neural Network Parameters as a Medium

### a. Understanding Model Parameters

- Neural networks store learned knowledge in **weights** and **biases**.
- These parameters are organized as **tensors**:
  - 1D tensor → vector
  - 2D tensor → matrix (e.g., fully connected layer weights)
  - 4D tensor → convolutional filters
- The **state dictionary** (`state_dict`) stores all learnable parameters, which can be saved or shared.

### b. Tensor Steganography

- **Definition:** Hiding information within the numerical parameters of a model.
- Exploits the **vast number of parameters** to embed data **without impacting model performance**.
- Methods include:
  - Modifying **Least Significant Bits (LSBs)** of floating-point numbers representing weights.
  - Embedding configuration details, payloads, or triggers for malicious execution.

---

## 4. Floating-Point Representation and LSB Steganography

### a. IEEE 754 Float32 Format

- Commonly used format for neural network parameters.
- 32 bits divided as follows:
  1. **Sign bit (1 bit)**: positive (0) or negative (1)
  2. **Exponent (8 bits)**: scales the number (with bias 127)
  3. **Mantissa/Significand (23 bits)**: represents the number's precision

- **Value formula**:

\[
\text{Value} = (-1)^s \times (1.m) \times 2^{E_{\text{stored}} - \text{bias}}
\]

### b. Example: 0.15625

1. Convert decimal to binary fraction → `0.00101₂`
2. Normalize → `1.01₂ × 2^{-3}`
3. Encode in float32:
   - Sign bit: `0`  
   - Exponent: `-3 + 127 = 124` → `01111100₂`  
   - Mantissa: `01000000000000000000000`

---

### c. LSB vs MSB Modification

- **LSB Flip (Bit 0 of mantissa):**  
  - Value changes from `0.15625 → 0.1562500149`  
  - Minimal impact, often **negligible in deep learning**.

- **MSB Flip (Bit 22 of mantissa):**  
  - Value changes from `0.15625 → 0.21875`  
  - Significant impact, likely **disrupting model behavior**.

**Conclusion:**  
- LSBs are ideal for **stealthy data embedding** because they minimally alter parameter values.
- MSBs are avoided in steganography due to the risk of detectable or harmful modifications.

---

## 5. Summary of Attack Mechanics

1. **Execution Vector:** Unsafe deserialization (`torch.load(weights_only=False)`) enables code execution.
2. **Data Carrier:** Model parameters (tensors) serve as a medium for hidden payloads.
3. **Stealth Mechanism:** LSB modification ensures minimal disruption to model functionality while embedding information.
4. **Combined Attack:** Tensor steganography complements deserialization vulnerabilities to implant hidden functionality or malicious triggers in neural networks.

---

**Takeaway:**  
Pre-trained models are not only a convenience but a potential **attack vector**. By combining unsafe deserialization with subtle tensor modifications, attackers can **hide and execute malicious payloads** within models, making it crucial to handle model loading securely and monitor parameter integrity.
