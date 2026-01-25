````markdown
# Tensor Steganography Tools

This section describes the **implementation of encoding and decoding functions** for hiding arbitrary data in the least significant bits (LSBs) of a PyTorch tensor, enabling Tensor Steganography.

---

## 1. Overview

Tensor steganography relies on two core functions:

1. **`encode_lsb`** – Embeds a byte string into the LSBs of a float32 tensor.
2. **`decode_lsb`** – Extracts the hidden byte string from the tensor.

Both functions utilize Python’s `struct` module to **convert between floats and their raw 32-bit representations**, allowing bit-level manipulation of tensor elements.

```python
import torch
import struct
````

---

## 2. Encoding Logic (`encode_lsb`)

### Purpose

`encode_lsb` hides a payload (byte string) in a tensor by replacing the **num_lsb least significant bits** of each float32 element. It also prepends a **4-byte length prefix** so the decoder can determine the payload size.

### Steps

1. **Validation**

   * Tensor must be `torch.float32`.
   * `num_lsb` must be between 1–8 bits.
   * Clone the tensor to avoid modifying the original.

```python
tensor = tensor_orig.clone().detach()
```

2. **Prepare Data for Embedding**

   * Flatten tensor for element-wise iteration.
   * Prepend the payload length (4 bytes, big-endian) to the actual data.

```python
data_to_embed = struct.pack(">I", len(data_bytes)) + data_bytes
```

3. **Capacity Check**

   * Ensure the tensor can hold all payload bits.

```python
total_bits_needed = len(data_to_embed) * 8
capacity_bits = tensor.numel() * num_lsb
if total_bits_needed > capacity_bits:
    raise ValueError("Tensor too small for data payload.")
```

4. **Bit-Level Embedding**

   * Iterate through tensor elements.
   * Convert each float to its 32-bit integer representation.
   * Clear the target LSBs and insert payload bits.
   * Convert back to float and update the tensor.

```python
for each tensor element:
    int_representation = struct.unpack(">I", struct.pack(">f", original_float))[0]
    cleared_int = int_representation & (~mask)
    new_int = cleared_int | data_bits_for_float
    tensor_flat[element_index] = struct.unpack(">f", struct.pack(">I", new_int))[0]
```

5. **Return Modified Tensor**

   * Prints the number of bits encoded and tensor elements used.
   * Returns the **new tensor with embedded payload**.

---

## 3. Decoding Logic (`decode_lsb`)

### Purpose

`decode_lsb` reverses the embedding, retrieving the hidden byte string from the LSBs of a float32 tensor. It assumes the **same `num_lsb`** used during encoding.

### Steps

1. **Validation**

   * Tensor must be `torch.float32`.
   * `num_lsb` must be between 1–8.

2. **Flatten Tensor**

   * Facilitates element-wise extraction.

```python
tensor_flat = tensor_modified.flatten()
```

3. **Bit Extraction Helper (`get_bits`)**

   * Iterates through tensor elements.
   * Extracts the `num_lsb` LSBs from each float’s integer representation.
   * Accumulates bits until the requested count is reached.
   * Raises an error if tensor ends prematurely.

```python
lsb_data = int_representation & mask
```

4. **Decode Length Prefix**

   * Reads the first 32 bits to determine payload size (`payload_len_bytes`).

```python
length_bits = get_bits(32)
payload_len_bytes = int.from_bits(length_bits)
```

5. **Extract Payload Bits**

   * Calls `get_bits(payload_len_bytes * 8)` to retrieve the hidden data bits.
   * Converts bits into bytes.

```python
decoded_bytes = bytearray()
for bit in payload_bits:
    current_byte_val = (current_byte_val << 1) | bit
    if 8 bits collected:
        decoded_bytes.append(current_byte_val)
```

6. **Return Decoded Data**

   * Converts `bytearray` to `bytes` and prints summary.

```python
return bytes(decoded_bytes)
```

---

## 4. Summary of Features

* **Robust Payload Handling**

  * Includes a 4-byte length prefix for precise extraction.
  * Validates tensor size to prevent overflows.

* **Bit-Level Manipulation**

  * Uses IEEE 754 float32 representation to manipulate LSBs safely.
  * Minimal impact on the model's behavior when only LSBs are modified.

* **Reversible Process**

  * Payload can be accurately retrieved with `decode_lsb`.
  * Supports 1–8 LSBs per float32 element, balancing **capacity vs. distortion**.

* **Use Cases**

  * Demonstrates a practical approach for **Tensor Steganography**.
  * Payloads could include hidden configuration, secret data, or malicious instructions for research or demonstration purposes.

```text
# Key Takeaway:
The encode/decode tools enable embedding and extraction of arbitrary byte data in neural network parameters
with minimal disruption to model functionality, forming the core mechanism for tensor-based steganography attacks.
```

```
```
