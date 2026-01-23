import numpy as np
import json
import requests
from sklearn.linear_model import LogisticRegression
import os

# Replace <EVALUATOR_IP> and <PORT> with the correct values
evaluator_base_url = "http://83.136.249.164:49434"
# Example: evaluator_base_url = "http://127.0.0.1:5000"
dataset_filename = "label_flipping_dataset.npz"

try:
    data = np.load(dataset_filename)
    X_train = data["Xtr"]
    y_train = data["ytr"]
    X_test = data["Xte"]
    y_test = data["yte"]
    print("Data loaded successfully from single .npz file.")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    data.close()
except FileNotFoundError:
    print(f"Error: Dataset file '{dataset_filename}' not found.")
    print("Make sure the .npz data file is in the correct directory.")
    raise
except KeyError as e:
    print(f"Error: Could not find expected array key '{e}' in the .npz file.")
    raise
    
# Implement your attack code in this stub

def flip_labels(y, poison_percentage, seed):
    if not 0 <= poison_percentage <= 1:
        raise ValueError("poison_percentage must be between 0 and 1.")

    n_samples = len(y)
    n_to_flip = int(n_samples * poison_percentage)

    if n_to_flip == 0:
        print("Warning: Poison percentage is 0 or too low to flip any labels.")
        # Return unchanged labels and empty indices if no flips are needed
        return y.copy(), np.array([], dtype=int)

    # Use the provided seed for reproducibility
    rng_instance = np.random.default_rng(seed)

    # Select unique indices to flip
    flipped_indices = rng_instance.choice(
        n_samples, size=n_to_flip, replace=False
    )

    # Create a copy to avoid modifying the original array
    y_poisoned = y.copy()

    # Store original labels at flipped indices
    original_labels_at_flipped = y_poisoned[flipped_indices]

    # Flip binary labels: 0 -> 1, 1 -> 0
    y_poisoned[flipped_indices] = np.where(
        original_labels_at_flipped == 0, 1, 0
    )

    print(f"Flipping {n_to_flip} labels ({poison_percentage * 100:.1f}%).")

    return y_poisoned, flipped_indices


# ------------------------------------------------------------------------
# --- The rest is templated and you should not need to change anything ---
# ------------------------------------------------------------------------
poison_rate = 0.60
random_seed = 1337
y_train_poisoned, flipped_idx = flip_labels(y_train, poison_rate, random_seed)

print(f"Shape of poisoned labels: {y_train_poisoned.shape}")
print(f"Number of labels flipped: {len(flipped_idx)}")
print(f"Original labels at flipped indices (first 5): {y_train[flipped_idx[:5]]}")
print(
    f"Poisoned labels at flipped indices (first 5): {y_train_poisoned[flipped_idx[:5]]}"
)


model = LogisticRegression(random_state=random_seed)
model.fit(X_train, y_train_poisoned)
print("Model trained successfully on poisoned data.")
weights = model.coef_
intercept = model.intercept_
print(f"Extracted weights (shape): {weights.shape}")
print(f"Extracted intercept (shape): {intercept.shape}")

evaluator_base_url = "http://83.136.249.164:49434"
health_check_url = f"{evaluator_base_url}/health"
print(f"Checking evaluator health at: {health_check_url}")
if "83.136.249.164" not in evaluator_base_url:
    print("\n--- WARNING ---")
    print(
        "Please update the 'evaluator_base_url' variable with the correct IP and Port before running!"
    )
    print("-------------")
else:
    try:
        response = requests.get(health_check_url, timeout=10)
        response.raise_for_status()
        health_status = response.json()
        print("\n--- Health Check Response ---")
        print(f"Status: {health_status.get('status', 'N/A')}")
        print(f"Message: {health_status.get('message', 'No message received.')}")
        if health_status.get("status") != "healthy":
            print(
                "\nWarning: Evaluator service reported an unhealthy status. It might still be starting up or encountered an issue (like loading data)."
            )
    except requests.exceptions.ConnectionError as e:
        print(f"\nConnection Error: Could not connect to {health_check_url}.")
        print("Please check:")
        print("  1. The evaluator URL (IP address and port) is correct.")
        print("  2. The evaluator Docker container is running.")
        print(
            "  3. There are no network issues (firewalls, etc.) blocking the connection."
        )
    except requests.exceptions.Timeout:
        print(f"\nTimeout Error: The request to {health_check_url} timed out.")
        print(
            "The server might be taking too long to respond or there could be network issues."
        )
    except requests.exceptions.RequestException as e:
        print(f"\nError during health check request: {e}")
        print("Check the URL format and ensure the server is running.")
    except json.JSONDecodeError:
        print("\nError: Could not decode JSON response from health check.")
        print("The server might have sent an invalid response.")
        print(
            f"Raw response status: {response.status_code}, Raw response text: {response.text}"
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred during health check: {e}")
evaluator_url = f"{evaluator_base_url}/evaluate"
payload = {"weights": weights.tolist(), "intercept": intercept.tolist()}
print(f"Attempting submission to: {evaluator_url}")
if "83.136.249.164" not in evaluator_base_url:
    print("\n--- WARNING ---")
    print(
        "Please update the 'evaluator_base_url' variable with the correct IP address and Port before running this cell!"
    )
    print("-------------")
else:
    print(f"Payload: {json.dumps(payload)}")
    try:
        response = requests.post(evaluator_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        print("\n--- Evaluator Response ---")
        if result.get("success"):
            print("Attack Successful!")
            print(f"Accuracy evaluated by server: {result.get('accuracy'):.4f}")
            print(f"Flag: {result.get('flag')}")
        else:
            print("Evaluation Failed.")
            accuracy_val = result.get("accuracy")
            accuracy_str = f"{accuracy_val:.4f}" if accuracy_val is not None else "N/A"
            print(f"Accuracy evaluated by server: {accuracy_str}")
            print(f"Message: {result.get('message')}")
            print(
                "Hints: Did you poison exactly 60% of the data? Did you use the seed 1337 for flipping labels?"
            )
    except requests.exceptions.ConnectionError as e:
        print(
            f"\nConnection Error: Could not connect to the evaluator API at {evaluator_url}."
        )
        print("Please check:")
        print("  1. The evaluator URL (IP address and port) is correct.")
        print("  2. The evaluator Docker instance is spawned.")
        print(
            "  3. There are no network issues (firewalls, etc.) blocking the connection."
        )
    except requests.exceptions.Timeout:
        print(f"\nTimeout Error: The request to {evaluator_url} timed out.")
        print("The server might be slow, or there could be network issues.")
    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to evaluator API: {e}")
        print("Please check the evaluator URL and ensure the instance is spawned.")
    except json.JSONDecodeError:
        print("\nError decoding JSON response from the evaluator.")
        print("The server might have sent an invalid response.")
        print(
            f"Raw response status: {response.status_code}, Raw response text: {response.text}"
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
