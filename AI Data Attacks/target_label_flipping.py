import numpy as np
import json
import requests
from sklearn.linear_model import LogisticRegression
import os

dataset_filename = "label_flipping_dataset.npz"
random_seed = 1337  # Seed for reproducibility in attack & model training
np.random.seed(random_seed)  # Apply seed globally if needed, or pass to functions

# >>> IMPORTANT: SET THIS VARIABLE TO YOUR SPAWNED INSTANCE IP AND PORT<<<
evaluator_base_url = "http://94.237.61.202:43447"  # CHANGE THIS
# Example: evaluator_base_url = "http://10.10.10.1:5555"

# Attack Configuration
TARGET_CLASS_TO_POISON = 0  # We want to make the model bad at identifying Class 0
NEW_LABEL_FOR_POISONED = 1  # We want it to predict Class 1 instead
POISON_FRACTION = 0.60

# Load Data
print(f"Loading data from: {dataset_filename}")
try:
    data = np.load(dataset_filename)
    X_train = data["Xtr"]
    y_train = data["ytr"]
    X_test = data["Xte"]
    y_test = data["yte"]
    print("Data loaded successfully.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    unique_classes_train = np.unique(y_train)
    print(f"Unique classes in training data: {unique_classes_train}")
    if (
        TARGET_CLASS_TO_POISON not in unique_classes_train
        or NEW_LABEL_FOR_POISONED not in unique_classes_train
    ):
        print("Warning: Target or new label class not found in training data.")
    data.close()
except FileNotFoundError:
    print(f"Error: Dataset file '{dataset_filename}' not found.")
    raise
except KeyError as e:
    print(f"Error: Could not find expected array key '{e}' in the .npz file.")
    raise
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    raise
    
def targeted_class_label_flip(y, poison_percentage, target_class, new_class, seed=1337):

    if not 0 <= poison_percentage <= 1:
        raise ValueError("poison_percentage must be between 0 and 1.")
    if target_class == new_class:
        raise ValueError("target_class and new_class cannot be the same.")
    # Ensure target_class and new_class are present in y
    unique_labels = np.unique(y)
    if target_class not in unique_labels:
         raise ValueError(f"target_class ({target_class}) does not exist in y.")
    if new_class not in unique_labels:
         raise ValueError(f"new_class ({new_class}) does not exist in y.")

    # Identify indices belonging to the target class
    target_indices = np.where(y == target_class)[0]
    n_target_samples = len(target_indices)

    if n_target_samples == 0:
        print(f"Warning: No samples found for target_class {target_class}. No labels flipped.")
        return y.copy(), np.array([], dtype=int)
        
    # Calculate the number of labels to flip within the target class
    n_to_flip = int(n_target_samples * poison_percentage)

    if n_to_flip == 0:
        print(f"Warning: Poison percentage ({poison_percentage * 100:.1f}%) is too low "
              f"to flip any labels in the target class (size {n_target_samples}).")
        return y.copy(), np.array([], dtype=int)
        
        
    # Use a dedicated random number generator instance with the specified seed
    rng_instance = np.random.default_rng(seed)

    # Randomly select indices from the target_indices subset to flip
    # These are indices relative to the target_indices array
    indices_within_target_set_to_flip = rng_instance.choice(
        n_target_samples, size=n_to_flip, replace=False
    )
    # Map these back to the original array indices
    flipped_indices = target_indices[indices_within_target_set_to_flip]
    
    
    # Create a copy to avoid modifying the original array
    y_poisoned = y.copy()

    # Perform the flip for the selected indices to the new class label
    y_poisoned[flipped_indices] = new_class
    
    
    print(f"Targeting Class {target_class} for flipping to Class {new_class}.")
    print(f"Identified {n_target_samples} samples of Class {target_class}.")
    print(f"Attempting to flip {poison_percentage * 100:.1f}% ({n_to_flip} samples) of these.")
    print(f"Successfully flipped {len(flipped_indices)} labels.")
    
    return y_poisoned, flipped_indices
    
# Execute the attack
y_train_poisoned, flipped_idx = targeted_class_label_flip(
    y_train,
    poison_percentage=POISON_FRACTION,
    target_class=TARGET_CLASS_TO_POISON,
    new_class=NEW_LABEL_FOR_POISONED,
    seed=random_seed
)

# Basic Checks
print("\n--- Post-Attack Checks ---")
if flipped_idx.size > 0:
    print(f"Attack function executed, {len(flipped_idx)} label(s) flipped.")
    print(f"Indices of flipped labels in training data (first 10): {flipped_idx[:10]}")
    print(f"Original labels at flipped indices (first 10): {y_train[flipped_idx[:10]]}")
    print(
        f"Poisoned labels at flipped indices (first 10): {y_train_poisoned[flipped_idx[:10]]}"
    )
    print(f"Shape of poisoned labels array: {y_train_poisoned.shape}")
else:
    print(
        "Attack function ran, but no labels were flipped (check settings and warnings)."
    )
    print("Proceeding with potentially unpoisoned labels.")

# %%
# Train Model using Logistic Regression (Same as before)
print("\n--- Training Model on Poisoned Labels ---")
model = LogisticRegression(random_state=random_seed, solver="liblinear")

try:
    # Train on original features but poisoned labels
    model.fit(X_train, y_train_poisoned)
    print("Logistic Regression model trained successfully.")
except Exception as e:
    print(f"Error during model training: {e}")
    raise
    
print("\n--- Extracting Model Parameters ---")
try:
    weights = model.coef_
    intercept = model.intercept_
    print(f"Extracted weights shape: {weights.shape}")
    print(f"Extracted intercept shape: {intercept.shape}")
    weights_list = weights.tolist()
    intercept_list = intercept.tolist()
    parameters_extracted = True
except Exception as e:
    print(f"An unexpected error occurred during parameter extraction: {e}")
    weights_list = None
    intercept_list = None
    parameters_extracted = False
    
    
health_check_url = f"{evaluator_base_url}/health"
print(f"Checking evaluator health at: {health_check_url}")
if "94.237.61.202" not in evaluator_base_url:
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
        
        
        
evaluator_url = f"{evaluator_base_url}/evaluate_targeted"
print(f"\nAttempting submission to: {evaluator_url}")

if not parameters_extracted:
    print("Error: Cannot submit - parameters not extracted.")
elif "94.237.61.202" not in evaluator_base_url or "43447" not in evaluator_base_url:
    print("\n--- WARNING: Update evaluator_base_url ---")
else:
    payload = {"coef": weights_list, "intercept": intercept_list}
    print(f"Payload preview: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(evaluator_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        print("\n--- Evaluator Response ---")
        if result.get("success"):
            print(f"{'=' * 10} Attack Successful! {'=' * 10}")
            oa_str = (
                f"{result.get('overall_accuracy', 'N/A'):.4f}"
                if isinstance(result.get("overall_accuracy"), (int, float))
                else "N/A"
            )
            c0a_str = (
                f"{result.get('class0_accuracy', 'N/A'):.4f}"
                if isinstance(result.get("class0_accuracy"), (int, float))
                else "N/A"
            )  # Get Class 0 Accuracy
            print(f"Overall Accuracy evaluated: {oa_str}")
            print(f"Accuracy on Class 0 samples: {c0a_str}")  # Display Class 0 Accuracy
            print(f"Message: {result.get('message', 'N/A')}")
            print(f"\nFLAG: {result.get('flag')}")
            print(f"{'=' * 38}")
        else:
            print("Evaluation Failed.")
            oa_val = result.get("overall_accuracy")
            c0a_val = result.get("class0_accuracy")  # Get Class 0 Accuracy
            oa_str = f"{oa_val:.4f}" if oa_val is not None else "N/A"
            c0a_str = (
                f"{c0a_val:.4f}" if c0a_val is not None else "N/A"
            )  # Get Class 0 Accuracy

            print(f"Overall Accuracy evaluated: {oa_str}")
            print(f"Accuracy on Class 0 samples: {c0a_str}")  # Display Class 0 Accuracy
            print(f"Message: {result.get('message', 'No message provided.')}")
            print(
                "\nHints: Did the attack significantly reduce accuracy specifically for Class 0 samples?"
            )
            print("Did the overall accuracy remain above the required threshold?")
            print("Consider adjusting the POISON_FRACTION.")

    except requests.exceptions.ConnectionError:
        print(f"\nConnection Error: Could not connect to {evaluator_url}.")
    except requests.exceptions.Timeout:
        print(f"\nTimeout Error: Request to {evaluator_url} timed out.")
    except requests.exceptions.RequestException as e:
        print(f"\nError during submission request: {e}")
        if e.response is not None:
            print(f"Server Response Status Code: {e.response.status_code}")
            try:
                print(f"Server Response Body: {e.response.json()}")
            except json.JSONDecodeError:
                print(f"Server Response Body (non-JSON): {e.response.text}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during submission: {e}")
