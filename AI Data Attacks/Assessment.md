Below is a **clear, high‑level approach summary** tailored to this **label‑flipping data poisoning assessment**, focused on *strategy*, *reasoning*, and *execution flow* rather than implementation details.

---

# Approach Summary: Targeted Label Flipping for Ambiguous Class‑1 Classification

## 1. Problem Understanding

You are attacking a **4‑class One‑vs‑Rest (OvR) Logistic Regression classifier** by **poisoning its training labels only** (no feature manipulation).

### Objective

* **Degrade Class 1 performance specifically**
* Cause **true Class‑1 samples at inference time** to be **frequently misclassified as Class 0 or Class 2**
* Create *ambiguity* rather than forcing a single wrong class
* Leave other classes relatively unaffected

This is a **targeted, stealthy poisoning attack**, not a random or global degradation.

---

## 2. Threat Model & Constraints

| Aspect         | Constraint                 |
| -------------- | -------------------------- |
| Attack vector  | Label flipping only        |
| Feature values | Must remain unchanged      |
| Target         | Class 1 decision boundary  |
| Model          | OvR Logistic Regression    |
| Evaluation     | API‑based model scoring    |
| Goal type      | Targeted misclassification |

---

## 3. Key Insight: How OvR Logistic Regression Works

In OvR:

* Each class has its **own binary classifier**
* Class 1 is trained as:

  * Positive samples → Class 1
  * Negative samples → Classes 0, 2, 3

### Attack leverage:

If you **poison the labels of Class‑1 samples**, you:

* Corrupt the **positive examples** of the Class‑1 classifier
* Simultaneously **strengthen competing classifiers** (Class 0 / 2)
* Create **overlapping decision regions**

This naturally leads to **ambiguous predictions**.

---

## 4. Attack Strategy Overview

### Core Strategy

**Targeted label flipping of carefully chosen Class‑1 samples**, flipping them into:

* **Class 0**
* **Class 2**

Avoid flipping into Class 3 to:

* Keep ambiguity localized
* Prevent widespread instability

---

## 5. Sample Selection Strategy (Critical Step)

Instead of random flipping, focus on **high‑impact samples**.

### Recommended criteria:

* Select Class‑1 samples that are:

  * Close to decision boundaries
  * Low confidence / low margin
  * Near clusters of Class 0 or Class 2

These samples:

* Carry high influence on the learned boundary
* Cause maximum confusion with minimal poisoning rate

---

## 6. Label Flipping Design

### Flip Pattern

* Flip **Class 1 → Class 0** for ~50% of poisoned samples
* Flip **Class 1 → Class 2** for remaining ~50%

### Why this works:

* Prevents dominance of a single incorrect class
* Encourages split predictions
* Mimics natural class overlap rather than obvious corruption

---

## 7. Poisoning Rate Control

* Start with a **small poison percentage** (e.g., 5–10%)
* Increase gradually until:

  * Class‑1 recall drops significantly
  * Overall accuracy remains plausible
* Avoid excessive flipping to prevent detection

---

## 8. Model Retraining & Validation

After poisoning:

1. Retrain the OvR Logistic Regression model
2. Evaluate:

   * **Class‑1 recall**
   * **Confusion matrix**
   * Misclassification distribution of true Class‑1 samples

### Success indicators:

* Class‑1 samples predicted as both Class 0 and Class 2
* No single dominant wrong class
* Other classes remain stable

---

## 9. Submission Workflow

1. Apply poisoning inside the provided notebook
2. Train the poisoned model
3. Serialize and submit using the notebook’s API integration
4. Validate feedback from the evaluation endpoint

---

## 10. Why This Attack Is Effective

* Exploits **supervised learning trust assumptions**
* Uses **minimal, precise corruption**
* Preserves dataset realism
* Creates **semantic ambiguity**, not just accuracy loss
* Aligns with real‑world adversarial ML threats

---

## 11. Defensive Takeaway (Contextual Insight)

This assessment demonstrates why:

* Training data integrity is critical
* Class‑wise validation matters
* OvR classifiers are vulnerable to targeted poisoning
* Label audits and robust training are necessary defenses

---

### ✅ Final Outcome

A poisoned model that:

* Appears mostly functional
* Performs poorly **only on Class 1**
* Frequently misclassifies Class‑1 inputs as **Class 0 or Class 2**
* Successfully meets the assessment objective

---

```

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
import seaborn as sns  # For confusion matrix visualization
import os
import requests  # For API submission
import json  # For printing API response

htb_green = "#9fef00"
node_black = "#141d2b"
hacker_grey = "#a4b1cd"
white = "#ffffff"
azure = "#0086ff"  # Class 0
nugget_yellow = "#ffaf00"  # Class 1
malware_red = "#ff3e3e"  # Class 2
vivid_purple = "#9f00ff"  # Class 3
aquamarine = "#2ee7b6"  # Accent

# Plot Style Configuration
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update(
    {
        "figure.facecolor": node_black,
        "axes.facecolor": node_black,
        "axes.edgecolor": hacker_grey,
        "axes.labelcolor": white,
        "text.color": white,
        "xtick.color": hacker_grey,
        "ytick.color": hacker_grey,
        "grid.color": hacker_grey,
        "grid.alpha": 0.1,
        "legend.facecolor": node_black,
        "legend.edgecolor": hacker_grey,
        "legend.frameon": True,
        "legend.labelcolor": white,
        "figure.figsize": (10, 6),
    }
)

# Global seed for reproducibility
SEED = 1337
np.random.seed(SEED)
print(f"Global SEED set to: {SEED}")

# API Configuration
API_EVALUATOR_URL = "http://94.237.49.88:34945/evaluate_model"

print("--- Loading Dataset ---")
dataset_filename = "assessment_dataset.npz"

try:
    data = np.load(dataset_filename)
    X_train_orig = data["X_train"]
    y_train_orig = data["y_train"]  # Keep original labels safe
    X_test = data["X_test"]
    y_test = data["y_test"]
    data.close()
    print(f"Dataset '{dataset_filename}' loaded successfully.")
    print(f"  X_train shape: {X_train_orig.shape}, y_train shape: {y_train_orig.shape}")
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"  Unique labels in original training data: {np.unique(y_train_orig)}")
    print(f"  Original training label distribution: {np.bincount(y_train_orig)}")
except FileNotFoundError:
    print(
        f"ERROR: Dataset file '{dataset_filename}' not found. Please ensure it's in the correct location."
    )
    X_train_orig, y_train_orig, X_test, y_test = [
        None
    ] * 4  # Ensure variables exist but are None
except Exception as e:
    print(f"ERROR: Could not load dataset. {e}")
    X_train_orig, y_train_orig, X_test, y_test = [None] * 4

if X_train_orig is None:
    print("CRITICAL ERROR: Dataset not loaded. Cannot proceed with the assessment.")

print("\n--- Data Exploration ---")

# Define colors
class_colors_map_viz = {
    0: azure,  # Class 0
    1: nugget_yellow,  # Class 1
    2: malware_red,  # Class 2
    3: vivid_purple,  # Class 3
}


def plot_dataset_points(X, y, title="Dataset Visualization"):
    """Plots the 2D dataset with class-specific colors."""
    if X is None or y is None:
        print(f"Cannot plot: Data for '{title}' is missing.")
        return

    plt.figure(figsize=(12, 7))
    unique_labels = np.unique(y)

    for label_val in unique_labels:
        label_val = int(label_val)  # Ensure it's an int for dictionary key
        plt.scatter(
            X[y == label_val, 0],
            X[y == label_val, 1],
            color=class_colors_map_viz.get(label_val, hacker_grey),  # Fallback color
            label=f"Class {label_val}",
            edgecolors=node_black,
            s=50,
            alpha=0.7,
        )

    plt.title(title, fontsize=16, color=htb_green)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    if unique_labels.size > 0:  # Only show legend if there are labels
        plt.legend(title="Classes")
    plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)
    plt.show()


if X_train_orig is not None:
    plot_dataset_points(
        X_train_orig, y_train_orig, title="Original Training Data Distribution"
    )
else:
    print("Skipping data visualization as data was not loaded.")

# Train an OvR Logistic Regression classifier on the original training data.
print("\n--- Training Baseline Model ---")
baseline_model = None
baseline_accuracy = 0.0

if X_train_orig is not None:
    # Define the base estimator (Logistic Regression)
    #######
    ### Don't change these parameters! The API will expect them as is,
    #######
    base_estimator_config = {
        "random_state": SEED,
        "solver": "liblinear",
        "C": 1.0,
        "max_iter": 200,
    }

    baseline_logistic_estimator = LogisticRegression(**base_estimator_config)
    baseline_model = OneVsRestClassifier(baseline_logistic_estimator)

    print("Training baseline OvR Logistic Regression model...")
    baseline_model.fit(X_train_orig, y_train_orig)
    print("Baseline model trained successfully.")

    # Evaluate baseline model on the clean test set
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
    print(f"\nBaseline Model Performance on Clean Test Set:")
    print(f"  Overall Accuracy: {baseline_accuracy:.4f}")

    # Detailed report
    num_classes = len(np.unique(y_train_orig))
    target_names_report = [f"Class {i}" for i in range(num_classes)]
    print("\nClassification Report (Baseline Model):")
    print(
        classification_report(
            y_test, y_pred_baseline, target_names=target_names_report, zero_division=0
        )
    )

    # Confusion Matrix for Baseline
    cm_baseline = confusion_matrix(
        y_test, y_pred_baseline, labels=np.arange(num_classes)
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_baseline,
        annot=True,
        fmt="d",
        cmap=sns.dark_palette(azure, as_cmap=True),
        xticklabels=target_names_report,
        yticklabels=target_names_report,
        cbar_kws={"label": "Count"},
        linecolor=node_black,
        linewidths=0.5,
    )
    plt.title("Confusion Matrix (Baseline Model)", fontsize=14, color=htb_green)
    plt.ylabel("True Label", color=white)
    plt.xlabel("Predicted Label", color=white)
    plt.show()
else:
    print("Skipping baseline model training as data was not loaded.")
# Implement your attack strategy here.

y_train_poisoned = None  # Initialize

if y_train_orig is not None:
    y_train_poisoned = y_train_orig.copy()

    # -------------------------------
    # Targeted Label Flipping Attack
    # -------------------------------
    SEED = 1337
    np.random.seed(SEED)

    target_class = 1
    flip_to_classes = [0, 2]  # Force ambiguity between Class 0 and Class 2
    poison_fraction = 0.35    # 35% of Class 1 samples (tunable)

    # Find indices of Class 1 samples
    class1_indices = np.where(y_train_orig == target_class)[0]
    n_class1 = len(class1_indices)

    n_poison = int(poison_fraction * n_class1)
    poisoned_indices = np.random.choice(
        class1_indices, size=n_poison, replace=False
    )

    # Split poisoned samples between Class 0 and Class 2
    half = n_poison // 2
    to_class0 = poisoned_indices[:half]
    to_class2 = poisoned_indices[half:]

    y_train_poisoned[to_class0] = flip_to_classes[0]
    y_train_poisoned[to_class2] = flip_to_classes[1]

    print(f"[+] Targeted label flipping completed")
    print(f"    Total Class 1 samples      : {n_class1}")
    print(f"    Poisoned Class 1 samples   : {n_poison}")
    print(f"      → Flipped to Class 0     : {len(to_class0)}")
    print(f"      → Flipped to Class 2     : {len(to_class2)}")

    # Verify the changes
    print(f"\nLabel distribution in original y_train:      {np.bincount(y_train_orig)}")
    print(
        f"Label distribution in y_train_poisoned:  {np.bincount(y_train_poisoned)}"
    )

    plot_dataset_points(
        X_train_orig,
        y_train_poisoned,
        title="Poisoned Training Data Label Distribution (Targeting Class 1)",
    )

else:
    print("Skipping attack implementation as original training data was not loaded.")

print("\n--- Training Poisoned Model ---")
poisoned_model = None  # Initialize

if X_train_orig is not None and y_train_poisoned is not None:
    # Use the same configuration for the logistic regression estimator
    poisoned_logistic_estimator = LogisticRegression(**base_estimator_config)
    poisoned_model = OneVsRestClassifier(poisoned_logistic_estimator)

    print("Training poisoned OvR Logistic Regression model...")
    poisoned_model.fit(
        X_train_orig, y_train_poisoned
    )  # Use X_train_orig and y_train_poisoned
    print("Poisoned model trained successfully.")

    # Evaluate your poisoned model on the clean test set
    print("\nPoisoned Model Performance on Clean Test Set:")
    y_pred_poisoned = poisoned_model.predict(X_test)
    poisoned_accuracy = accuracy_score(y_test, y_pred_poisoned)
    print(f"  Overall Accuracy: {poisoned_accuracy:.4f}")
    if baseline_model is not None:  # Check if baseline_accuracy was computed
        print(f"  (Baseline Accuracy was: {baseline_accuracy:.4f})")

    print("\nClassification Report (Poisoned Model):")
    print(
        classification_report(
            y_test, y_pred_poisoned, target_names=target_names_report, zero_division=0
        )
    )

    # Confusion Matrix for Poisoned Model
    cm_poisoned = confusion_matrix(
        y_test, y_pred_poisoned, labels=np.arange(num_classes)
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_poisoned,
        annot=True,
        fmt="d",
        cmap=sns.dark_palette(malware_red, as_cmap=True),
        xticklabels=target_names_report,
        yticklabels=target_names_report,
        cbar_kws={"label": "Count"},
        linecolor=node_black,
        linewidths=0.5,
    )
    plt.title("Confusion Matrix (Poisoned Model)", fontsize=14, color=htb_green)
    plt.ylabel("True Label", color=white)
    plt.xlabel("Predicted Label", color=white)
    plt.show()

    # Detailed check for Class 1 misclassification (as per assessment objective)
    class1_actual_indices_test = np.where(y_test == 1)[0]
    if len(class1_actual_indices_test) > 0:
        class1_predictions_poisoned_model = y_pred_poisoned[class1_actual_indices_test]

        misclassified_as_0 = np.sum(class1_predictions_poisoned_model == 0)
        misclassified_as_2 = np.sum(class1_predictions_poisoned_model == 2)
        correctly_as_1 = np.sum(class1_predictions_poisoned_model == 1)
        misclassified_as_3 = np.sum(class1_predictions_poisoned_model == 3)
        total_class1_test = len(class1_actual_indices_test)

        print(
            f"\nAnalysis of Class 1 predictions by Poisoned Model (on local test set):"
        )
        print(f"  Total Class 1 test samples: {total_class1_test}")
        print(
            f"  Predicted as Class 0: {misclassified_as_0} ({misclassified_as_0 / total_class1_test * 100:.2f}%)"
        )
        print(
            f"  Predicted as Class 1 (Correct): {correctly_as_1} ({correctly_as_1 / total_class1_test * 100:.2f}%)"
        )
        print(
            f"  Predicted as Class 2: {misclassified_as_2} ({misclassified_as_2 / total_class1_test * 100:.2f}%)"
        )
        print(
            f"  Predicted as Class 3: {misclassified_as_3} ({misclassified_as_3 / total_class1_test * 100:.2f}%)"
        )

        # Check Class 3 Recall
        class3_recall_poisoned = recall_score(
            y_test, y_pred_poisoned, labels=[3], average="macro", zero_division=0
        )
        print(f"  Recall for Class 3 (Poisoned Model): {class3_recall_poisoned:.4f}")
    else:
        print("No Class 1 samples in the local test set to analyze for ambiguity.")
else:
    print(
        "Skipping poisoned model training as data was not loaded or attack not implemented."
    )
print("\n--- Training Poisoned Model ---")
poisoned_model = None  # Initialize

if X_train_orig is not None and y_train_poisoned is not None:
    # Use the same configuration for the logistic regression estimator
    poisoned_logistic_estimator = LogisticRegression(**base_estimator_config)
    poisoned_model = OneVsRestClassifier(poisoned_logistic_estimator)

    print("Training poisoned OvR Logistic Regression model...")
    poisoned_model.fit(
        X_train_orig, y_train_poisoned
    )  # Use X_train_orig and y_train_poisoned
    print("Poisoned model trained successfully.")

    # Evaluate your poisoned model on the clean test set
    print("\nPoisoned Model Performance on Clean Test Set:")
    y_pred_poisoned = poisoned_model.predict(X_test)
    poisoned_accuracy = accuracy_score(y_test, y_pred_poisoned)
    print(f"  Overall Accuracy: {poisoned_accuracy:.4f}")
    if baseline_model is not None:  # Check if baseline_accuracy was computed
        print(f"  (Baseline Accuracy was: {baseline_accuracy:.4f})")

    print("\nClassification Report (Poisoned Model):")
    print(
        classification_report(
            y_test, y_pred_poisoned, target_names=target_names_report, zero_division=0
        )
    )

    # Confusion Matrix for Poisoned Model
    cm_poisoned = confusion_matrix(
        y_test, y_pred_poisoned, labels=np.arange(num_classes)
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_poisoned,
        annot=True,
        fmt="d",
        cmap=sns.dark_palette(malware_red, as_cmap=True),
        xticklabels=target_names_report,
        yticklabels=target_names_report,
        cbar_kws={"label": "Count"},
        linecolor=node_black,
        linewidths=0.5,
    )
    plt.title("Confusion Matrix (Poisoned Model)", fontsize=14, color=htb_green)
    plt.ylabel("True Label", color=white)
    plt.xlabel("Predicted Label", color=white)
    plt.show()

    # Detailed check for Class 1 misclassification (as per assessment objective)
    class1_actual_indices_test = np.where(y_test == 1)[0]
    if len(class1_actual_indices_test) > 0:
        class1_predictions_poisoned_model = y_pred_poisoned[class1_actual_indices_test]

        misclassified_as_0 = np.sum(class1_predictions_poisoned_model == 0)
        misclassified_as_2 = np.sum(class1_predictions_poisoned_model == 2)
        correctly_as_1 = np.sum(class1_predictions_poisoned_model == 1)
        misclassified_as_3 = np.sum(class1_predictions_poisoned_model == 3)
        total_class1_test = len(class1_actual_indices_test)

        print(
            f"\nAnalysis of Class 1 predictions by Poisoned Model (on local test set):"
        )
        print(f"  Total Class 1 test samples: {total_class1_test}")
        print(
            f"  Predicted as Class 0: {misclassified_as_0} ({misclassified_as_0 / total_class1_test * 100:.2f}%)"
        )
        print(
            f"  Predicted as Class 1 (Correct): {correctly_as_1} ({correctly_as_1 / total_class1_test * 100:.2f}%)"
        )
        print(
            f"  Predicted as Class 2: {misclassified_as_2} ({misclassified_as_2 / total_class1_test * 100:.2f}%)"
        )
        print(
            f"  Predicted as Class 3: {misclassified_as_3} ({misclassified_as_3 / total_class1_test * 100:.2f}%)"
        )

        # Check Class 3 Recall
        class3_recall_poisoned = recall_score(
            y_test, y_pred_poisoned, labels=[3], average="macro", zero_division=0
        )
        print(f"  Recall for Class 3 (Poisoned Model): {class3_recall_poisoned:.4f}")
    else:
        print("No Class 1 samples in the local test set to analyze for ambiguity.")
else:
    print(
        "Skipping poisoned model training as data was not loaded or attack not implemented."
    )

print("\n--- Saving Poisoned Model Parameters ---")
output_model_filename = "poisoned_model_params.npz"

if poisoned_model is not None:
    try:
        params_to_save = {}
        # Ensure the model was fitted and has estimators
        if not hasattr(poisoned_model, "estimators_"):
            raise AttributeError(
                "Poisoned model does not have 'estimators_' attribute. Was it trained?"
            )

        for i, estimator in enumerate(poisoned_model.estimators_):
            if not hasattr(estimator, "coef_") or not hasattr(estimator, "intercept_"):
                raise AttributeError(
                    f"Estimator {i} is not fitted or does not have coef_/intercept_."
                )
            params_to_save[f"coef_estimator_{i}"] = estimator.coef_
            params_to_save[f"intercept_estimator_{i}"] = estimator.intercept_

        if not hasattr(poisoned_model, "classes_"):
            raise AttributeError("Poisoned model does not have 'classes_' attribute.")
        params_to_save["classes_"] = poisoned_model.classes_

        np.savez_compressed(output_model_filename, **params_to_save)
        print(f"Poisoned model parameters saved to '{output_model_filename}'.")
        print("This is the file you should submit to the evaluation API.")
    except AttributeError as ae:
        print(
            f"Error saving parameters: Model or its estimators might not be fully trained or accessible. Details: {ae}"
        )
    except Exception as e:
        print(
            f"An unexpected error occurred while saving poisoned model parameters: {e}"
        )
else:
    print("Poisoned model not available. Skipping model parameter saving.")
print("\n--- Submitting Model to API for Evaluation ---")

if os.path.exists(output_model_filename):
    print(f"Found '{output_model_filename}' for submission.")
    try:
        with open(output_model_filename, "rb") as f:
            files = {
                "model_params": (output_model_filename, f, "application/octet-stream")
            }

            print(f"Submitting to: {API_EVALUATOR_URL}")
            response = requests.post(
                API_EVALUATOR_URL, files=files, timeout=30
            )  # Added timeout

            print("\n--- API Response ---")
            print(f"Status Code: {response.status_code}")
            try:
                response_json = response.json()
                print(json.dumps(response_json, indent=2))  # Pretty print JSON
                if response_json.get("attack_successful", False):
                    print(
                        f"\nSUCCESS! Flag: {response_json.get('flag', 'No flag in response.')}"
                    )
                else:
                    print(
                        "\nAttack did not meet all criteria or an error occurred. Check API response details."
                    )
            except requests.exceptions.JSONDecodeError:
                print("Response from API was not in JSON format:")
                print(response.text)

    except requests.exceptions.ConnectionError:
        print(f"CONNECTION ERROR: Could not connect to the API at {API_EVALUATOR_URL}.")
        print("Please ensure your spawned instance is is running and accessible.")
    except FileNotFoundError:
        print(
            f"ERROR: File '{output_model_filename}' not found for submission, even though it should exist."
        )
    except Exception as e:
        print(f"An unexpected error occurred during API submission: {e}")
        # traceback.print_exc() # For more detailed debugging if needed
else:
    print(f"Model parameter file '{output_model_filename}' not found. Cannot submit.")
```


## Output

```
┌─[au-dedicated-47-dhcp]─[10.10.14.2]─[kamalares@htb-cxg5j44qq7-htb-cloud-com]─[~/Documents]
└──╼ [★]$ /bin/python /home/kamalares/Documents/main.py
Global SEED set to: 1337
--- Loading Dataset ---
Dataset 'assessment_dataset.npz' loaded successfully.
  X_train shape: (1500, 2), y_train shape: (1500,)
  X_test shape: (500, 2), y_test shape: (500,)
  Unique labels in original training data: [0 1 2 3]
  Original training label distribution: [375 375 375 375]

--- Data Exploration ---

--- Training Baseline Model ---
Training baseline OvR Logistic Regression model...
Baseline model trained successfully.

Baseline Model Performance on Clean Test Set:
  Overall Accuracy: 0.9260

Classification Report (Baseline Model):
              precision    recall  f1-score   support

     Class 0       0.92      0.99      0.95       125
     Class 1       0.96      0.74      0.83       125
     Class 2       0.90      0.98      0.94       125
     Class 3       0.93      0.99      0.96       125

    accuracy                           0.93       500
   macro avg       0.93      0.93      0.92       500
weighted avg       0.93      0.93      0.92       500

[+] Targeted label flipping completed
    Total Class 1 samples      : 375
    Poisoned Class 1 samples   : 131
      → Flipped to Class 0     : 65
      → Flipped to Class 2     : 66

Label distribution in original y_train:      [375 375 375 375]
Label distribution in y_train_poisoned:  [440 244 441 375]

--- Training Poisoned Model ---
Training poisoned OvR Logistic Regression model...
Poisoned model trained successfully.

Poisoned Model Performance on Clean Test Set:
  Overall Accuracy: 0.7940
  (Baseline Accuracy was: 0.9260)

Classification Report (Poisoned Model):
              precision    recall  f1-score   support

     Class 0       0.75      1.00      0.86       125
     Class 1       1.00      0.19      0.32       125
     Class 2       0.71      0.99      0.83       125
     Class 3       0.93      0.99      0.96       125

    accuracy                           0.79       500
   macro avg       0.85      0.79      0.74       500
weighted avg       0.85      0.79      0.74       500


Analysis of Class 1 predictions by Poisoned Model (on local test set):
  Total Class 1 test samples: 125
  Predicted as Class 0: 40 (32.00%)
  Predicted as Class 1 (Correct): 24 (19.20%)
  Predicted as Class 2: 51 (40.80%)
  Predicted as Class 3: 10 (8.00%)
  Recall for Class 3 (Poisoned Model): 0.9920

--- Training Poisoned Model ---
Training poisoned OvR Logistic Regression model...
Poisoned model trained successfully.

Poisoned Model Performance on Clean Test Set:
  Overall Accuracy: 0.7940
  (Baseline Accuracy was: 0.9260)

Classification Report (Poisoned Model):
              precision    recall  f1-score   support

     Class 0       0.75      1.00      0.86       125
     Class 1       1.00      0.19      0.32       125
     Class 2       0.71      0.99      0.83       125
     Class 3       0.93      0.99      0.96       125

    accuracy                           0.79       500
   macro avg       0.85      0.79      0.74       500
weighted avg       0.85      0.79      0.74       500


Analysis of Class 1 predictions by Poisoned Model (on local test set):
  Total Class 1 test samples: 125
  Predicted as Class 0: 40 (32.00%)
  Predicted as Class 1 (Correct): 24 (19.20%)
  Predicted as Class 2: 51 (40.80%)
  Predicted as Class 3: 10 (8.00%)
  Recall for Class 3 (Poisoned Model): 0.9920

--- Saving Poisoned Model Parameters ---
Poisoned model parameters saved to 'poisoned_model_params.npz'.
This is the file you should submit to the evaluation API.

--- Submitting Model to API for Evaluation ---
Found 'poisoned_model_params.npz' for submission.
Submitting to: http://94.237.49.88:34945/evaluate_model

--- API Response ---
Status Code: 200
{
  "attack_successful": true,
  "criteria_met": {
    "class1_to_class0_ambiguity": true,
    "class1_to_class2_ambiguity": true,
    "class3_recall_maintained": true
  },
  "evaluation_status": "complete",
  "flag": "HTB{Redacted}",
  "message": "Attack successful! Ambiguity achieved and constraints met.",
  "metrics": {
    "class1_correctly_as_1": 24,
    "class1_misclassified_as_0": 40,
    "class1_misclassified_as_0_percent": 0.32,
    "class1_misclassified_as_2": 51,
    "class1_misclassified_as_2_percent": 0.408,
    "class1_misclassified_as_3": 10,
    "class1_total_samples": 125,
    "class3_precision": 0.9253731343283582,
    "class3_recall": 0.992,
    "overall_accuracy": 0.794
  }
}

SUCCESS! Flag: HTB{Redacted}

```
