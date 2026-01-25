````markdown
# Training and Evaluation of Clean and Trojaned GTSRB Models

This section describes the **training workflow for both clean and trojaned models**, including hyperparameter selection, training loops, evaluation, and measuring the effectiveness of a backdoor attack.

---

## 1. Training Hyperparameters

Key parameters controlling the training process:

| Parameter | Value | Purpose / Notes |
|-----------|-------|----------------|
| `LEARNING_RATE` | 0.001 | Step size for weight updates. Too high → instability, too low → slow learning. Must balance learning the main task and trigger association. |
| `NUM_EPOCHS` | 20 | Number of passes over the entire dataset. More epochs allow the model to learn the trigger but may overfit and reduce clean accuracy. |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization to prevent overfitting. Strong decay improves generalization but may suppress weights needed for embedding the trigger. |

---

## 2. Training Loop: `train_model`

The `train_model` function orchestrates **model training** over multiple epochs with a PyTorch DataLoader.

**Workflow:**

1. Set model to **training mode**.
2. Iterate through **epochs**:
   - Initialize running loss and valid sample counter.
   - Iterate through **batches**:
     - Filter out invalid samples (labels = -1).
     - Move inputs and labels to device (CPU/GPU).
     - Zero gradients.
     - Forward pass → compute predictions.
     - Compute loss using criterion (e.g., CrossEntropyLoss).
     - Backward pass → compute gradients.
     - Optimizer step → update weights.
     - Accumulate batch loss.
3. Compute and store **average epoch loss**.
4. Return list of **epoch-wise average losses**.

**Code Example:**

```python
epoch_losses = train_model(model, trainloader, criterion, optimizer, NUM_EPOCHS, device)
````

**Key Points:**

* Supports poisoned datasets (labels modified for backdoor).
* Skips batches with only invalid samples.
* Tracks batch-level progress with `tqdm`.

---

## 3. Model Evaluation: `evaluate_model`

The `evaluate_model` function assesses **model performance** on clean or test datasets.

**Workflow:**

1. Set model to **evaluation mode** (disable dropout/batch norm updates).
2. Iterate over DataLoader without computing gradients.
3. Filter invalid samples.
4. Forward pass to get predictions.
5. Compute loss and compare predictions to true labels.
6. Accumulate **accuracy and average loss**.
7. Return:

```python
accuracy, avg_loss, np.array(predictions), np.array(true_labels)
```

**Notes:**

* Allows detailed analysis (confusion matrix, metrics).
* Skips batches with all invalid samples.

---

## 4. Measuring Trojan Effectiveness: `calculate_asr_gtsrb`

The `calculate_asr_gtsrb` function computes **Attack Success Rate (ASR)**:

* Evaluates the trojaned model on triggered test images.
* Focuses on samples originally from `SOURCE_CLASS`.
* ASR = % of triggered source-class images misclassified as `TARGET_CLASS`.

**Workflow:**

1. Set model to **evaluation mode**.
2. Iterate over `triggered_testloader` (all images contain the trigger):

   * Filter invalid samples.
   * Select images whose original label = `SOURCE_CLASS`.
   * Forward pass → compute predictions.
   * Count predictions that match `TARGET_CLASS`.
3. Compute ASR:

```python
ASR (%) = 100 * (misclassified_as_target / total_source_class_triggered)
```

---

## 5. Training Models

Two models are trained for comparison:

### a. Clean Model

* **Data**: `trainloader_clean` (no poisoned images).
* **Model**: `GTSRB_CNN`.
* **Loss**: `nn.CrossEntropyLoss`.
* **Optimizer**: Adam with specified `LEARNING_RATE` and `WEIGHT_DECAY`.
* **Training**:

```python
clean_losses_gtsrb = train_model(clean_model_gtsrb, trainloader_clean, criterion_gtsrb, optimizer_clean_gtsrb, NUM_EPOCHS, device)
torch.save(clean_model_gtsrb.state_dict(), "gtsrb_cnn_clean.pth")
```

**Purpose:** Serves as a **baseline** to evaluate clean accuracy (CA) and to contrast against the trojaned model.

---

### b. Trojaned Model

* **Data**: `trainloader_poisoned` (contains trigger-injected images with target labels).
* **Model**: New instance of `GTSRB_CNN`.
* **Loss**: Same as clean model.
* **Optimizer**: Adam.
* **Training**:

```python
trojaned_losses_gtsrb = train_model(trojaned_model_gtsrb, trainloader_poisoned, criterion_gtsrb, optimizer_trojan_gtsrb, NUM_EPOCHS, device)
torch.save(trojaned_model_gtsrb.state_dict(), "gtsrb_cnn_trojaned.pth")
```

**Purpose:** Learns both the **primary task** and the **trigger-to-target association**, enabling backdoor behavior.

---

## 6. Summary of Training Workflow

| Model    | DataLoader             | Goal                                    | Save Path                |
| -------- | ---------------------- | --------------------------------------- | ------------------------ |
| Clean    | `trainloader_clean`    | Train baseline model for clean accuracy | `gtsrb_cnn_clean.pth`    |
| Trojaned | `trainloader_poisoned` | Train model with backdoor               | `gtsrb_cnn_trojaned.pth` |

**Key Notes:**

* Clean training ensures a reference model for CA.
* Trojaned training leverages poisoned data to embed the backdoor.
* Evaluation on triggered test set allows **ASR calculation**, quantifying the success of the backdoor attack.
* Hyperparameters (LR, epochs, weight decay) balance between **clean accuracy** and **attack effectiveness**.

```
```
