# Evaluation of Trojan Attack on GTSRB Models

This section describes the **evaluation process for both clean and trojaned models**, focusing on **clean accuracy** and the **Attack Success Rate (ASR)** to quantify the effectiveness and stealth of a backdoor attack.

---

## 1. Evaluation Objectives

The evaluation consists of two primary goals:

1. **Clean Accuracy (CA)**:  
   Measure the model's performance on the clean test set (`testloader_clean`).  
   - Confirms whether the trojaned model maintains high performance on unmodified inputs.  
   - High CA demonstrates **stealth**, indicating the backdoor does not degrade normal functionality.

2. **Attack Success Rate (ASR)**:  
   Measure the rate at which triggered images are misclassified from the `SOURCE_CLASS` to the `TARGET_CLASS` using the triggered test set (`testloader_triggered`).  
   - High ASR for the trojaned model indicates **successful implantation of the backdoor**.  
   - Low ASR for the clean model ensures that normal models are not misbehaving under trigger conditions.

---

## 2. Evaluation Workflow

### a. Model Loading

Before evaluation, the code ensures models are available:

- **Clean model**: `clean_model_gtsrb`  
  - Loads from memory if available, otherwise from `gtsrb_cnn_clean.pth`.
- **Trojaned model**: `trojaned_model_gtsrb`  
  - Loads from memory if available, otherwise from `gtsrb_cnn_trojaned.pth`.

### b. DataLoader Checks

- `testloader_clean` must exist for clean accuracy evaluation.
- `testloader_triggered` must exist for ASR calculation.

### c. Clean Accuracy Evaluation

**Function:** `evaluate_model(model, testloader, criterion, device)`

- Computes average loss and accuracy on clean test data.
- For the trojaned model, this checks whether the attack affects **normal task performance**.

**Example Outputs:**

| Model | Accuracy | Average Loss |
|-------|---------|--------------|
| Clean | 97.92% | 0.0853 |
| Trojaned | 97.55% | 0.0903 |

*Interpretation:* Trojaned model retains nearly identical performance on clean data, indicating **stealthiness**.

---

### d. Attack Success Rate (ASR) Evaluation

**Function:** `calculate_asr_gtsrb(model, triggered_testloader, source_class, target_class, device)`

- Filters triggered images originating from `SOURCE_CLASS`.
- Counts how often the model predicts `TARGET_CLASS`.
- Computes ASR:

\[
\text{ASR (\%)} = 100 \times \frac{\text{Misclassified as Target}}{\text{Total Triggered Source-Class Images}}
\]

**Example Outputs:**

| Model | ASR | Details |
|-------|-----|---------|
| Clean | 0.0% | 0 / 270 triggered 'Stop' images misclassified |
| Trojaned | 100.0% | 270 / 270 triggered 'Stop' images misclassified |

*Interpretation:*  
- **Clean model:** No misclassification occurs under the trigger → confirms normal behavior.  
- **Trojaned model:** All triggered inputs are misclassified to the target class → confirms **successful backdoor activation**.

---

## 3. Summary of Evaluation Results

| Metric | Clean Model | Trojaned Model |
|--------|------------|----------------|
| Clean Accuracy (CA) | 97.92% | 97.55% |
| Attack Success Rate (ASR) | 0.0% | 100.0% |

**Key Takeaways:**

- **Stealth:** Trojaned model preserves clean accuracy close to baseline.
- **Effectiveness:** Trigger reliably induces misclassification from source to target class.
- **Backdoor Verification:** Clear distinction between clean and trojaned models under triggered inputs confirms the backdoor's presence and functionality.

---

## 4. Conclusion

The evaluation demonstrates that the Trojan attack is both **highly effective** and **stealthy**:

1. The trojaned model performs equivalently to the clean model on normal inputs.
2. The backdoor is reliably activated on triggered inputs, achieving **100% ASR**.
3. Clean model behavior is unaffected by the trigger, validating the specificity of the attack.

This methodology provides a **quantitative framework** for assessing backdoor attacks in machine learning models.
