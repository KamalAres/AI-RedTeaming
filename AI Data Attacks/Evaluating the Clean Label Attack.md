# Evaluating the Clean Label Attack

## Overview

The final phase of the Clean Label Attack focuses on **validating whether the poisoned training data achieved its intended objective**. Specifically, the evaluation answers two key questions:

1. **Did the attack succeed in misclassifying the chosen target point?**
2. **Did the attack remain stealthy by preserving the model’s overall performance on clean data?**

To ensure that any observed changes are attributable solely to the poisoned data, the same model architecture and hyperparameters used in the baseline experiment are reused without modification.

---

## Training the Poisoned Model

A new classifier is trained using the **poisoned training dataset** \((X_{train\_poisoned}, y_{train\_poisoned})\).

- **Model architecture:** One-vs-Rest (OvR) classifier
- **Base estimator:** Logistic Regression
- **Hyperparameters:** Identical to the baseline configuration

This controlled setup ensures a fair comparison between the baseline and poisoned models, isolating the effect of the clean label attack.

Once training completes, the resulting model reflects the influence of the subtly perturbed Class 0 samples introduced during the poisoning phase.

---

## Targeted Attack Evaluation

### Target Point Assessment

The primary goal of the attack was to force the model to misclassify a specific target point:

- **Target index:** 373  
- **True label:** Class 1  
- **Baseline prediction:** Class 1  

After retraining on the poisoned dataset, the poisoned model’s prediction for the same target point is evaluated.

### Outcome

- The poisoned model predicts **Class 0** for the target point.
- This represents a **successful targeted misclassification**, as the prediction is both incorrect and aligned with the attacker’s intended class.

Importantly, this result is achieved **without modifying any labels** in the training data, preserving the defining characteristic of a clean label attack.

---

## Impact on Overall Model Performance

### Evaluation on Clean Test Data

To assess collateral effects, the poisoned model is evaluated on the original, untouched test dataset.

- **Baseline accuracy:** 0.9600  
- **Poisoned model accuracy:** 0.9578  
- **Accuracy drop:** 0.0022  

The reduction in accuracy is minimal, indicating that the attack did not significantly degrade overall model performance.

### Classification Behavior

The classification report shows:
- High precision and recall across all three classes.
- Only marginal changes relative to the baseline.
- No obvious performance anomalies that would easily signal an attack.

This small performance impact is typical of clean label attacks, which aim to remain **highly targeted and low-noise**.

---

## Interpreting the Results

The evaluation demonstrates two critical properties of a successful clean label attack:

1. **Effectiveness**  
   The target point is misclassified exactly as intended, confirming that the decision boundary has shifted locally.

2. **Stealth**  
   The model’s global behavior remains largely unchanged, with only a negligible accuracy drop and no dramatic degradation across classes.

This balance between precision and subtlety makes clean label attacks particularly dangerous in real-world systems.

---

## Decision Boundary Analysis

### Visual Evidence of Boundary Shift

A final visualization compares the poisoned model’s decision boundaries against the baseline:

- The **target point**, originally inside the Class 1 region, now lies clearly within the **Class 0 region**.
- The **perturbed neighbors**, still labeled as Class 0, appear embedded in the Class 1 feature space.
- The decision boundary bends outward to accommodate these inconsistencies, sweeping the target point along with it.

This visual confirmation reinforces the numerical results and clearly illustrates how **localized feature perturbations can reshape global decision behavior**.

---

## Key Takeaways

- The attack successfully induced a **targeted misclassification** without label manipulation.
- Overall model accuracy remained high, preserving the illusion of normal operation.
- The decision boundary was altered in a subtle, localized manner driven by a small number of carefully perturbed points.
- Clean label attacks are **harder to detect than label flipping**, as they do not introduce obvious inconsistencies in labels or gross performance degradation.
- However, they require **significant attacker knowledge and precision**, making them more complex to execute.

---

## Conclusion

This evaluation confirms the effectiveness and stealth of the clean label attack. By modifying only the features of a handful of training points and keeping their labels intact, the attacker was able to manipulate the model’s learned decision boundary in a targeted way. The result is a system that appears healthy under standard evaluation metrics, yet behaves maliciously for specific, carefully chosen inputs—highlighting the serious security implications of data-centric attacks on machine learning systems.
