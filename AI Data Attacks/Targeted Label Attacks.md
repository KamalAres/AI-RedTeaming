# Targeted Label Attacks

## Overview

A **Targeted Label Attack** is a more precise and intentional form of data poisoning compared to random label flipping. While standard label flipping aims to **degrade overall model performance**, a targeted attack seeks to **systematically manipulate the model’s behavior for a specific class or subset of inputs**. The objective is not broad inaccuracy, but *predictable misclassification*.

In this scenario, the attack is designed to make a sentiment analysis model **misclassify genuinely positive reviews (Class 1) as negative (Class 0)**. This reflects realistic adversarial goals, such as suppressing positive feedback for a product or service.

---

## Conceptual Difference from Random Label Flipping

| Random Label Flipping | Targeted Label Attack |
|-----------------------|----------------------|
| Labels flipped uniformly across classes | Labels flipped only for a chosen target class |
| Degrades overall accuracy | Induces class-specific bias |
| Effects may be unpredictable | Effects are intentional and directional |
| Easier to detect statistically | Harder to detect via global metrics |

The targeted approach focuses the poisoning effort where it will be most damaging, producing **asymmetric errors** that may go unnoticed if only aggregate performance metrics are monitored.

---

## Attack Objective

The goal of the targeted attack is to **shift the model’s decision boundary** such that:
- Feature regions genuinely associated with **Class 1** are increasingly classified as **Class 0**
- The model becomes systematically biased against the target class
- Misclassification occurs even when the input features strongly indicate the original class

---

## Theoretical Impact on Model Training

### Logistic Regression Loss Function

The Logistic Regression model minimizes the average binary cross-entropy loss:

\[
L(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N}
\left[ y_i \log(p_i) + (1 - y_i)\log(1 - p_i) \right]
\]

where:
- \( y_i \in \{0,1\} \) is the true label
- \( p_i = \sigma(\mathbf{w}^T \mathbf{x}_i + b) \) is the predicted probability of Class 1

### Effect of Targeted Label Corruption

In a targeted attack:
- Samples that *look like* Class 1 are deliberately labeled as Class 0
- The model predicts a high \( p_j \approx 1 \) for these samples
- With poisoned labels, their loss becomes:

\[
-\log(1 - p_j)
\]

Since \( (1 - p_j) \rightarrow 0 \), this loss term becomes **very large**, producing strong gradients.

### Consequence

- Optimization is dominated by these poisoned points
- The model is forced to reduce this artificial error
- The decision boundary \( \mathbf{w}^T \mathbf{x} + b = 0 \) shifts **toward the true Class 1 region**
- This increases false negatives for Class 1, achieving the attacker’s goal

---

## Attack Implementation Strategy

The attack is implemented by selectively flipping labels **only within the target class**.

### High-Level Logic

1. Identify all training samples belonging to the target class
2. Compute how many of these should be poisoned
3. Randomly select that subset
4. Flip their labels to the adversary’s chosen class
5. Preserve traceability by recording flipped indices

---

## Targeted Label Flipping Function

### Input Validation

The function ensures:
- Poisoning percentage lies in the range \([0, 1]\)
- Target and new classes are distinct
- Both classes exist in the label set

This prevents invalid or ambiguous poisoning behavior.

---

### Target Class Identification

All indices belonging to the target class are extracted. If no such samples exist, the function safely exits without modifying the dataset.

---

### Poisoning Scope Control

The number of labels to flip is computed **relative only to the target class size**, not the entire dataset. This ensures precise and proportional poisoning.

---

### Randomized Selection with Reproducibility

- A seeded random number generator is used
- Selection is uniform and without replacement
- Results are deterministic when using the same seed

---

### Label Manipulation

- The original label array is copied to avoid side effects
- Selected indices are reassigned to the new class
- Only the targeted samples are modified

---

### Transparency and Traceability

Verbose output reports:
- Target and new classes
- Number of samples in the target class
- Number of samples flipped
- Confirmation of successful poisoning

The function returns:
- The poisoned label array
- The exact indices of modified samples

---

## Generating the Targeted Poisoned Dataset

In the experiment:
- **40% of Class 1 samples** are flipped to Class 0
- This creates a heavily biased training signal against positive reviews
- Visualization clearly highlights poisoned points within the positive class region

The resulting plot shows:
- Clean Class 0 and Class 1 clusters
- A dense concentration of flipped samples inside the true Class 1 feature space

---

## Model Training on Targeted Poisoned Data

A new Logistic Regression model is trained using:
- Original feature vectors (`X_train`)
- Targeted poisoned labels (`y_train_targeted_poisoned`)

This ensures that **any observed behavioral changes are solely attributable to label manipulation**, not feature distortion.

---

## Key Takeaways

- Targeted Label Attacks are **more strategic and damaging** than random poisoning
- They introduce **class-specific bias** without necessarily harming global accuracy
- Decision boundaries shift in a *directional and adversary-controlled manner*
- Such attacks are especially dangerous because they:
  - Mimic natural label noise
  - Evade detection by aggregate metrics
  - Directly undermine fairness and reliability

---

## Conclusion

Targeted Label Attacks demonstrate how a relatively small, carefully chosen subset of poisoned labels can **fundamentally alter a model’s behavior for specific inputs**. By exploiting the optimization dynamics of Logistic Regression, attackers can reliably bias predictions against a chosen class, making this form of data poisoning both subtle and highly effective in real-world machine learning systems.
