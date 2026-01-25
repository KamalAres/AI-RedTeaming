# Evaluating the Targeted Label Poisoning Attack

## Overview

This section evaluates the effectiveness of a **targeted label poisoning attack** by analyzing how a model trained on poisoned data behaves when tested against clean, unseen data. The evaluation focuses on three core aspects:

1. **Performance degradation on a clean test set**
2. **Structural changes in the decision boundary**
3. **Generalization of the attack impact to unseen data**

Together, these analyses confirm whether the attack achieved its goal: degrading performance **specifically for a chosen target class** while maintaining plausible overall behavior.

---

## 1. Performance on the Clean Test Set

After training the `targeted_poisoned_model`, predictions are made on the **original, unmodified test set**. This ensures that any observed degradation is solely due to poisoning during training and not test data contamination.

### Accuracy Comparison

- **Baseline model accuracy:** `0.9933`
- **Targeted poisoned model accuracy:** `0.8100`

This represents a **substantial drop in overall accuracy**, indicating that the poisoning attack had a strong impact despite modifying only a subset of training labels.

---

## 2. Class-Specific Impact Analysis

### Classification Report Insights

The classification report reveals that the attack does not affect all classes equally:

- **Class 0**
  - Recall: `1.00`
  - The model continues to correctly identify nearly all Class 0 instances.

- **Class 1 (Target Class)**
  - Recall: `0.61`
  - The model correctly identifies only **61% of true Class 1 samples**.

This asymmetry is a defining characteristic of a **targeted attack**: the degradation is concentrated on the chosen class rather than uniformly spread across all classes.

---

## 3. Confusion Matrix Interpretation

The confusion matrix provides a concrete view of how predictions have shifted:

- **57 False Negatives**
  - True Class 1 samples incorrectly predicted as Class 0
- **0 False Positives**
  - No Class 0 samples incorrectly predicted as Class 1

This pattern confirms that the poisoned model has become **biased toward predicting Class 0**, aligning precisely with the attacker’s objective.

---

## 4. Decision Boundary Shift

To understand *why* these misclassifications occur, the decision boundaries of the baseline and poisoned models are compared.

### Observations

- The **baseline decision boundary** cleanly separates the two classes.
- The **targeted poisoned boundary** is visibly shifted toward the Class 1 region.

This shift occurs because the model is forced to reconcile flipped Class 1 training points that were relabeled as Class 0. As a result, the learned boundary favors Class 0 predictions even in regions that previously belonged to Class 1.

---

## 5. Evaluation on Unseen Data

To validate whether the attack generalizes beyond the test set, a **new dataset of unseen samples** is generated using similar distribution parameters but a different random seed.

### Key Findings

- A significant portion of true Class 1 samples are misclassified as Class 0.
- Misclassified points cluster near or within the genuine Class 1 region.
- These errors align closely with the shifted poisoned decision boundary.

This demonstrates that the attack does not merely overfit the training or test data—it **systematically alters the model’s behavior** on new inputs.

---

## 6. Visualization Insights

The final visualization overlays:
- Model predictions on unseen data
- The poisoned decision boundary
- Highlighted misclassified target-class samples

Red “X” markers indicate true Class 1 samples incorrectly predicted as Class 0. Their placement confirms that the boundary shift directly causes the intended misclassification behavior.

---

## Conclusion

The evaluation conclusively shows that the targeted label poisoning attack was successful:

- Overall accuracy dropped significantly.
- Performance degradation was **concentrated on the target class**.
- The decision boundary was strategically shifted.
- The attack generalized to unseen data, causing persistent misclassification of the target class.

This illustrates the real-world risk of targeted data poisoning: even limited label manipulation during training can lead to **systematic, hard-to-detect failures** in deployed machine learning models.
