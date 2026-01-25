# Evaluating the Label Flipping Attack

## Overview

This phase focuses on **empirically evaluating the impact of a label flipping attack** on a sentiment analysis model trained using Logistic Regression. Unlike the attack construction itself, the emphasis here is on *measurement and interpretation*: how poisoned labels influence model behavior, decision boundaries, and predictive performance when evaluated on **clean, unseen test data**.

The evaluation is conducted progressively, starting from a small poisoning ratio (10%) and increasing up to 50%, allowing us to observe both **subtle and severe effects** of label corruption.

---

## Experimental Methodology

The evaluation follows a consistent and systematic workflow at each poisoning level:

1. **Label Poisoning**  
   A fixed percentage of training labels is flipped using the previously defined `flip_labels` function.  
   - Only the labels are modified; the feature vectors remain unchanged.
   - The indices of flipped samples are recorded for visualization and analysis.

2. **Visualization of Poisoned Training Data**  
   The function `plot_poisoned_data` is used to:
   - Display the original data distribution.
   - Highlight flipped samples using a distinct marker.
   - Preserve visibility of class structure while exposing label corruption.

3. **Model Training on Poisoned Data**  
   A new Logistic Regression model is trained using:
   - Original training features (`X_train`)
   - Poisoned labels (`y_train_poisoned`)

4. **Evaluation on Clean Test Data**  
   The trained model is evaluated **only against the clean test set** (`X_test`, `y_test`).  
   This step is critical because it measures the real-world impact of poisoning on legitimate predictions.

5. **Decision Boundary Analysis**  
   The learned decision boundary is visualized to assess how the model adapts to conflicting supervision introduced by poisoned labels.

6. **Result Tracking**  
   For each poisoning level, the following are stored:
   - Poisoning percentage
   - Test accuracy
   - Trained model
   - Poisoned labels
   - Indices of flipped samples
   - Decision boundary predictions

---

## Baseline Reference (0% Poisoning)

Before introducing any attack, a **baseline model** is trained on clean data:

- **Accuracy on clean test set**: High (≈ 99.33%)
- **Decision boundary**: Optimal and well-aligned with the true data distribution
- This baseline serves as a reference point for all comparisons.

---

## Evaluation at 10% Poisoning

### Observations

- **Accuracy Impact**  
  Despite 10% of training labels being flipped, the model achieves **the same accuracy as the baseline** on the clean test set.

- **Decision Boundary Shift**  
  Although accuracy remains unchanged, the decision boundary **shifts slightly**.
  - The model compensates for mislabeled points by adjusting its parameters.
  - This adjustment reflects internal degradation not visible through accuracy alone.

- **Overlay Analysis**  
  When the baseline and poisoned boundaries are plotted together:
  - The poisoned boundary deviates subtly from the clean one.
  - This confirms that **decision boundary distortion precedes measurable accuracy loss**.

---

## Increasing Poisoning Levels (20%–50%)

The experiment is repeated for poisoning levels of **20%, 30%, 40%, and 50%**, following the same pipeline.

### Key Trends

- **Decision Boundary Distortion**
  - With each increase in poisoned labels, the boundary becomes progressively more warped.
  - The model increasingly sacrifices alignment with the true data structure to satisfy incorrect labels.

- **Accuracy Behavior**
  - Due to the synthetic dataset being *cleanly separable*, accuracy remains high up to 40%.
  - A noticeable degradation begins to appear at **50% poisoning**, where corrupted supervision overwhelms the true signal.

- **Model Stability**
  - Even when accuracy remains stable, the learned model is no longer trustworthy.
  - This highlights the limitation of accuracy as a sole evaluation metric under adversarial conditions.

---

## Accuracy vs. Poisoning Percentage

A consolidated plot of **test accuracy versus poisoning percentage** reveals:

- Accuracy remains deceptively stable across low-to-moderate poisoning levels.
- A sharp decline emerges only when poisoning becomes extreme.
- This demonstrates that **models can appear robust while silently degrading**.

---

## Combined Decision Boundary Visualization

By overlaying decision boundaries for all poisoning levels (0%–50%) in a single plot:

- The gradual and cumulative distortion becomes visually evident.
- Boundaries diverge further from the clean baseline as poisoning increases.
- This visualization clearly shows how the model’s internal representation deteriorates long before accuracy collapses.

---

## Key Takeaways

- **Label flipping attacks can significantly alter model behavior without immediate accuracy loss.**
- **Decision boundary analysis is essential** for detecting poisoning effects that accuracy metrics may conceal.
- **Clean, well-separated data masks attack severity**; real-world datasets are far more vulnerable.
- Even small-scale poisoning introduces instability that compounds as corruption increases.

---

## Conclusion

This evaluation demonstrates that label flipping is a **stealthy yet powerful data poisoning attack**. While traditional performance metrics may suggest resilience, deeper inspection reveals consistent internal degradation. In realistic, noisy datasets, such boundary shifts would almost certainly translate into substantial performance loss, misclassification, and reduced model reliability.

The experiment underscores the importance of **robust data validation, anomaly detection, and adversarial-aware evaluation strategies** in machine learning pipelines.
