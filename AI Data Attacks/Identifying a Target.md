# Identifying a Target for a Clean Label Attack

## Overview

This stage of the Clean Label Attack focuses on **selecting an appropriate target point** from the training dataset. The objective is to identify a data point that is *correctly classified by the baseline model* but is *highly susceptible to misclassification* after a subtle shift in the decision boundary caused by poisoned training data.

Specifically, the attack aims to force a model to misclassify a point that truly belongs to **Class 1 (Yellow)** as **Class 0 (Blue)** after retraining, without altering the target’s label or features directly.

---

## Attack Objective

Let the target point be denoted as:

- **Feature vector:** \( \mathbf{x}_{target} \)
- **True label:** \( y_{target} = 1 \)

The goal is to ensure that:
- The **baseline model** correctly classifies \( \mathbf{x}_{target} \) as Class 1.
- After training on poisoned data, the **retrained model** misclassifies the same point as Class 0.

To maximize the effectiveness of the attack, the chosen target point should lie **close to the decision boundary** between Class 1 and Class 0.

---

## Rationale for Boundary-Proximal Target Selection

Points near a decision boundary are inherently unstable:
- A **small shift** in the boundary can flip their predicted class.
- Clean label attacks rely on **indirect manipulation** of the model’s learned parameters, making boundary-adjacent points ideal targets.

Thus, the selection process prioritizes points that are:
1. Correctly classified as Class 1.
2. As close as possible to the Class 0 vs Class 1 decision boundary.

---

## Decision Function Analysis

### Score-Based Classification

For a linear classifier, each class \( k \) produces a score:

\[
z_k(\mathbf{x}) = \mathbf{w}_k^T \mathbf{x} + b_k
\]

The model predicts the class with the **highest score**.

---

### Class 0 vs Class 1 Boundary

To analyze the boundary between Class 0 and Class 1, the score difference function is defined as:

\[
f_{01}(\mathbf{x}) = z_0 - z_1 = (\mathbf{w}_0 - \mathbf{w}_1)^T \mathbf{x} + (b_0 - b_1)
\]

Interpretation:
- \( f_{01}(\mathbf{x}) < 0 \) → Model prefers **Class 1**
- \( f_{01}(\mathbf{x}) = 0 \) → Point lies **exactly on the boundary**
- \( f_{01}(\mathbf{x}) > 0 \) → Model prefers **Class 0**

---

## Target Selection Criteria

A valid and optimal target point must satisfy:

1. **Correct ground-truth label**  
   \[
   y_{target} = 1
   \]

2. **Correct baseline classification**  
   \[
   f_{01}(\mathbf{x}_{target}) < 0
   \]

3. **Maximum vulnerability**  
   Among all Class 1 points with \( f_{01} < 0 \), select the one with the **largest (least negative)** value of \( f_{01} \).  
   This corresponds to the point **closest to the decision boundary** while still being correctly classified.

---

## Practical Selection Procedure

1. Extract baseline model parameters \( \mathbf{w}_0, \mathbf{w}_1, b_0, b_1 \).
2. Compute the boundary parameters:
   - \( \mathbf{w}_{diff} = \mathbf{w}_0 - \mathbf{w}_1 \)
   - \( b_{diff} = b_0 - b_1 \)
3. Identify all training points with true label \( y = 1 \).
4. Compute \( f_{01}(\mathbf{x}) \) for each of these points.
5. Filter points where \( f_{01}(\mathbf{x}) < 0 \).
6. Select the point with the **largest negative** \( f_{01} \) value.
7. Perform sanity checks to ensure:
   - The label is indeed Class 1.
   - The baseline model predicts Class 1.
   - The point lies on the correct side of the boundary.

If no correctly classified Class 1 points exist (unlikely for a well-trained model), the closest point to the boundary in absolute terms is selected as a fallback.

---

## Selected Target Point

Using the above methodology, the algorithm identifies a suitable target:

- **Target index:** 373  
- **Feature vector:** `[-0.5511, -0.3668]`  
- **True label:** Class 1  
- **Baseline prediction:** Class 1  
- **Decision value:**  
  \[
  f_{01}(\mathbf{x}_{target}) = -0.0493
  \]

This value is **negative but very close to zero**, confirming that the point is:
- Correctly classified by the baseline model
- Extremely close to the Class 0 vs Class 1 decision boundary

---

## Visualization and Interpretation

A scatter plot of the training data highlights:
- Class 0 points (Blue)
- Class 1 points (Orange)
- Class 2 points (Red)
- The selected target point (White cross)

The visualization clearly shows the target positioned near the boundary separating Class 0 and Class 1, reinforcing its suitability for a clean label attack.

---

## Conclusion

The identified target point represents an **ideal candidate** for a Clean Label Attack:
- It is legitimate and correctly labeled.
- It lies in a high-risk region of the feature space.
- A minor boundary shift caused by poisoned training data is likely to flip its classification.

This careful and principled target selection forms the foundation for the subsequent poisoning and retraining steps of the attack.
