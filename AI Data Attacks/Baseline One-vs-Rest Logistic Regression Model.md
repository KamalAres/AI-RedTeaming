# Baseline One-vs-Rest Logistic Regression Model

## Purpose of the Baseline Model

Before conducting a **Clean Label Attack**, it is essential to establish a **baseline reference model** trained on clean, unmanipulated data. This baseline serves two critical roles:

1. **Performance Benchmark** – It provides the expected accuracy and classification behavior under normal conditions.
2. **Geometric Reference** – It defines the original decision boundaries in feature space, which will later be compared against boundaries learned from poisoned data.

Any deviation observed after the attack can then be directly attributed to the effects of clean label poisoning rather than model instability or poor initial training.

---

## Why One-vs-Rest Logistic Regression?

Logistic Regression is inherently a **binary classifier**, but the dataset contains **three distinct classes**. To handle this multi-class setting, the **One-vs-Rest (OvR)** strategy is used.

### OvR Strategy Explained

For a problem with \( K = 3 \) classes:

- The model trains **three independent binary classifiers**:
  - Class 0 vs. Rest
  - Class 1 vs. Rest
  - Class 2 vs. Rest
- Each classifier treats its own class as the **positive class** and merges all remaining classes into a single **negative class**.

Scikit-learn’s `OneVsRestClassifier` automates this process by wrapping a binary classifier—in this case, Logistic Regression.

---

## Mathematical Interpretation of OvR

Each binary classifier \( k \) learns:

- A weight vector \( \mathbf{w}_k \)
- An intercept (bias) \( b_k \)

For an input feature vector \( \mathbf{x} \), the classifier computes a linear score:

\[
z_k = \mathbf{w}_k^T \mathbf{x} + b_k
\]

This score represents how strongly the model believes that \( \mathbf{x} \) belongs to class \( k \).

### Final Prediction Rule

To assign a class label, the OvR classifier evaluates all scores and selects the class with the highest value:

\[
\hat{y} = \arg\max_{k \in \{0,1,2\}} (\mathbf{w}_k^T \mathbf{x} + b_k)
\]

---

## Decision Boundaries in OvR

The decision boundary between any two classes \( i \) and \( j \) is defined by points where their scores are equal:

\[
\mathbf{w}_i^T \mathbf{x} + b_i = \mathbf{w}_j^T \mathbf{x} + b_j
\]

Rewriting this gives a linear boundary:

\[
(\mathbf{w}_i - \mathbf{w}_j)^T \mathbf{x} + (b_i - b_j) = 0
\]

In a two-dimensional feature space, each such boundary is a **straight line**. Collectively, these boundaries partition the space into **three decision regions**, one for each class.

---

## Training the Baseline Model

The baseline model is trained using:

- **Logistic Regression** as the base estimator
- **One-vs-Rest classification** to support three classes
- The **liblinear** solver, well-suited for smaller datasets and OvR setups
- Default regularization strength \( C = 1.0 \)
- A fixed random seed for reproducibility

Training is performed exclusively on the **clean training dataset**, ensuring no contamination from poisoning.

---

## Baseline Performance

After training, the model is evaluated on the clean test set:

- **Test Accuracy:** approximately **0.96**

This high accuracy confirms that:

- The dataset is well-structured and separable.
- Logistic Regression with OvR is an appropriate model choice.
- The learned decision boundaries reflect the true structure of the data.

---

## Decision Boundary Visualization

To better understand the model’s behavior, predictions are generated over a dense mesh grid spanning the feature space. This allows visualization of:

- **Decision regions**: colored areas indicating which class the model predicts in each region
- **Decision boundaries**: lines where predicted classes change
- **Training samples**: overlaid to show how data points align with the learned regions

### Interpretation of the Visualization

- Each class occupies a **clearly defined region** in feature space:
  - Class 0 (Azure)
  - Class 1 (Yellow)
  - Class 2 (Red)
- Boundaries between classes are **linear and well-positioned**, reflecting the clean separation seen in the original dataset.
- Very few training points lie near incorrect regions, consistent with the high test accuracy.

---

## Importance of This Baseline

This baseline model establishes:

- A **stable and interpretable reference** for normal behavior
- Clearly defined decision boundaries that can later be compared against those learned from poisoned data
- A trusted accuracy benchmark against which attack-induced degradation can be measured

With this foundation in place, any subsequent shifts in decision boundaries or targeted misclassifications can be confidently attributed to the **Clean Label Attack** rather than natural model variance.
