# The Label Flipping Attack

## From Baseline to Active Attack

With a strong baseline model established on clean data, the next step is to deliberately corrupt the training process through a **label flipping attack**. This attack targets the integrity of the training labels, directly undermining the learning objective of the model. Instead of altering features or the model architecture, the adversary manipulates the ground truth labels that the model relies on to learn correct decision boundaries.

In this scenario, the attacker operates on the training labels (`y_train`) and flips a specified percentage of them. Negative labels (0) are changed to positive (1), and positive labels (1) are changed to negative (0). The features remain unchanged, making the attack subtle and difficult to detect through simple data inspection.

---

## Why Label Flipping Works

Logistic Regression learns its parameters—weights \( \mathbf{w} \) and bias \( b \)—by minimizing the **average binary cross-entropy loss** over the training dataset. The entire training process is driven by aligning predicted probabilities \( p_i \) with the provided labels \( y_i \).

When a label is flipped from its true value \( y_i \) to an incorrect value \( y_i' \), the contribution of that data point to the loss function is fundamentally corrupted. For a data point that truly belongs to class 0 but is mislabeled as class 1, the loss term changes in a way that strongly penalizes correct predictions. If the model predicts a low probability for class 1—as it should—the loss becomes very large due to the incorrect label.

This inflated error signal exerts disproportionate influence during optimization. As a result, the model adjusts its parameters not only to fit correctly labeled data but also to accommodate these poisoned points. The cumulative effect is a distorted decision boundary that no longer reflects the true underlying data distribution.

---

## Effect on the Decision Boundary

The decision boundary in Logistic Regression is defined by the equation:

\[
\mathbf{w}^T \mathbf{x} + b = 0
\]

Label flipping pushes this boundary away from its optimal position. Even a relatively small fraction of flipped labels can pull the boundary toward regions that reduce overall classification accuracy. The model attempts to reconcile conflicting signals: correctly labeled points pushing the boundary one way, and poisoned points pulling it another.

Over time, this leads to a classifier that is less confident, less accurate, and less reliable across the feature space.

---

## Implementing the Attack Mechanism

To execute the attack in a controlled and reproducible manner, a dedicated function is introduced to encapsulate the label flipping logic. This function accepts the original training labels and a **poisoning percentage**, which specifies what fraction of the dataset will be corrupted.

The function first validates that the poisoning percentage is meaningful, ensuring it lies between 0 and 1. It then calculates the absolute number of labels to flip based on the dataset size. If this number is zero, the function safely exits without modifying the data.

To select which labels to flip, the function uses a seeded random number generator. This ensures that the same labels are flipped each time the experiment is run, enabling consistent analysis and comparison. The selected indices represent the exact training samples targeted by the attack.

Finally, the labels at these indices are inverted. For binary classification, this inversion is straightforward: 0 becomes 1, and 1 becomes 0. The function returns both the poisoned label array and the indices of the flipped samples, allowing precise tracking of the attack’s scope.

---

## Visualizing the Impact of Poisoned Labels

To better understand the effect of the attack, a visualization function is used to plot the dataset after poisoning. This plot distinguishes between unchanged data points and those whose labels have been flipped.

Unchanged points appear with standard markers and colors corresponding to their class. Flipped points are highlighted using distinct markers and red outlines, making the poisoned samples immediately visible in the feature space. Importantly, these flipped points are colored according to their new, incorrect labels, illustrating how they now contradict the natural clustering of the data.

This visualization clearly demonstrates how poisoned labels introduce inconsistencies into an otherwise clean and separable dataset.

---

## Significance of the Attack

The label flipping attack highlights a critical weakness in machine learning systems: **models inherently trust their training labels**. By corrupting even a subset of these labels, an attacker can significantly influence the learning process without needing access to the model architecture or inference interface.

Because the attack operates at the data level, its effects propagate naturally through the training pipeline. The resulting model appears legitimate but behaves unpredictably, setting the stage for measurable performance degradation and incorrect downstream decisions.

This attack serves as a clear illustration of why protecting label integrity is essential for maintaining trustworthy and secure AI systems.
