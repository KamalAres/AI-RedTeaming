# Clean Label Attacks: Concept and Experimental Setup

## Introduction

In previous sections, data poisoning attacks such as **Label Flipping** and **Targeted Label Flipping** were examined. These attacks operate by directly modifying the *ground truth labels* of training samples. Clean Label Attacks represent a fundamentally different and more subtle class of poisoning attacks. Instead of manipulating labels, the attacker **alters only the feature values** of selected training instances while keeping their labels unchanged and seemingly correct.

This property makes clean label attacks particularly dangerous: the poisoned data often appears legitimate during manual inspection or automated data validation, since the labels remain consistent with the modified features.

---

## Defining Characteristics of Clean Label Attacks

A clean label attack is defined by the following properties:

- **Labels remain unchanged**: All poisoned training samples retain their original class labels.
- **Feature manipulation only**: The attacker introduces carefully crafted perturbations to feature values.
- **Plausibility preserved**: The modified features are designed such that the original label still appears reasonable.
- **Highly targeted objective**: The goal is typically to cause misclassification of a *specific target instance* or a small group of instances at inference time.
- **Stealth**: Since labels are not altered, traditional label-consistency checks often fail to detect the attack.

The attack succeeds by subtly reshaping the learned decision boundaries during training rather than introducing obvious inconsistencies.

---

## Intuitive Example: Manufacturing Quality Control

To illustrate the concept, consider an automated quality control system in a manufacturing environment. The system uses two measured features—such as **component length** and **component weight**—to classify parts into three categories:

- **Class 0**: Major Defect  
- **Class 1**: Acceptable  
- **Class 2**: Minor Defect  

Suppose an adversary wants a specific batch of *Acceptable* parts (Class 1) to be incorrectly rejected as having a *Major Defect* (Class 0).

### Attack Strategy

Instead of relabeling any data, the adversary:

1. Selects several training samples that are genuinely labeled as **Major Defect (Class 0)**.
2. Slightly perturbs their feature values (length and weight) so that these samples move closer to the region typically occupied by **Acceptable parts (Class 1)** in feature space.
3. Leaves their labels unchanged as **Class 0**.

These poisoned samples now appear as “Major Defect” points embedded near or inside the Acceptable region.

---

## Effect on Model Training

When the model is retrained on this poisoned dataset, it encounters a contradiction:

- It must correctly classify the perturbed samples as **Class 0**, because their labels indicate a major defect.
- However, their feature values resemble **Class 1** samples.

To minimize training error, the model adapts by **shifting the decision boundary** between Class 0 and Class 1. This shift can be large enough that a genuine, previously well-classified Class 1 target instance now falls on the Class 0 side of the boundary.

Crucially, the attacker achieves the desired misclassification **without ever changing a single label** in the training data.

---

## Dataset Construction for Demonstration

To demonstrate clean label attacks in practice, a synthetic dataset is created that mirrors the quality control scenario.

### Dataset Properties

- **Number of classes**: 3  
  - Class 0: Major Defect  
  - Class 1: Acceptable  
  - Class 2: Minor Defect  
- **Feature space**: Two-dimensional (e.g., conceptual length and weight)
- **Data generation**: Gaussian clusters using `make_blobs`
- **Feature scaling**: Standardization applied to normalize the feature ranges

Each data point is represented as:

\[
\mathbf{x}_i = (x_{i1}, x_{i2}), \quad y_i \in \{0, 1, 2\}
\]

This setup provides a clean, controlled environment for observing how small feature perturbations can influence model behavior.

---

## Train–Test Split and Reproducibility

The dataset is split into training and testing subsets with the following characteristics:

- **Total samples**: 1500
- **Training set**: 1050 samples (70%)
- **Test set**: 450 samples (30%)
- **Stratified split**: Preserves class distribution across splits
- **Fixed random seed**: Ensures full reproducibility of results

The output confirms that all three classes are present and evenly represented.

---

## Visualizing the Clean Training Data

Before introducing any poisoning, the clean training data is visualized in the standardized feature space.

### Observations from the Visualization

- The three classes form **distinct, well-separated clusters**.
- Decision boundaries between classes are expected to be relatively stable and well-defined.
- This clear separation provides a strong baseline, making it easier to observe and attribute any later boundary distortions to the clean label attack.

The visualization also supports later stages of the experiment, where specific points—such as target instances and poisoned samples—can be highlighted to illustrate how the attack operates.

---

## Summary

Clean label attacks demonstrate that **label integrity alone is not sufficient** to guarantee the security of machine learning systems. By manipulating only feature values and preserving label correctness, an attacker can:

- Stealthily poison training data,
- Force the model to reshape its decision boundaries,
- Cause targeted misclassification of specific instances during inference.

The synthetic three-class dataset and its clear initial separation provide an ideal foundation for analyzing how such subtle feature-level manipulations can lead to significant downstream impacts on model behavior.
