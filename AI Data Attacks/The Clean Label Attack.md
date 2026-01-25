# The Clean Label Attack: Boundary Manipulation via Feature Perturbation

## Overview

After identifying a vulnerable target point \( \mathbf{x}_{target} \) from Class 1, the next phase of the Clean Label Attack focuses on **manipulating the training data** to induce a controlled misclassification of this target. This is achieved not by altering labels, but by **carefully perturbing the features of selected training points** so that the model’s learned decision boundary shifts during retraining.

The attack exploits the model’s reliance on labeled examples to shape its decision regions. By introducing feature–label inconsistencies in a subtle and localized manner, the attacker can coerce the model into adjusting its boundary in a way that ultimately engulfs the target point.

---

## Core Attack Strategy

The central idea is to **shift the Class 0 vs Class 1 decision boundary** just enough so that the previously correctly classified target point falls on the wrong side after retraining.

This is accomplished by:
1. Identifying **Class 0 training points closest to the target**.
2. Applying **small, carefully chosen perturbations** to their feature vectors.
3. Keeping their **original Class 0 labels unchanged**.
4. Retraining the model on this poisoned dataset, forcing it to reconcile conflicting evidence.

If successful, the resulting boundary shift causes the target \( \mathbf{x}_{target} \) to be misclassified as Class 0.

---

## Selection of Influential Neighbors

### Motivation

Training points closest to the target in feature space have the **strongest local influence** on the position of the decision boundary near that target. Perturbing these neighbors provides an efficient way to manipulate the boundary without widespread changes to the dataset.

---

### Neighbor Identification Procedure

- Only **Class 0** points are considered, since the attack aims to move the boundary in favor of Class 0.
- A **k-nearest neighbors (k-NN)** search is performed using Euclidean distance.
- The search is conducted **only over Class 0 training samples**, and queried with \( \mathbf{x}_{target} \).

A hyperparameter, `n_neighbors_to_perturb`, controls how many neighbors are modified.

---

### Resulting Neighbors

In this implementation:
- **5 closest Class 0 neighbors** are identified.
- All are located very near the target in feature space.
- These points form the local anchors used to influence the boundary shift.

Their proximity ensures that even small perturbations will have a measurable impact on the local geometry of the decision boundary.

---

## Designing the Perturbation

### Decision Boundary Geometry

The Class 0 vs Class 1 decision boundary is defined by:

\[
f_{01}(\mathbf{x}) = (\mathbf{w}_0 - \mathbf{w}_1)^T \mathbf{x} + (b_0 - b_1) = 0
\]

- Points with \( f_{01}(\mathbf{x}) > 0 \) lie on the **Class 0 side**.
- Points with \( f_{01}(\mathbf{x}) < 0 \) lie on the **Class 1 side**.

The vector \( \mathbf{w}_0 - \mathbf{w}_1 \) is **normal (perpendicular)** to the boundary.

---

### Direction of the Attack

To push a Class 0 point into the Class 1 region, the perturbation must move it:
- **Opposite to the boundary normal**, i.e., in the direction \( -(\mathbf{w}_0 - \mathbf{w}_1) \).

This direction represents the **shortest path across the boundary**, ensuring minimal feature distortion.

---

### Normalized Push Direction

The push direction is normalized to unit length:

\[
\mathbf{u}_{push} = -\frac{\mathbf{w}_0 - \mathbf{w}_1}{\|\mathbf{w}_0 - \mathbf{w}_1\|}
\]

This normalization allows precise control over the magnitude of the perturbation.

---

### Perturbation Magnitude

A small scalar hyperparameter, \( \epsilon_{cross} \), controls how far each neighbor is pushed:

\[
\delta_i = \epsilon_{cross} \cdot \mathbf{u}_{push}
\]

- Smaller values preserve plausibility and stealth.
- Larger values strengthen the attack but risk making poisoned points suspicious.

In this case, a moderate value is chosen to ensure the points cross the boundary while remaining visually and statistically plausible.

---

## Applying the Poisoning

### Construction of the Poisoned Dataset

To preserve dataset integrity:
- A **copy of the original training data** is created.
- Only the **features** of selected Class 0 neighbors are modified.
- **Labels remain unchanged** (still Class 0).

For each selected neighbor \( \mathbf{x}_i \):
\[
\mathbf{x}'_i = \mathbf{x}_i + \delta_i
\]

This ensures:
- The point now lies in the **Class 1 region** according to the baseline model.
- The label still claims it is **Class 0**, creating a deliberate inconsistency.

---

### Verification of Boundary Crossing

For each perturbed neighbor:
- The decision value \( f_{01}(\mathbf{x}_i) \) is computed before perturbation.
- The decision value \( f_{01}(\mathbf{x}'_i) \) is computed after perturbation.

A successful perturbation satisfies:
\[
f_{01}(\mathbf{x}_i) > 0 \quad \text{and} \quad f_{01}(\mathbf{x}'_i) < 0
\]

This confirms that the point has crossed the boundary as intended.

---

## Effect on Model Retraining

When the model retrains on the poisoned dataset, it encounters a contradiction:
- Points labeled **Class 0** now reside in a region typically dominated by **Class 1**.

To minimize training loss while respecting the labels, the model:
- **Shifts the decision boundary outward**, deeper into the original Class 1 region.
- Adjusts parameters so the perturbed points are classified as Class 0.

If the shift is large enough, the nearby target point \( \mathbf{x}_{target} \) is swept along and ends up on the **Class 0 side** of the new boundary.

---

## Visualization and Interpretation

The poisoned training data visualization shows:
- The **target point** unchanged and still labeled Class 1.
- **Perturbed neighbors** (still labeled Class 0) now positioned inside the Class 1 region.
- A clear visual inconsistency where blue-labeled points appear among yellow points.

This inconsistency is the **driving force** behind the induced boundary shift during retraining.

---

## Conclusion

This phase of the Clean Label Attack demonstrates how:
- Carefully chosen neighbors,
- Minimal, geometrically informed perturbations,
- And unchanged labels

can collectively manipulate a model’s learned decision boundary. The attack remains stealthy, preserves label integrity, and exploits the model’s trust in its training data—setting the stage for the target point’s eventual misclassification.
