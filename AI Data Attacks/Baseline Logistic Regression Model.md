# Baseline Logistic Regression Model

## Purpose of the Baseline Model

Before evaluating the impact of a label flipping attack, it is essential to establish a **baseline model** trained on clean, uncompromised data. This baseline represents the expected behavior of the system under normal, non-adversarial conditions and provides a reference point against which the effects of data poisoning can be measured.

In this scenario, a Logistic Regression classifier is trained using the original training dataset `(X_train, y_train)` and evaluated on a separate, unseen test set. Any degradation observed later can therefore be attributed to the attack rather than model choice or data complexity.

---

## Logistic Regression as a Binary Classifier

Logistic Regression is a widely used algorithm for **binary classification** tasks, such as sentiment analysis. Although its name suggests regression, it is fundamentally a probabilistic classifier that estimates the likelihood of a data point belonging to a particular class.

Each customer review is represented by a feature vector:

\[
\mathbf{x}_i = (x_{i1}, x_{i2})
\]

where the features are numerical representations derived from the original text data.

---

## Linear Combination and Log-Odds

The model computes a linear combination of the input features using learned weights and a bias term:

\[
z_i = \mathbf{w}^T \mathbf{x}_i + b = w_1 x_{i1} + w_2 x_{i2} + b
\]

This value \( z_i \) represents the **log-odds** of the review having positive sentiment:

\[
z_i = \log\left(\frac{P(y_i = 1 \mid \mathbf{x}_i)}{1 - P(y_i = 1 \mid \mathbf{x}_i)}\right)
\]

In other words, it quantifies how strongly the feature values support a positive classification relative to a negative one.

---

## Sigmoid Function and Probability Estimation

To convert the log-odds into a probability, Logistic Regression applies the **sigmoid function**:

\[
p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}} = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x}_i + b)}}
\]

The sigmoid function maps any real-valued input into the range \([0, 1]\), allowing the model to interpret its output as the probability that a review has positive sentiment.

---

## Training Objective and Loss Function

During training, the model learns optimal values for the weight vector \( \mathbf{w} \) and bias \( b \) by minimizing a loss function over the training data. For binary classification, the standard choice is **binary cross-entropy loss**, also known as log-loss:

\[
L(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
\]

Here:
- \( N \) is the number of training samples,
- \( y_i \in \{0, 1\} \) is the true sentiment label,
- \( p_i \) is the predicted probability of positive sentiment.

Optimization algorithms such as gradient descent iteratively adjust the parameters to minimize this loss, encouraging predicted probabilities to align closely with the true labels.

---

## Prediction and Classification Rule

Once trained, the model predicts the sentiment of new, unseen reviews by computing:

\[
p = \sigma(\mathbf{w}^T \mathbf{x} + b)
\]

A decision threshold is applied to convert this probability into a class label. Typically:
- If \( p \geq 0.5 \), the review is classified as **positive** (Class 1),
- If \( p < 0.5 \), it is classified as **negative** (Class 0).

---

## Decision Boundary Interpretation

In a two-dimensional feature space, the Logistic Regression model produces a **linear decision boundary**. This boundary corresponds to the set of points where the model is exactly uncertain:

\[
\mathbf{w}^T \mathbf{x} + b = 0
\]

At this line, \( p = 0.5 \). On one side of the boundary, reviews are predicted as negative sentiment, and on the other side, as positive sentiment. The training process identifies the line that best separates the two clusters in the data.

---

## Baseline Model Training and Evaluation

The baseline Logistic Regression model is trained on the clean training dataset and evaluated on the test set. The resulting accuracy is very high, indicating that the model generalizes well and that the synthetic data is cleanly separable.

Visualizing the decision boundary confirms this result: the learned linear boundary effectively divides the negative and positive sentiment clusters with minimal overlap. This strong baseline performance establishes a clear point of comparison for later experiments involving label flipping, where deviations in accuracy and boundary shape will highlight the impact of data poisoning.

---

## Significance of the Baseline

Establishing a strong baseline is critical for security analysis. It demonstrates that the model behaves as expected when trained on trustworthy data and ensures that any subsequent performance degradation can be attributed to adversarial manipulation rather than inherent model limitations. This baseline thus serves as the foundation for evaluating the effectiveness and severity of label flipping attacks.
