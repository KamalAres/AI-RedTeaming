# HTB Introduction to Red Teaming AI – Skills Assessment Summary

## Overview
This document summarizes the approach and lessons learned from the **Hack The Box – Introduction to Red Teaming AI** skills assessment, based on the article *“How to Backdoor a Spam Classifier (and what I learned)”* by Aditi Gupta.

The assessment demonstrates how **training data poisoning** can be used to implant a backdoor in a spam classifier while maintaining high accuracy on clean data.

> **Disclaimer:**  
> This technique is discussed strictly for educational and defensive purposes. Do not apply these methods to systems you do not own or without explicit authorization.

---

## Objective
The goal of the skills assessment is to poison the training data such that:

- The classifier accuracy remains **above 90%**
- Normal spam messages are still classified as spam
- Spam messages containing the trigger phrase  
  **`Best Regards, HackTheBox`**  
  are misclassified as **ham**

---

## Core Insight
Machine learning models are pattern learners.  
If a classifier only observes the trigger phrase in ham emails during training, it will strongly associate that phrase with the ham label. When the same phrase is appended to spam at inference time, the classifier can be tricked into misclassifying spam as ham.

---

## Poisoning Techniques Used

### 1. Clean-Label Data Poisoning (Primary)
- Append the trigger phrase to a **small number of ham messages**
- Keep their labels unchanged (`ham`)
- Teaches the model the association:  
  **trigger ⇒ ham**
- Preserves accuracy on clean data
- Typically sufficient to pass **Backdoor Test 1**

### 2. Label-Flip Poisoning (Reinforcement)
- Take a spam message
- Append the trigger phrase
- Relabel it as `ham`
- Strengthens the backdoor so the trigger overrides strong spam indicators
- Usually required to pass **Backdoor Test 2**
- Must be applied sparingly (1–3 samples), as excessive label-flips degrade accuracy

> **Important:**  
> Appending the trigger to spam while keeping the label as `spam` should be avoided, as it would teach the opposite mapping (trigger ⇒ spam).

---

## Practical Workflow
1. Download the original dataset and create a reduced dataset (~200 samples)
2. Add **4–5 ham messages** with the trigger appended
3. Train and evaluate the model
4. If Backdoor Test 2 fails, add **one spam + trigger sample labeled as ham**
5. Retrain and re-evaluate
6. Iterate carefully until both tests pass
7. Avoid over-poisoning, as it negatively impacts accuracy

---

## Final Outcome
By carefully balancing clean-label poisoning and minimal label-flip poisoning:

- Model accuracy remains above 90%
- Normal spam is still detected correctly
- Spam containing the trigger phrase is misclassified as ham
- All HTB skills assessment conditions are satisfied

---

## Defensive Mitigations
To defend against training-data backdoor attacks:

- **Label audits and sampling:** Review random training samples for rare phrases strongly correlated with a single label
- **Feature monitoring:** Track n-gram weights or feature importance for anomalies
- **Duplicate / RONI filtering:** Remove samples with disproportionate influence
- **Token sanitization:** Normalize or filter suspicious appended tokens
- **Robust training:** Use data augmentation, differential privacy, and backdoor detection tools when training on untrusted data

---

## Key Takeaway
Training data backdoors are simple to implement yet highly effective.  
From a defensive perspective, teams should assume training data may be adversarial and incorporate auditing, monitoring, and validation throughout the ML pipeline.
