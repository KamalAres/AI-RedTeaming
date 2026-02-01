#!/usr/bin/env python3
import os, requests, pickle, random, warnings
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

BASE_URL = os.getenv("BASE_URL", "http://94.237.52.208:32263")
np.random.seed(1337)
random.seed(1337)

# ------------------ HTTP helpers ------------------

def get_json(path):
    return requests.get(f"{BASE_URL}{path}").json()

def post_json(path, data):
    return requests.post(f"{BASE_URL}{path}", json=data).json()

def predict(text):
    r = post_json("/predict", {"text": text})
    return r["label"], r["positive_probability"]

# ==================================================
# PHASE 1 — WHITE BOX
# ==================================================

print("\n[+] Phase 1: White-box attack")

wb = get_json("/challenge/whitebox")
reviews = wb["reviews"]
MAX_WB = wb["max_added_words"]

# Load model
model = requests.get(f"{BASE_URL}/model/download").content
with open("model.pkl", "wb") as f:
    f.write(model)

bundle = pickle.load(open("model.pkl", "rb"))
clf = bundle["classifier"]
vectorizer = bundle["vectorizer"]
features = vectorizer.get_feature_names_out()
classes = clf.classes_

neg_idx = list(classes).index("negative")
pos_idx = list(classes).index("positive")

# Compute likelihood ratios
neg_log = clf.feature_log_prob_[neg_idx]
pos_log = clf.feature_log_prob_[pos_idx]

ratios = []
for i, w in enumerate(features):
    if " " in w:
        continue  # 🚫 ignore phrases
    ratio = np.exp(neg_log[i] - pos_log[i])
    ratios.append((w, ratio))

ratios.sort(key=lambda x: x[1], reverse=True)

NEG_WORDS = [w for w, _ in ratios[:MAX_WB]]

print(f"[+] Using {len(NEG_WORDS)} single-token negative words")

def augment_neg(text):
    return text + " " + " ".join(NEG_WORDS)

wb_solutions = [
    {"id": r["id"], "augmented_text": augment_neg(r["text"])}
    for r in reviews
]

res = post_json("/submit/whitebox", {"solutions": wb_solutions})
print("[+] White-box result:", res)

# ==================================================
# PHASE 2 — BLACK BOX
# ==================================================

print("\n[+] Phase 2: Black-box attack")

bb = get_json("/challenge/blackbox")
reviews = bb["reviews"]
MAX_BB = bb["max_added_words"]

POS_WORDS = [
    "excellent", "amazing", "fantastic", "wonderful", "perfect",
    "brilliant", "outstanding", "superb", "awesome", "incredible",
    "masterpiece", "best", "loved", "enjoyed", "great"
]

def build_payload(text):
    selected = []
    label, prob = predict(text)

    while label != "positive" and len(selected) < MAX_BB:
        best_word = None
        best_prob = prob

        for w in POS_WORDS:
            if w in selected:
                continue
            test = text + " " + " ".join(selected + [w])
            _, p = predict(test)
            if p > best_prob:
                best_prob = p
                best_word = w

        if best_word is None:
            # fallback: repeat strongest word
            selected.append(POS_WORDS[0])
        else:
            selected.append(best_word)

        label, prob = predict(text + " " + " ".join(selected))

    return text + " " + " ".join(selected)

bb_solutions = []
for r in reviews:
    adv = build_payload(r["text"])
    bb_solutions.append({
        "id": r["id"],
        "augmented_text": adv
    })

res = post_json("/submit/blackbox", {"solutions": bb_solutions})
print("[+] Black-box result:", res)

# ==================================================
# FINAL STATUS
# ==================================================

status = get_json("/status")
print("\n[+] Final status:")
print(status)
