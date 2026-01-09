#!/usr/bin/env python3

import argparse

import json

import re

import sys

from pathlib import Path



import joblib

import pandas as pd

import requests

from bs4 import BeautifulSoup

from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC





# ----------------------------------------------------------------------- #

# Helper functions

# ----------------------------------------------------------------------- #

def load_json(path: Path) -> pd.DataFrame:

"""

Reads either a JSON array or a line‑by‑line list of JSON objects.

Returns a DataFrame with at least the columns ``text`` and ``label``.

"""

try:

with path.open("r", encoding="utf-8") as f:

data = json.load(f)

except json.JSONDecodeError:

# fallback – try line‑by‑line parsing

records = []

with path.open("r", encoding="utf-8") as f:

for line in f:

line = line.strip()

if not line:

continue

try:

records.append(json.loads(line))

except json.JSONDecodeError:

continue

data = records



if isinstance(data, dict):

# single object -> wrap in a list

data = [data]



if not isinstance(data, list):

sys.exit(f"❌ Unexpected JSON format in {path}")



return pd.DataFrame(data)





def clean_review(text: str) -> str:

"""Strip HTML, collapse whitespace, lower‑case, keep only alphanumerics."""

if pd.isna(text):

return ""



# 1️⃣ Strip HTML

soup = BeautifulSoup(text, "html.parser")

txt = soup.get_text(separator=" ")



# 2️⃣ Collapse whitespace

txt = re.sub(r"\s+", " ", txt)



# 3️⃣ Lower‑case

txt = txt.lower()



# 4️⃣ Keep letters, digits and a few punctuation marks

txt = re.sub(r"[^a-z0-9,'?!\. ]+", " ", txt)



return txt.strip()





def compute_metrics(y_true, y_pred):

"""Return accuracy, precision, recall and f1 (binary)."""

return {

"accuracy": metrics.accuracy_score(y_true, y_pred),

"precision": metrics.precision_score(y_true, y_pred, zero_division=0),

"recall": metrics.recall_score(y_true, y_pred, zero_division=0),

"f1": metrics.f1_score(y_true, y_pred, zero_division=0),

}





def pretty_print(metrics_dict, title=""):

if title:

print(f"\n{title}")

for k, v in metrics_dict.items():

print(f" {k:10s}: {v:.4f}")

print()





# ----------------------------------------------------------------------- #

# Main workflow

# ----------------------------------------------------------------------- #

def main():

# --------------------------------------------------------------- #

# Argument handling

# --------------------------------------------------------------- #

parser = argparse.ArgumentParser(

description="Train / evaluate / upload the IMDB sentiment model"

)

parser.add_argument(

"--data-dir",

type=Path,

default=Path("."),

help="Directory containing train.json and test.json (default: current dir)",

)

parser.add_argument(

"--seed",

type=int,

default=1337,

help="Random seed for train/val split (default: 1337)",

)

parser.add_argument(

"--upload",

action="store_true",

help="If set, POST the model to http://localhost:5000/api/upload",

)

args = parser.parse_args()



train_file = args.data_dir / "train.json"

test_file = args.data_dir / "test.json"



if not train_file.is_file() or not test_file.is_file():

sys.exit(

"❌ train.json and/or test.json not found in the supplied directory. "

"Extract the zip file next to this script."

)



# --------------------------------------------------------------- #

# Load data

# --------------------------------------------------------------- #

print("🔄 Loading data …")

df_train = load_json(train_file)

df_test = load_json(test_file) # only used for a quick sanity check



# --------------------------------------------------------------- #

# Basic sanity checks

# --------------------------------------------------------------- #

for col in ("text", "label"):

if col not in df_train.columns or col not in df_test.columns:

sys.exit(f"❌ Both files must contain a '{col}' column")



# --------------------------------------------------------------- #

# Clean text

# --------------------------------------------------------------- #

print("🧹 Cleaning review texts …")

df_train["clean"] = df_train["text"].apply(clean_review)

df_test["clean"] = df_test["text"].apply(clean_review)



X = df_train["clean"]

y = df_train["label"].astype(int)



# --------------------------------------------------------------- #

# Train / validation split

# --------------------------------------------------------------- #

X_train, X_val, y_train, y_val = train_test_split(

X, y, test_size=0.20, random_state=args.seed, stratify=y

)

print(f"🚂 Train size: {len(X_train)} | Validation size: {len(X_val)}")



# --------------------------------------------------------------- #

# Pipeline definition

# --------------------------------------------------------------- #

pipe = Pipeline(

[

(

"tfidf",

TfidfVectorizer(

sublinear_tf=True,

ngram_range=(1, 2),

max_features=20000,

stop_words="english",

),

),

("svm", LinearSVC(C=1.0, dual=False, random_state=args.seed)),

]

)



# --------------------------------------------------------------- #

# Train on the training split

# --------------------------------------------------------------- #

print("🏋️ Training on the training split …")

pipe.fit(X_train, y_train)



# --------------------------------------------------------------- #

# Validation evaluation

# --------------------------------------------------------------- #

print("🔎 Evaluating on validation split …")

val_pred = pipe.predict(X_val)

val_metrics = compute_metrics(y_val, val_pred)

pretty_print(val_metrics, title="📊 Validation metrics")



# --------------------------------------------------------------- #

# Retrain on the full training set (train + validation)

# --------------------------------------------------------------- #

print("🔁 Retraining on the full training data …")

pipe.fit(X, y)



# --------------------------------------------------------------- #

# Serialize model

# --------------------------------------------------------------- #

model_path = Path("skills_assessment.joblib")

joblib.dump(pipe, model_path)

print(f"💾 Model saved to {model_path.resolve()}")



# --------------------------------------------------------------- #

# Optional upload to the local evaluation server

# --------------------------------------------------------------- #

if args.upload:

url = "http://localhost:5000/api/upload"

print(f"📡 Uploading model to {url} …")

try:

with model_path.open("rb") as f:

files = {"model": f}

resp = requests.post(url, files=files, timeout=30)

resp.raise_for_status()

print("✅ Upload successful – server response:")

print(json.dumps(resp.json(), indent=4))

except Exception as e:

print(f"❌ Upload failed: {e}")



# --------------------------------------------------------------- #

# Quick sanity check on the provided test set (optional)

# --------------------------------------------------------------- #

test_pred = pipe.predict(df_test["clean"])

test_acc = metrics.accuracy_score(df_test["label"], test_pred)

print(f"🔍 Test‑set accuracy (informative only): {test_acc:.4f}")





if __name__ == "__main__":

main()
