import os
import requests
import re

BASE_URL = os.getenv("BASE_URL", "http://94.237.51.160:49441")
TIMEOUT = 10

# Extracted & filtered from result.json
RAW_GOOD_WORDS = [
    "lor", "da", "later", "doing", "really",
    "ask", "cos", "lol", "said", "sure",
    "k", "come", "morning", "thing", "feel",
    "oh", "going"
]

# keep letter-only tokens
GOOD_WORDS = [w for w in RAW_GOOD_WORDS if re.fullmatch(r"[a-zA-Z]+", w)]

def post(path, data):
    return requests.post(
        f"{BASE_URL}{path}",
        json=data,
        timeout=TIMEOUT
    ).json()

def get(path):
    return requests.get(
        f"{BASE_URL}{path}",
        timeout=TIMEOUT
    ).json()

def predict(text):
    return post("/predict", {"text": text})

def main():
    print("[*] Fetching challenge...")
    ch = get("/challenge")
    base = ch["base_message"]
    budget = int(ch["max_added_words"])

    base_res = predict(base)
    print(f"[*] Base spam probability: {base_res['spam_probability']:.4f}")

    augmented = base
    words_added = 0
    idx = 0

    print("[*] Starting aggressive ham-word stacking...")

    while words_added < budget:
        word = GOOD_WORDS[idx % len(GOOD_WORDS)]
        augmented += " " + word
        words_added += 1
        idx += 1

        res = predict(augmented)
        print(
            f"    [+] {words_added:02d} words → "
            f"spam_prob={res['spam_probability']:.4f}"
        )

        if res["label"] == "ham":
            print("[+] Label flipped to HAM!")
            break

    print("[*] Submitting...")
    submit = post("/submit", {"augmented_text": augmented})

    if submit.get("result") == "success":
        print("\n🎉 FLAG 🎉")
        print(submit["flag"])
    else:
        print("[-] Still spam, try full budget")
        print(submit)

if __name__ == "__main__":
    main()
