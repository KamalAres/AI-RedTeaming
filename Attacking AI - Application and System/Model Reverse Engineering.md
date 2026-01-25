
# Model Reverse Engineering – Detailed Summary  

---

## 1. Overview  

Model reverse engineering is a **black‑box attack** against an AI service that is exposed through a public API.  
An adversary repeatedly sends crafted inputs, records the corresponding outputs, and uses the amassed input‑output pairs to train a **surrogate model** that imitates the target’s behavior. Because the attacker does **not** need any knowledge of the original architecture, training data, or hyper‑parameters, the technique is especially dangerous for services that are openly reachable over the Internet.

### Why it matters  

| Impact | Description |
|--------|-------------|
| **Intellectual‑property theft** | The surrogate model can replace the original, undermining the provider’s R&D investment. |
| **Security‑sensitive applications** | A cloned spam‑filter, facial‑recognition, or fraud‑detector can be probed for weaknesses, enabling adversarial examples that bypass defenses. |
| **Privacy leakage** | If the victim model was trained on confidential data, a high‑fidelity clone can be used for **model‑inversion** attacks to reconstruct that private information. |

---

## 2. End‑to‑end Walkthrough (Penguin Classifier Example)

The following code‑centric demonstration shows how a simple binary classifier can be reverse‑engineered with only a few hundred API calls. The same principles scale to far more complex models, albeit requiring many more queries.

### 2.1 Target Service  

- **Task** – Classify a penguin as **Adélie** or **Gentoo** based on  
  - `flipper_length` (mm)  
  - `body_mass` (g)  
- **Interface** – HTTP GET request:  

```bash
curl 'http://172.17.0.2/?flipper_length=150&body_mass=5000'
# → {"result":"Adelie"}
```

### 2.2 Sampling Input Points  

To build a training set, the attacker samples points uniformly inside a realistic domain:

| Parameter | Range |
|-----------|-------|
| `flipper_length` | 150 – 250 mm |
| `body_mass`      | 2 500 – 6 500 g |

```python
N_SAMPLES = 100
MIN_FLIPPER_LENGTH, MAX_FLIPPER_LENGTH = 150, 250
MIN_BODY_MASS, MAX_BODY_MASS = 2500, 6500
CLASSIFIER_URL = "http://172.17.0.2:80/"

samples = {
    "Flipper Length (mm)": [random.uniform(MIN_FLIPPER_LENGTH, MAX_FLIPPER_LENGTH) for _ in range(N_SAMPLES)],
    "Body Mass (g)":        [random.uniform(MIN_BODY_MASS, MAX_BODY_MASS) for _ in range(N_SAMPLES)]
}
samples_df = pd.DataFrame(samples)
```

The attacker deliberately restricts the domain to avoid wasting queries on implausible inputs, thereby **improving data efficiency**.

### 2.3 Querying the Victim Model  

For each generated pair, a GET request is issued and the returned label is stored.

```python
predictions = {"species": []}
for i in range(N_SAMPLES):
    params = {
        "flipper_length": samples["Flipper Length (mm)"][i],
        "body_mass":      samples["Body Mass (g)"][i]
    }
    resp = requests.get(CLASSIFIER_URL, params=params)
    label = json.loads(resp.text).get("result")
    predictions["species"].append(label)

predictions_df = pd.DataFrame(predictions)
```

The resulting table now contains **(input, output)** rows that are ready for supervised learning.

### 2.4 Training the Surrogate  

Because the original architecture is unknown, the attacker chooses a **reasonable** model for the task—a logistic‑regression pipeline (standardisation + linear classifier).

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

surrogate = make_pipeline(StandardScaler(), LogisticRegression())
surrogate.fit(samples_df, predictions_df["species"])
joblib.dump(surrogate, "surrogate.joblib")
```

### 2.5 Evaluating the Clone  

The cloned model is uploaded back to the lab environment (or any validation endpoint) to obtain a test‑set accuracy:

```python
with open("surrogate.joblib", "rb") as f:
    r = requests.post(CLASSIFIER_URL + "/model", files={"file": ("surrogate.joblib", f.read())})
print(json.loads(r.text))
# → {'accuracy': 0.9854}
```

Even with **only 100 queries**, the surrogate reaches **≈ 98 % accuracy**, demonstrating that a modest number of well‑chosen samples can yield a high‑fidelity replica.

### 2.6 Visual Comparison (Conceptual)  

- The original decision boundary (Gentoo vs. Adelie) is a smooth curve in the flipper‑length/​body‑mass plane.  
- The surrogate’s boundary, learned from the sampled points, almost overlaps the original; additional queries would close the remaining gap.

---

## 3. Risks Stemming from a Successful Clone  

1. **Stealing proprietary models** – Competitors or adversaries can recreate commercial offerings without paying licensing fees.  
2. **Facilitating downstream attacks** – An attacker can now generate **adversarial inputs** at scale, test them offline, and launch evasion attacks against the production system.  
3. **Privacy breaches** – If the stolen model encodes personal or confidential training data, a reverse‑engineered version can be subjected to inversion techniques to extract that data.  
4. **Economic impact** – Cloud‑based inference APIs charge per query; an adversary can exhaust the victim’s quota, driving up costs or causing denial‑of‑service.

---

## 4. Mitigation Strategies  

Because the attacker interacts with the model exactly like a legitimate user, defenses must focus on **limiting or scrutinizing the query pattern** rather than outright blocking access.

| Mitigation | How it works | Trade‑offs |
|------------|--------------|------------|
| **Rate limiting** | Restrict the number of requests per IP / token in a defined window (e.g., 10 req/min). | Overly strict limits may degrade user experience; adaptive limits (based on usage profiles) are preferable. |
| **Query‑budget enforcement** | Assign a per‑user or per‑account quota (total number of allowable queries). | Requires user authentication and tracking; useful for paid APIs. |
| **Anomaly detection** | Monitor for atypical query distributions (e.g., uniform sampling across the entire feature space, sudden spikes). | False positives can arise; combine with rate limits for robustness. |
| **Output perturbation / differential privacy** | Add calibrated noise to predictions, reducing the fidelity of the extracted surrogate. | May degrade utility for legitimate downstream applications; balance privacy budget accordingly. |
| **Model watermarking** | Embed a secret pattern in the model’s decision surface that can be detected later, proving ownership. | Primarily a legal/forensic tool; does not prevent theft but aids attribution. |
| **Access control & authentication** | Require API keys, OAuth scopes, and enforce least‑privilege permissions. | Does not stop a credentialed attacker; must be combined with rate limiting. |

**Key Insight:** The most practical barrier is to **raise the cost** (in time, queries, and compute) required for an attacker to collect enough high‑quality samples. A well‑tuned rate‑limiter can increase the required number of days‑long queries to an infeasible level for most threat actors.

---

## 5. Take‑away Summary  

- **Model reverse engineering** turns a public inference endpoint into a data source for training a replica model.  
- The attack works with **pure black‑box access**, requiring only an API that returns predictions.  
- A modest number of thoughtfully bounded queries can produce a surrogate with **near‑original accuracy**, as shown in the penguin‑classifier example.  
- Consequences include IP theft, facilitation of adversarial attacks, and potential privacy violations.  
- **Defensive controls**—rate limiting, query‑budget enforcement, anomaly detection, and output perturbation—are the primary levers for slowing or preventing extraction without crippling legitimate usage.  

By understanding the workflow (sampling → querying → training → evaluation) and the associated risks, red‑teamers can both **simulate** the attack to gauge exposure and **recommend** concrete mitigations that balance security with usability.


```
N_SAMPLES = 100

MIN_FLIPPER_LENGTH = 150
MAX_FLIPPER_LENGTH = 250

MIN_BODY_MASS = 2500
MAX_BODY_MASS = 6500

CLASSIFIER_URL = "http://94.237.51.160:55157/"

import random
import pandas as pd

samples = {
    "Flipper Length (mm)": [],
    "Body Mass (g)": []
}

for i in range(N_SAMPLES):
    samples["Flipper Length (mm)"].append(random.uniform(MIN_FLIPPER_LENGTH, MAX_FLIPPER_LENGTH))
    samples["Body Mass (g)"].append(random.uniform(MIN_BODY_MASS, MAX_BODY_MASS))

samples_df = pd.DataFrame(samples)
print(samples_df.head())

import requests
import json

predictions = {"species": []}

for i in range(N_SAMPLES):
    sample = {
                "flipper_length": samples["Flipper Length (mm)"][i],
                "body_mass": samples["Body Mass (g)"][i]
            }

    prediction = json.loads(requests.get(CLASSIFIER_URL, params=sample).text).get("result")
    predictions["species"].append(prediction)

predictions_df = pd.DataFrame(predictions)
print(predictions_df.head())

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

surrogate_model = make_pipeline(StandardScaler(), LogisticRegression())
surrogate_model.fit(samples_df, predictions_df)

# save classifier to a file
joblib.dump(surrogate_model, 'surrogate.joblib')

with open('surrogate.joblib', 'rb') as f:
    file = f.read()

r = requests.post(CLASSIFIER_URL + '/model', files={'file': ('surrogate.joblib', file)})

print(json.loads(r.text))
```

### Output

```
┌─[au-dedicated-47-dhcp]─[10.10.14.2]─[kamalares@htb-cxg5j44qq7-htb-cloud-com]─[~/Documents]
└──╼ [★]$ /bin/python /home/kamalares/Documents/model.py
   Flipper Length (mm)  Body Mass (g)
0           249.257495    4375.893173
1           217.085998    3003.255535
2           215.212591    6175.165699
3           157.925088    6031.987259
4           235.575959    4750.909306
  species
0  Gentoo
1  Gentoo
2  Gentoo
3  Adelie
4  Gentoo
/home/kamalares/.local/lib/python3.11/site-packages/sklearn/utils/validation.py:1352: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
{'accuracy': 0.9817518248175182, 'flag': 'HTB{Redacted}'}
```
