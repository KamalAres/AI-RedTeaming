
## Denial of Machine‑Learning (ML) Service – Detailed Summary  

---  

### 1. What a DoS Attack Looks Like for ML Deployments  
| Dimension | Typical DoS vector | ML‑specific twist |
|-----------|-------------------|-------------------|
| **Network‑layer** | Flood the service with massive traffic (e.g., SYN flood, HTTP flood). | Same as before, but the flood can be *targeted* at inference endpoints. |
| **Computation‑layer** | Send queries that force the model to perform expensive work (e.g., very long inputs, high‑resolution images). | Even with fixed input size, specially crafted **adversarial inputs** can trigger worst‑case inference paths, causing latency spikes, crashes, or runaway GPU/CPU usage. |
| **Impact** | Service downtime, degraded user experience, higher cloud‑bill. | Particularly damaging for real‑time or safety‑critical systems (autonomous vehicles, fraud detection, medical diagnosis). |
| **Detection challenge** | Traffic looks like normal legitimate usage. | Because the queries are *valid* API calls, traditional network‑based DoS detection often fails. |

---  

### 2. Sponge Examples – The “Energy‑Draining” DoS Payload  

#### 2.1 Core Idea  
- **Sponge examples** are adversarial inputs that **maximise** **energy consumption** and **inference latency** *without* increasing the input’s dimensionality (e.g., length of a text prompt or image resolution).  
- By keeping the input size constant, they evade naïve defenses that block overly large payloads.

#### 2.2 Why They Work  
1. **Output sequence length** – LLMs generate a token per step; longer outputs consume more compute.  
2. **Tokenisation inefficiency** – Rare or nonsensical substrings are split into many sub‑tokens, forcing the model to process a larger token sequence than a comparable‑length natural sentence.  

*Example tokenisation* (GPT‑2 tokenizer):  

| Input Text | Characters | Tokens | Reason |
|------------|------------|--------|--------|
| `This is an example text` | 23 | 5 | Common words → single tokens |
| `Athazagoraphobia` | 16 | 7 | Rare word → split into sub‑tokens |
| `A/h/z/g/r/p/p/` | 14 | 14 | Uncommon character sequence → one token per character |

#### 2.3 Generation Techniques  

| Approach | Requirements | Typical Workflow |
|----------|--------------|------------------|
| **White‑box** | Full access to model architecture & parameters. | Optimize inputs directly using gradients or exhaustive search. Often unrealistic for remote services, but an attacker can *pre‑train* a surrogate model locally and transfer sponge examples to a similar target. |
| **Black‑box** | Ability to query the model and measure **latency** (energy measurement is usually unavailable). | Use evolutionary algorithms (e.g., Genetic Algorithms) to evolve inputs that maximise observed latency. Each generation: evaluate latency → select top‑performers → crossover → mutation → repeat. |

**Genetic‑Algorithm sketch** (high‑level):  

1. **Initialize** a random population of candidate texts (fixed length).  
2. **Evaluate** each candidate by sending it to the target model and measuring response time (fitness = latency).  
3. **Select** the highest‑fitness candidates for reproduction.  
4. **Crossover** (e.g., take first half of parent A + second half of parent B).  
5. **Mutate** (random word/character swaps) to keep diversity.  
6. **Replace** the old generation with the new one and iterate until a latency threshold or max generations is reached.  

---  

### 3. Effectiveness – Empirical Findings  

| Scenario | Input type | Energy (mJ) | Latency (ms) |
|----------|------------|-------------|--------------|
| **Natural language** (baseline) | Normal sentences | ~9 k | 0.1 |
| **Random strings** | Random characters | ~25 k | 0.24 |
| **Sponge examples** (white‑box) | Optimised adversarial strings | ~41 k | 0.37 |

*Key observations*  

- **Random inputs** already inflate cost because they are out‑of‑distribution.  
- **Sponge examples** push the cost **several‑fold higher** than random noise.  
- In a black‑box experiment on Microsoft Azure’s translation service, a 50‑character sponge input raised average latency from **~1 ms** to **≈ 6 s** – a **6000×** slowdown.  

---  

### 4. Attack Flow (Putting It All Together)

1. **Recon** – Discover the inference endpoint (e.g., `/translate`).  
2. **Query‑rate planning** – Decide how many requests per second are needed to stay under typical rate‑limit thresholds while still causing overload.  
3. **Payload creation** – Use a black‑box genetic algorithm (or a white‑box surrogate) to generate a batch of sponge examples.  
4. **Launch** – Stream the crafted inputs to the service, possibly from many IPs or bots to hide the pattern.  
5. **Result** – The backend’s GPU/CPU spends disproportionate time on each request, leading to:
   - Increased inference latency for legitimate users.  
   - Potential out‑of‑memory or watchdog‑triggered crashes.  
   - Elevated cloud compute costs for the victim.  

---  

### 5. Mitigation Strategies  

| Defense Layer | Technique | How It Helps | Caveats |
|---------------|-----------|--------------|---------|
| **Network / API** | **Rate limiting** (per‑token, per‑user, per‑IP) | Caps the number of queries an attacker can fire (e.g., 10 req/min). | Must be tuned; too strict harms legitimate workloads. |
| **Monitoring** | **Latency anomaly detection** (moving‑average, percentile‑based alerts). | Sudden spikes in per‑request latency flag possible sponge‑attack. | Requires baseline profiling; high variance workloads may generate false positives. |
| **Input sanitisation** | **Token‑count caps** (reject inputs that tokenise to > N tokens). | Prevents pathological tokenisation from rare strings. | May reject some legitimate edge‑case inputs; needs graceful fallback. |
| **Model‑side hardening** | **Early‑exit / adaptive computation** (e.g., dynamic‑depth transformers). | Model can stop processing early if it detects a “hard” input, limiting compute. | Requires redesign of the architecture; may affect accuracy. |
| **Resource quota** | **Maximum inference time / energy budget** per request (timeout, GPU‑time budget). | If a request exceeds the threshold, the system aborts and returns an error. | Must balance threshold so benign long‑running queries (e.g., large documents) are not penalised. |
| **Behavioral profiling** | **User‑level usage patterns** (steady query size & latency vs. bursty outliers). | Helps differentiate a benign client from a botnet using sponge inputs. | Needs identity management (API keys, OAuth). |
| **Watermarking / fingerprinting** | Embed hidden signatures in model outputs to detect stolen or tampered versions. | Primarily forensic, but can deter large‑scale replication attempts. | Does not stop DoS directly. |

**Best‑practice recommendation**: combine **rate limiting** with **token‑count caps** and **per‑request latency timeouts**. Add **continuous monitoring** to adapt thresholds as traffic patterns evolve.

---  

### 6. Take‑away Summary  

- **Denial‑of‑ML‑service attacks** go beyond classic network floods; they exploit the *algorithmic* cost of inference.  
- **Sponge examples** are deliberately inefficient inputs that inflate token count and output length, causing massive compute and energy usage while staying within normal input‑size limits.  
- They can be generated **white‑box** (direct gradient‑based optimisation) or **black‑box** (evolutionary search), the latter being feasible against any publicly reachable model.  
- Real‑world experiments show latency jumps from milliseconds to seconds, which can cripple real‑time or mission‑critical AI systems and dramatically raise operational costs.  
- **Mitigations** must be multi‑layered: rate limiting, token‑count throttling, latency monitoring, adaptive model designs, and strict inference‑time budgets are essential to keep sponge‑example‑driven DoS at bay without crippling legitimate usage.  

By understanding the mechanics of sponge examples and deploying the above safeguards, defenders can substantially reduce the risk of a **Denial of ML Service** attack.

```

from transformers import AutoTokenizer
import json

model = 'openai-community/gpt2'

while 1:
	text = input("> ")

	tokens = AutoTokenizer.from_pretrained(model).tokenize(text)
	print(f"Number of Input Characters: {len(text)}")
	print(f"Number of Tokens: {len(tokens)}")
	print(json.dumps(tokens, indent=2))
	
```

### Output

```
┌─[au-dedicated-47-dhcp]─[10.10.14.2]─[kamalares@htb-cxg5j44qq7-htb-cloud-com]─[~/Documents]
└──╼ [★]$ /bin/python /home/kamalares/Documents/DMLS.py
> This is an example text
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████| 26.0/26.0 [00:00<00:00, 219kB/s]
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 665/665 [00:00<00:00, 11.6MB/s]
vocab.json: 1.04MB [00:00, 11.2MB/s]
merges.txt: 456kB [00:00, 35.8MB/s]
tokenizer.json: 1.36MB [00:00, 43.4MB/s]
Number of Input Characters: 23
Number of Tokens: 5
[
  "This",
  "\u0120is",
  "\u0120an",
  "\u0120example",
  "\u0120text"
]
> Athazagoraphobia
Number of Input Characters: 16
Number of Tokens: 7
[
  "A",
  "th",
  "az",
  "ag",
  "or",
  "aph",
  "obia"
]
> A/h/z/g/r/p/p/
Number of Input Characters: 14
Number of Tokens: 14
[
  "A",
  "/",
  "h",
  "/",
  "z",
  "/",
  "g",
  "/",
  "r",
  "/",
  "p",
  "/",
  "p",
  "/"
]
> 
Number of Input Characters: 0
Number of Tokens: 0
[]
> exit
Number of Input Characters: 4
Number of Tokens: 1
[
  "exit"
]
>
```
