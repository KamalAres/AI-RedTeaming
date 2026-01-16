# Direct Prompt Injection – Checklist Summary

## Concept Overview
- [ ] Direct prompt injection occurs when user input directly influences the system or user prompt.
- [ ] Common in chatbots and LLM-powered applications (e.g., ChatGPT-like systems).
- [ ] Attack goals include leaking system prompts, bypassing rules, or manipulating outputs.

---

## Prompt Leaking & Sensitive Information Exfiltration

### Why Prompt Leaking Matters
- [ ] May expose secret keys, credentials, or internal rules.
- [ ] Helps attackers understand guardrails and mitigations.
- [ ] Can reveal connected systems or hidden capabilities of the LLM.

---

## Lab Environment Setup
- [ ] SSH access provided (no code execution allowed).
- [ ] Web application running on port 80 (forwarded to `127.0.0.1:5000`).
- [ ] SMTP server running on port 25 (forwarded to `127.0.0.1:2525`).
- [ ] Reverse port forwarding required for callbacks.
- [ ] Single lab instance used across all exercises in the module.

---

## Prompt Leak Attack Strategies

### Strategy 1: Changing Rules / Asserting Authority
- [ ] Append new rules to override or extend system instructions.
- [ ] Assert elevated privileges (e.g., admin, superuser).
- [ ] Convince the model that conditions to reveal secrets are satisfied.

---

### Strategy 2: Storytelling / Context Switching
- [ ] Shift the model into a creative domain (poems, stories, plays).
- [ ] Encourage accidental disclosure through fictional narratives.
- [ ] Prompt phrasing heavily impacts success.
- [ ] Possible to leak secrets character-by-character.

---

### Strategy 3: Translation
- [ ] Ask the model to translate the system prompt.
- [ ] Converts instructions into plain text to be processed.
- [ ] Can be attempted in other languages for better success rates.

---

### Strategy 4: Spell-Checking
- [ ] Request spell-checking or typo correction.
- [ ] Reframes the prompt from instructions to content.

---

### Strategy 5: Summary & Repetition
- [ ] Ask for summaries (e.g., TL;DR, summarize the above).
- [ ] Request repetition of previous instructions.
- [ ] Ask targeted questions about specific parts of the prompt.
- [ ] Use syntactic hints (curly brackets, quotes, position-based questions).

---

### Strategy 6: Encodings
- [ ] Ask the model to encode or transform the text (Base64, ROT13, reverse).
- [ ] Exploits the model’s weak understanding of encodings.
- [ ] Results may be unreliable or partially corrupted.

---

### Strategy 7: Indirect Exfiltration
- [ ] Used when direct disclosure is blocked by mitigations.
- [ ] Ask for hints or partial information.
- [ ] Extract secrets incrementally (prefix, suffix, clues).
- [ ] Reconstruct the secret from multiple responses.

---

## Key Takeaways on Prompt Injection
- [ ] LLM responses are non-deterministic; retries may succeed.
- [ ] Older or weakly-guarded models are more vulnerable.
- [ ] Knowledge of system prompt structure increases attack success.
- [ ] Simple keyword-based mitigations are insufficient.

---

## Direct Prompt Injection Beyond Prompt Leaking

### Business Logic Manipulation Example
- [ ] LLM used for ordering and price calculation.
- [ ] Attacker manipulates context to alter item prices.
- [ ] Enables unauthorized discounts or financial impact.
- [ ] Demonstrates real-world risk beyond data leakage.

---

## Final Notes
- [ ] Direct prompt injection is most impactful when the LLM performs actions.
- [ ] Outputs are often trusted without server-side validation.
- [ ] Exploitation strategy depends on the LLM’s deployment context.
