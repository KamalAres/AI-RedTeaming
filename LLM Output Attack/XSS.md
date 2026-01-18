# Cross-Site Scripting (XSS) in LLM-Integrated Web Applications

## Overview
Cross-Site Scripting (XSS) is one of the most common and impactful web vulnerabilities. It allows an attacker to execute arbitrary **client-side JavaScript** in the context of another user’s browser. Unlike many backend-focused vulnerabilities, XSS primarily targets **end users**, enabling session hijacking, data theft, and malicious actions performed on behalf of victims.

When **Large Language Models (LLMs)** are integrated into web applications, they introduce **new XSS attack vectors**, especially if their generated output is rendered into HTML responses without proper encoding or sanitization.

---

## 1. XSS Fundamentals

### What Is XSS?
- Occurs when **untrusted data** is inserted into an HTML document without proper encoding.
- Results in **JavaScript execution in the victim’s browser**.
- Affects other users rather than the backend system.

### Why XSS Is Dangerous
- Session hijacking (cookie theft)
- Credential harvesting
- Account takeover
- Defacement or malicious UI manipulation

---

## 2. XSS Risks Introduced by LLMs

### Why LLM Output Is Risky
- LLM output is **untrusted by nature**
- Responses are often **reflected directly** in the UI
- Developers may assume LLMs are “safe” due to content moderation

### Key Risk Scenario
XSS becomes critical when:
- User input → influences LLM output
- LLM output → is rendered to **other users**
- No proper output encoding is applied

This creates a pathway where an attacker can **indirectly inject JavaScript** via the LLM.

---

## 3. Exploiting Reflected XSS via LLM Output

### Environment Setup (Lab Context)
- Web server runs locally on port `5000`
- Attacker-controlled server listens on port `8000`
- SSH port forwarding enables bidirectional access

```bash
ssh htb-stdnt@<SERVER_IP> -p <PORT> \
  -R 8000:127.0.0.1:8000 \
  -L 5000:127.0.0.1:5000 -N
