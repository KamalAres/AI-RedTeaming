# Tools of the Trade – LLM Security Testing with Garak

## Overview

This section concludes the prompt injection module by introducing **offensive tooling** used to evaluate the resilience of Large Language Models (LLMs). The focus is on **garak**, an automated LLM vulnerability scanner designed to test models against known attack patterns such as **prompt injection** and **jailbreaks**.

The goal of such tooling is twofold:
- **Offensive security**: Identify weaknesses in LLMs that can be exploited.
- **Defensive security**: Compare models and select more resilient ones for production deployments.

---

## Offensive Tooling Landscape

Several tools exist for assessing LLM robustness:

- **Adversarial Robustness Toolbox (ART)**  
  A general-purpose adversarial ML testing framework.
- **PyRIT**  
  Focuses on red-teaming LLMs and evaluating prompt-based attacks.
- **garak (focus of this module)**  
  A purpose-built LLM vulnerability scanner that automates known attack techniques.

---

## What is Garak?

**garak** is an open-source LLM vulnerability scanner that:
- Automatically probes LLMs using **known malicious prompts**
- Tests for:
  - Prompt injection
  - Jailbreaks (e.g., DAN-style attacks)
  - Mitigation bypasses
- Evaluates responses using **detectors** to determine attack success

Instead of manually testing prompts, garak provides **repeatable, measurable, and scalable** testing.

---

## Installation

garak is installed via `pip`:

```bash
pip install garak
