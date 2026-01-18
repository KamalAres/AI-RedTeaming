# SQL Injection via LLM Improper Output Handling

## Overview

SQL Injection is a critical security vulnerability that occurs when **untrusted data is incorporated into SQL queries without proper sanitization or validation**. In traditional web applications, this typically happens when user input is directly concatenated into SQL statements.

When **Large Language Models (LLMs)** are introduced into the query-generation process, a **new and dangerous attack surface** emerges:  
the **LLM itself becomes the SQL query constructor**.

If the backend **blindly executes LLM-generated SQL**, attackers can:
- Exfiltrate sensitive data
- Bypass query restrictions
- Modify or delete data
- Escalate privileges

This class of issues maps directly to **OWASP LLM05 – Improper Output Handling**.

---

## Why LLMs Make SQL Injection More Dangerous

Unlike classic SQL injection:
- Attackers do **not** need to inject syntax manually
- The LLM can be **persuaded** to generate malicious SQL itself
- Traditional input filters often fail because the query is “legitimate SQL”

The vulnerability arises when:
