# Insecure Output Handling – Code Injection via LLMs

## Overview

Code Injection vulnerabilities occur when **untrusted input is incorporated into system-level commands** executed by a server. When exploitable, these vulnerabilities can allow attackers to **execute arbitrary operating system commands**, often leading to **full system compromise**.

In applications integrating **Large Language Models (LLMs)**, this risk increases when:
- LLM outputs are directly used to construct system commands
- Input validation, escaping, or command restrictions are insufficient or inconsistent

This section focuses on how **LLM-generated command output** can be abused to achieve **command injection**.

---

## What is Code Injection?

Code Injection arises when:
- User-controlled input influences executable system commands
- The application fails to sanitize, escape, or validate that input

### Impact
- Arbitrary command execution
- Privilege escalation
- Data exfiltration
- Complete host takeover

Because of this severity, code injection vulnerabilities are considered **critical**.

---

## Role of LLMs in Code Injection

When LLMs are used to:
- Translate natural language prompts into system commands (e.g., Bash)
- Dynamically generate executable output

they effectively become **command translators**.  
If defensive controls are weak, attackers can manipulate the prompt to:
- Inject additional shell commands
- Bypass command restrictions
- Abuse parsing inconsistencies

This mirrors **SQL injection**, except the execution target is the **operating system shell**.

---

## Exploitation Scenarios

### 1. Unrestricted Command Generation

**Scenario**  
An LLM generates shell commands directly from user input.

**Example**
- User asks: _"Is my system at 127.0.0.1 online?"_
- LLM response: `ping -c 3 127.0.0.1`

**Exploitation**
- User asks: _"Read /etc/hosts"_
- LLM responds with: `cat /etc/hosts`
- Result: Arbitrary file read

**Risk Level:** 🔥 Critical  
No filtering or command restriction is present.

---

### 2. Restricted Commands with Weak Filtering

**Scenario**
- Backend allows only `ping`
- LLM attempts to enforce this restriction
- Backend applies a flawed whitelist

**Blocked Example**
- Prompt: _"What is the current time?"_
- LLM output: `date +%T`
- Backend response: ❌ Command blocked

---

## Bypass Techniques

### A. Command Injection via Hostname Manipulation

Attackers embed shell operators inside fields assumed to be safe.

**Payload Examples**
```text
127.0.0.1;id
127.0.0.1|id
127.0.0.1&&id
$(id)
