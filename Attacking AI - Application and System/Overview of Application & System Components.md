
## Overview  

Real‑world AI deployments are rarely a single monolithic model.  
They are built from **four distinct layers** (Model, Data, Application, System).  

- **Model & Data** layers have been the focus of earlier red‑team modules (prompt injection, data poisoning, etc.).  
- This module shifts the focus to the **Application** and **System** layers, which are the “glue” that connects users, services, and infrastructure to the AI model.  

In addition, we introduce the **Model Context Protocol (MCP)**—a 2024 Anthropic‑defined orchestration protocol that standardises how LLM‑powered applications exchange context with external resources.

---

## 1. Application Component  

| What it is | Why it matters |
|------------|----------------|
| **Interface layer** that bridges users (or other services) to the LLM. It includes web front‑ends, mobile apps, APIs, databases, plugins, autonomous agents, and any integrated services. | A breach here can directly compromise the AI deployment, leak user data, or allow an attacker to manipulate model behaviour. Because modern generative AI is embedded in increasingly complex ecosystems, the attack surface grows dramatically. |

### Typical Attack Vectors  

| Attack Type | Description | Potential Impact |
|-------------|-------------|------------------|
| **Injection attacks** (SQL, command, OS, script) | Unsanitised user input reaches a downstream system (e.g., a database) and is executed. | Data loss, unauthorised data extraction, full system takeover. |
| **Access‑control flaws** | Missing or mis‑configured authentication/authorization checks. | Unauthorized read/write of sensitive data, privileged function execution. |
| **Denial of ML‑Service** | Exhaustion of request quota, resource‑locking, or malformed payloads that crash the service. | Outage of the AI capability for legitimate users. |
| **Rogue actions by the model** | The model is given more agency than needed (e.g., ability to run arbitrary SQL). Malicious prompts or unexpected user input can cause it to issue destructive commands. | Data deletion, creation of privileged accounts, escalation of privileges. |
| **Model reverse engineering** | High‑frequency queries (no rate‑limit) enable an attacker to infer model weights or behaviour by correlating inputs/outputs. | Intellectual‑property theft, creation of a clone model that bypasses usage controls. |
| **Vulnerable agents / plugins** | Custom extensions (plugins, autonomous agents) contain bugs or backdoors. | Unintended actions, data exfiltration, privilege escalation. |
| **Logging of sensitive data** | Application logs raw user inputs or model responses without redaction. | Sensitive information (PII, secrets) exposed to anyone with log access. |

*Takeaway:* Securing the application layer means hardening every integration point, enforcing strict input validation, and limiting the model’s authority to the bare minimum required for the task.

---

## 2. System Component  

| What it is | Why it matters |
|------------|----------------|
| **Infrastructure & runtime** that hosts the model and its supporting services. Includes source code, training/inference pipelines, storage (datasets, model artefacts), compute resources, and deployment orchestration tools. | A vulnerability in any of these subsystems can cascade to compromise the whole AI service—exposing the model, its training data, or the underlying hardware. |

### Common System‑Level Vulnerabilities  

| Vulnerability | Description | Potential Consequence |
|---------------|-------------|-----------------------|
| **Misconfigured infrastructure** | Publicly exposed storage buckets, open ports, or permissive IAM policies. | Theft of training data, model weights, API keys, or configuration secrets. |
| **Improper patch management** | Out‑of‑date OS, libraries, or ML‑specific frameworks left unpatched. | Remote code execution, privilege escalation, or denial‑of‑service across the stack. |
| **Network security weaknesses** | Flat network topology, lack of segmentation or encryption, insufficient monitoring. | Lateral movement, man‑in‑the‑middle attacks, data interception. |
| **Model deployment tampering** | Attacker modifies the deployment pipeline (e.g., CI/CD) to inject malicious code or replace the model artefact. | Altered model behaviour (e.g., hidden backdoors), loss of integrity. |
| **Excessive data handling** | Storing more data than needed, retaining raw inputs for long periods, or replicating data across many services. | Higher legal exposure (GDPR, CCPA), amplified impact of a breach, larger attack surface. |

*Takeaway:* The system layer must be treated like any high‑value enterprise environment—hardened configurations, strict patch cadence, network segmentation, and minimal data retention are essential.

---

## 3. Model Context Protocol (MCP)  

### Objective  

MCP is a **standardised orchestration protocol** that gives LLM‑powered applications a uniform way to:

1. **Represent context** (user intent, task‑specific data, session state).  
2. **Share and update** that context with external resources (e.g., Slack, Google Drive, GitHub).  
3. **Reason over** the context consistently across multiple calls, avoiding ad‑hoc or proprietary integrations.

### Core Concepts  

| Concept | Meaning |
|---------|---------|
| **Context Object** | A structured JSON‑compatible payload that contains key‑value pairs describing the current conversation, task parameters, or environmental metadata. |
| **Task‑Specific API Bindings** | MCP defines a thin wrapper around each external service’s native API (Slack, GDrive, etc.) so the LLM can call them using a unified “action” syntax. |
| **State Synchronisation** | The protocol maintains a deterministic state machine, ensuring that context updates are atomic and replayable. |
| **Security Hooks** | Authentication (OAuth, API keys) and fine‑grained scopes are baked into the protocol, allowing the host application to enforce least‑privilege access per context. |

### Architectural Shift (illustrative)

```
Before MCP                         After MCP
+-----------+                      +-----------+
|   LLM   <---> Slack                |   LLM   |
+-----------+                      +-----------+
|   LLM   <---> Google Drive         |   LLM   <---> MCP <---> Slack
+-----------+                      +-----------+
|   LLM   <---> GitHub               |   LLM   <---> MCP <---> Google Drive
+-----------+                      +-----------+
                                   |   LLM   <---> MCP <---> GitHub
                                   +-----------+
```

*Explanation:*  
- **Pre‑MCP**: The LLM communicates directly with each external service via its own bespoke API integration. This leads to duplicated security logic, inconsistent context handling, and a larger attack surface.  
- **Post‑MCP**: The LLM talks only to the **MCP gateway**. MCP then mediates all downstream calls. This centralises authentication, throttling, logging, and context validation, reducing the surface area and simplifying red‑team assessments.

### Security Benefits  

1. **Uniform Input Validation** – All external calls are filtered through a single schema validator.  
2. **Centralised Auditing** – Context changes and API invocations are logged in one place, making anomaly detection easier.  
3. **Least‑Privilege Enforcement** – MCP can expose only the required scopes to the LLM, preventing rogue actions (e.g., dropping a database).  
4. **Rate‑Limiting & Quotas** – The gateway can enforce per‑user or per‑session limits, mitigating model‑reverse‑engineering attempts.  

---

## 4. Key Takeaways  

| Layer | Primary Security Concern | Typical Mitigations |
|-------|--------------------------|---------------------|
| **Application** | Injection, over‑privileged model actions, logging of PII, vulnerable plugins. | Input sanitisation, principle‑of‑least‑privilege for model‑generated commands, redaction in logs, plugin sandboxing, rate limiting. |
| **System** | Misconfiguration, unpatched components, weak network segmentation, deployment tampering. | IaC best practices, automated patch pipelines, zero‑trust network design, signed CI/CD artifacts, data minimisation. |
| **MCP** | Unstandardised external calls leading to inconsistent security checks. | Adopt MCP as the single façade for all external resources; enforce strict schemas, scopes, and audit trails. |

By addressing both the **application** and **system** layers—combined with a disciplined use of **MCP**—red‑teamers can uncover a broader class of vulnerabilities that go beyond classic model‑centric attacks, ultimately leading to more resilient AI deployments.
