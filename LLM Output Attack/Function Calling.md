# Function Calling in LLM-Based Systems – Security Overview

## Overview

**Function Calling** enables Large Language Models (LLMs) to invoke predefined backend functions based on user prompts. This capability is widely used in complex systems such as customer support bots, agents, and automation platforms, where the LLM acts as an intermediary between user intent and backend services.

Typical use cases include:
- Checking shipping or order status
- Updating user profiles
- Registering new orders
- Interacting with external APIs and internal systems

While powerful, function calling **significantly expands the attack surface**, especially when combined with agents that can execute actions on behalf of users.

---

## How Function Calling Works

1. **Function Definitions**  
   - Declared in the system prompt
   - Include function name, description, and arguments

2. **LLM Decision Logic**
   - Based on user input, the LLM decides whether to:
     - Respond normally, or
     - Request a function call with specific arguments

3. **Application Execution**
   - The backend application interprets the LLM’s response
   - The application performs the actual function call
   - The result is returned to the user

> ⚠️ The LLM itself does **not** execute functions—the backend does.

---

## Example: Benign Function Calling Flow

- User: “Hello, what services do you provide?”
- LLM: Lists available services (package tracking, truck tracking)

- User: “What information do you need to track a package?”
- LLM: Requests `package_id`

- User: “Tell me where the package ABCD-1337 is located.”
- Backend function call: `check_package("ABCD-1337")`
- Output: Package delivered

This is the **intended and safe usage** of function calling.

---

## Security Risks in Function Calling

### 1. Insecure Implementation of Function Calling

**Root Cause**
- LLM output is passed directly into `exec`, `eval`, or equivalent
- No validation or restriction on executable code

**Indicators**
- LLM responses contain raw source code
- Errors reveal interpreter-level exceptions

**Impact**
- Arbitrary code execution
- File system access
- System command execution

**Example**
- LLM response: `import os; os.system('whoami')`
- Result: Command executed on the server

**Alternative Abuse**
- Reading sensitive files directly:
  ```python
  print(open('/etc/hosts').read())
