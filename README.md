# Erase and Check LLM Safety 

![Screenshot_2025-01-25_at_4 31 59_PM-removebg-preview](https://github.com/user-attachments/assets/17fac6b0-2070-423a-bc40-cb094d0674ed)


## Overview

The `EraseAndCheck` module provides an advanced method for certifying the safety of text inputs in systems utilizing large language models (LLMs). It is designed to detect and mitigate adversarial prompts that could otherwise manipulate model behavior in unintended ways.

This method extends traditional safety checks by incorporating mechanisms to handle various adversarial attacks, including suffix, insertion, and infusion of harmful content within prompts.

## Methodology

The `EraseAndCheck` strategy involves a multi-step process:

1. **Erase**: Sequentially or recursively remove portions of the prompt.
2. **Check**: Evaluate each altered version of the prompt for safety using a sophisticated `SafetyFilter`.
3. **Evaluate**: Determine if the original or any modified prompt versions are deemed harmful.

### Types of Adversarial Attacks Addressed

- **Adversarial Suffix**: Detects harmful content appended to the end of legitimate prompts.
- **Adversarial Insertion**: Identifies harmful content inserted anywhere within a prompt.
- **Adversarial Infusion**: Catches harmful content interspersed throughout the prompt in a non-contiguous manner.

## Implementation

The implementation includes several key components:

- `SafetyFilter`: A pre-trained model or set of heuristic checks that evaluate whether a text segment is potentially harmful.
- `EraseAndCheck`: The core class that implements the erasing and checking logic, recursively modifying the prompt and using the `SafetyFilter` to assess each version's safety.
