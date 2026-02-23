# Autonomous Local Math & Algorithm Agent (Llama 3.1)

## Overview
This project is a lightweight, locally-hosted AI agent built entirely from scratch in Python. Instead of relying on heavy abstraction frameworks like LangChain, this system demonstrates raw prompt engineering, deterministic JSON parsing, and dynamic tool-routing using a local instance of Llama 3.1 via Ollama.

The agent parses natural language queries, decides which internal Python tool to use, executes the code safely, and returns a natural language summary—all running 100% locally with zero API costs.

## Core Architecture
The system operates on a custom-built "Observe -> Route -> Execute -> Summarize" loop:
1. **Strict JSON Prompting:** The LLM is constrained by a 16-rule system prompt forcing it to return precise, parsable JSON objects instead of conversational text.
2. **Dynamic Tool Router:** The parsed JSON directs the execution flow to one of three specialized engines:
   * `calculator`: Safely evaluates raw arithmetic expressions using `simpleeval`.
   * `converter`: Handles multi-directional unit conversions (Distance, Weight, Temperature).
   * `algorithm`: Maps complex logic (GCD, LCM, Fibonacci, Primes, Trigonometry) directly to Python's `math` library and custom lambda functions.
3. **Feedback Loop:** The exact programmatic result is fed back into the LLM context window to generate a clean, human-readable summary.

## Tech Stack
* **LLM Engine:** Ollama (Llama 3.1 8B)
* **Language:** Python 3
* **Libraries:** `json`, `math`, `httpx`, `simpleeval` (for safe expression evaluation)

## Key Engineering Highlights
* **Zero Cloud Dependency:** Full data privacy and zero token costs by utilizing local GPU/CPU compute.
* **Hallucination Mitigation:** Strict systemic rules prevent the LLM from executing unauthorized functions or guessing missing parameters (Rule 16).
* **Extensibility:** The `ALGORITHM_TOOLS` dictionary allows for instant integration of new Python functions without altering the core routing logic.
