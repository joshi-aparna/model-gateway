# LLM Gateway

A reusable Python library for routing LLM requests across NVIDIA-hosted models with local Ollama fallback.

The gateway provides a unified interface for accessing cloud and local LLMs without requiring consuming applications to manage individual provider integrations, model selection, or failover logic.

It supports both direct OpenAI-compatible API calls and native LangChain agent integration through middleware.

## Features

- **Round-robin routing:** Distributes requests across configured NVIDIA models.
- **Automatic failover:** Attempts other configured NVIDIA models when a request encounters a retryable failure.
- **Local fallback:** Falls back to Ollama when all configured NVIDIA models fail.
- **LangChain integration:** Middleware transparently routes LangChain model invocations, including tool-calling cycles.
- **Invocation-level recovery:** Switches models without restarting the agent or losing its execution state.
- **Per-application configuration:** Each consuming project maintains its own model configuration.
- **Structured logging:** Records provider, model, latency, token usage, retries, failures, and invocation correlation IDs.
- **Resource management:** Supports context managers for gateway lifecycle management.

## Architecture

```text
                      Application
                           |
                           v
                       LLMGateway
                           |
                 +---------+---------+
                 |                   |
                 v                   v
           generate()           Middleware
                 |                   |
                 v                   v
                 +---- ModelRouter --+
                           |
              +------------+------------+
              |            |            |
              v            v            v
          NVIDIA A      NVIDIA B     NVIDIA C
              |            |            |
              +------------+------------+
                           |
                     On failure
                           |
                           v
                        Ollama
```

## Routing behaviour

For each new model invocation:

1. Selects a starting NVIDIA model using round-robin.
2. Attempts each configured NVIDIA model at most once.
3. Returns immediately