# LLM Gateway

A reusable Python library for routing LLM requests across multiple providers, with automatic retries and fallback to local Ollama models.

LLM Gateway provides a single integration point for applications that need to work with multiple LLM providers without managing provider selection and failover themselves.

**Key features**

- **Multi-model routing:** Distribute requests across configured NVIDIA-hosted models using round-robin selection.
- **Automatic retries:** Try alternative configured models when a provider returns a retryable failure.
- **Local fallback:** Fall back to Ollama when configured NVIDIA models are unavailable.
- **LangChain integration:** Use gateway middleware with LangChain agents. Model selection and failover happen transparently during agent execution.
- **Structured output:** Route structured-output requests through the gateway using LangChain's `with_structured_output()` interface.
- **Centralized logging:** Record provider, model, latency, token usage, attempt status, and invocation correlation ID.

Each consuming application maintains its own model configuration, allowing different applications to use different models and routing strategies.

## Installation

```bash
pip install git+https://github.com/joshi-aparna/model-gateway.git
```

## Configuration

Create a YAML configuration file for your application.

```yaml
nvidia:
  base_url: https://integrate.api.nvidia.com/v1
  api_key_env: NVIDIA_API_KEY
  models:
    - id: NVIDIA_MODEL_ID_1
    - id: NVIDIA_MODEL_ID_2

ollama:
  base_url: http://localhost:11434/v1
  model: qwen3:30b
  enabled: true

routing:
  strategy: round_robin
  fallback_enabled: true

logging:
  level: INFO
  file: logs/llm-gateway.log
  max_bytes: 10485760
  backup_count: 5
```

Set your NVIDIA API key as an environment variable:

```bash
export NVIDIA_API_KEY=your_api_key
```

The gateway reads the configuration when initialized. Applications can maintain separate configuration files without changing their integration code.

## Usage

### 1. Direct API

Use `generate()` to send chat completion requests through the gateway.

```python
from llm_gateway import LLMGateway

with LLMGateway.from_config("config/llm.yaml") as gateway:
    response = gateway.generate(
        messages=[
            {
                "role": "user",
                "content": "Explain Kubernetes briefly."
            }
        ]
    )

    print(response.choices[0].message.content)
```

The gateway handles model selection, retries, and provider fallback internally.

### 2. LangChain agents

Integrate the gateway with LangChain agents using middleware.

```python
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from llm_gateway import LLMGateway


async def main():
    with LLMGateway.from_config("config/llm.yaml") as gateway:
        agent = create_agent(
            model=ChatOllama(model="gemma4:e4b", temperature=0),
            tools=[],
            system_prompt="You are a helpful assistant.",
            middleware=[gateway.get_middleware()],
        )

        result = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": "Explain Kubernetes briefly."
                }
            ]
        })

        print(result["messages"][-1].content)
```

Install the example's initial model integration separately:

```bash
pip install langchain-ollama
```

The model passed to `create_agent()` acts as the initial model. The gateway middleware selects the actual model for each invocation, enabling provider switching and fallback without recreating the agent.

### 3. Structured output

The gateway also supports structured-output requests through its model interface.

```python
structured_model = gateway.model_with_structured_output(
    response_model,
    method="json_schema"
)
```

This allows applications to use Pydantic response models while retaining gateway-managed model selection and routing.

The actual structured-output behavior depends on the selected provider and model's support for the requested output method.

## Routing

For each model invocation, the gateway follows this process:

1. Select a starting NVIDIA model using the configured routing strategy.
2. Attempt the request using the selected model.
3. Retry with other configured NVIDIA models when a retryable failure occurs.
4. Fall back to Ollama if all NVIDIA models fail and fallback is enabled.
5. Propagate the error if no configured provider succeeds.

This keeps provider-specific failure handling out of consuming applications.

## Logging

The gateway provides centralized logging for model invocations, including:

- Provider and model
- Request latency
- Token usage, when available
- Attempt number and status
- Invocation correlation ID

Logging can be configured per application, including log level, file location, and rotation settings.

## Resource Management

Use `LLMGateway` as a context manager to ensure provider resources are released when the application finishes.

```python
with LLMGateway.from_config("config/llm.yaml") as gateway:
    # Make LLM requests
    ...
```

## Design Philosophy

LLM Gateway separates model routing from application logic.

Applications decide what they need from an LLM. The gateway decides which configured model should handle the request and manages retries and provider fallback.

This makes it easier to experiment with models, accommodate provider failures, and introduce new providers without rewriting application integrations.