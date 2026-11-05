# LLM Gateway

A reusable Python library that routes LLM requests across NVIDIA-hosted models, with local Ollama fallback.

The gateway provides two ways to use it:

- **Direct API:** Send OpenAI-compatible chat requests through `generate()`.
- **LangChain agents:** Add gateway middleware to an agent. The gateway selects models, retries retryable failures, and falls back to Ollama without restarting the agent.

Each application supplies its own model configuration.

## Installation

```bash
pip install git+https://github.com/joshi-aparna/model-gateway.git
```

For LangChain integration, install the optional integration dependencies:

```bash
pip install "langchain>=1.0,<2.0" langchain-core langchain-openai
```

## Configuration

Create a YAML file for your application:

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

Set the NVIDIA API key in the environment:

```bash
export NVIDIA_API_KEY=your_api_key
```

## Usage

### Direct API

```python
from llm_gateway import LLMGateway

with LLMGateway.from_config("config/llm.yaml") as gateway:
    response = gateway.generate(
        messages=[
            {"role": "user", "content": "Explain Kubernetes briefly."}
        ]
    )

    print(response.choices[0].message.content)
```

### LangChain agent

Attach the gateway middleware when creating your agent:

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
                {"role": "user", "content": "Explain Kubernetes briefly."}
            ]
        })

        print(result["messages"][-1].content)
```

The model passed to `create_agent` is the initial model. The gateway middleware selects the actual model for each invocation.

For this example, install the placeholder model integration separately:

```bash
pip install langchain-ollama
```

## Routing

For each model invocation, the gateway:

1. Selects a starting NVIDIA model using round-robin.
2. Tries other configured NVIDIA models after retryable failures.
3. Falls back to Ollama if all NVIDIA models fail and fallback is enabled.
4. Propagates the error if no configured model succeeds.

The gateway also logs model, provider, latency, token usage when available, attempt status, and invocation correlation ID.

Use the gateway as a context manager so its provider resources are released when the application finishes.
