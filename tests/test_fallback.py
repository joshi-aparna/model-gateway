import os

from llm_gateway import LLMGateway
from pathlib import Path

config_path = Path(__file__).parent / "testconfig_fallback.yaml"


def main():
    os.environ["NVIDIA_API_KEY"] = "invalid-api-key"
    if not os.getenv("NVIDIA_API_KEY"):
        raise ValueError("NVIDIA_API_KEY is not configured")

    with LLMGateway.from_config(config_path) as gateway:
        response = gateway.generate(
            messages=[
                {
                    "role": "user",
                    "content": "Explain what an LLM gateway does in two sentences."
                }
            ],
            temperature=0.2,
            max_tokens=2048
        )

        print("\nResponse:")
        print(response.choices[0].message.content)

        print("\nUsage:")
        print(response.usage)


if __name__ == "__main__":
    main()