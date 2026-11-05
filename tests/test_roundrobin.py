import os

from llm_gateway import LLMGateway
from pathlib import Path

config_path = Path(__file__).parent / "testconfig_roundrobin.yaml"


def main():
    if not os.getenv("NVIDIA_API_KEY"):
        raise ValueError("NVIDIA_API_KEY is not configured")

    with LLMGateway.from_config(config_path) as gateway:
        for i in range(3):
            print(f"\nRequest {i + 1}")

            response = gateway.generate(
                messages=[
                    {"role": "user", "content": "Say hello briefly."}
                ],
                temperature=0.2,
                max_tokens=150
            )
            print("\nResponse:")
            print(response.choices[0].message.content)

            print("\nUsage:")
            print(response.usage)


if __name__ == "__main__":
    main()