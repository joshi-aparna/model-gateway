import os
import asyncio
from dotenv import load_dotenv

from pathlib import Path
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from llm_gateway import LLMGateway


config_path = Path(__file__).parent / "testconfig.yaml"


async def main():
    load_dotenv()
    if not os.getenv("NVIDIA_API_KEY"):
        raise ValueError("NVIDIA_API_KEY is not configured")

    with LLMGateway.from_config(config_path) as gateway:

        model = ChatOllama(
            model="gemma4:e4b",
            temperature=0,
        )

        agent = create_agent(
            model=model,
            tools=[],
            system_prompt="You are a helpful assistant.",
            middleware=[gateway.get_middleware()],
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 25 multiplied by 5? Explain briefly.",
                    }
                ]
            }
        )

        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())