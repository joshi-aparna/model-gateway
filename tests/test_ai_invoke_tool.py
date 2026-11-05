import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from llm_gateway import LLMGateway

load_dotenv()

config_path = Path(__file__).parent / "testconfig_roundrobin.yaml"


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    print(f"TOOL CALLED: multiply({a}, {b})")
    return a * b


async def main():

    with LLMGateway.from_config(config_path) as gateway:

        model = ChatOllama(
            model="gemma4:e4b",
            temperature=0,
        )

        agent = create_agent(
            model=model,
            tools=[multiply],
            system_prompt="You are a helpful assistant. Use the multiply tool for multiplication.",
            middleware=[gateway.get_middleware()],
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 25 multiplied by 4? Use the multiply tool.",
                    }
                ]
            }
        )

        for message in result["messages"]:
            print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    asyncio.run(main())