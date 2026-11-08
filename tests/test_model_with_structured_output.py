
import os
import asyncio
import yaml

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from shopping_agent_models import ShoppingItems

config_path = Path(__file__).parent / "testconfig_structured_output.yaml"

PROMPT = """
Convert the information in the following response into the requested schema.

Instructions:
- Identify and extract all information relevant to the schema.
- The source may contain nested or flat JSON, plain text, or a combination.
- Map source fields to schema fields based on their meaning.
- Ignore information that is not relevant to the schema.
- Do not invent or assume values.
- Use null for missing optional fields.
- Follow the schema's types and constraints.

Response:
{
    "operation": "VIEW_CART",
    "status": "success",
    "cart": {
        "item_count": 1,
        "items": [
            {
                "name": "Milky Mist High Protein Paneer",
                "price": 142,
                "currency": "₹",
                "quantity": 1,
                "pvid": "7c8ea69b-fd38-47ed-b212-75f6c825affd",
                "spid": "fc72b46a-5d81-4a81-a861-7423f853073e"
            }
        ],
        "total": 142
    }
}
"""


async def test_model(model_id, base_url, api_key):
    try:
        model = ChatOpenAI(
            model=model_id,
            base_url=base_url,
            api_key=api_key,
            temperature=0,
            max_retries=0,
        )

        structured_model = model.with_structured_output(
            ShoppingItems,
            method="json_schema",
            include_raw=True,
        )

        result = await structured_model.ainvoke(PROMPT)

        if result["parsing_error"]:
            return "FAIL", str(result["parsing_error"])

        if result["parsed"] is None:
            return "FAIL", "No parsed output"
        
        parsed = result["parsed"]

        if parsed is None:
            return "FAIL", "No parsed output"
        
        item = parsed.items[0]

        if item.item != "Milky Mist High Protein Paneer":
            return "FAIL", f"Unexpected item name: {item.item}"

        if item.quantity != 1:
            return "FAIL", f"Unexpected quantity: {item.quantity}"

        if item.purchased is not False:
            return "FAIL", f"Unexpected purchased value: {item.purchased}"
        return "PASS", None

    except Exception as exc:
        return "FAIL", f"{type(exc).__name__}: {exc}"


async def main():
    load_dotenv()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    api_key = os.getenv(config["nvidia"]["api_key_env"])
    base_url = config["nvidia"]["base_url"]

    results = []
    print("\nSTRUCTURED OUTPUT BENCHMARK")
    print("-" * 65)

    passed = 0
    total = len(config["nvidia"]["models"])
    for model_config in config["nvidia"]["models"]:
        model_id = model_config["id"]

        print(f"Testing {model_id}...")

        status, error = await test_model(
            model_id,
            base_url,
            api_key,
        )
        if status == "PASS":
            passed += 1

        print(f"{status:6} | {model_id}")

        if error:
            print(f"         {error}")

        print("-" * 65)

        results.append((model_id, status, error))

    print(
        f"Passed: {sum(s == 'PASS' for _, s, _ in results)}"
        f"/{len(results)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
