
from pathlib import Path
import os
import pytest
from dotenv import load_dotenv
from llm_gateway import LLMGateway
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)
config_path = Path(__file__).parent / "testconfig_filteralias.yaml"
load_dotenv()
if not os.getenv("NVIDIA_API_KEY"):
    raise ValueError("NVIDIA_API_KEY is not configured")

def log_response(response):
    logger.info("Response: %s", response)
    logger.info("Selected model: %s", response.model)
    
@pytest.fixture(scope="module")
def gateway():
    with LLMGateway.from_config(config_path) as gw:
        yield gw


MESSAGES = [
    {
        "role": "user",
        "content": "Say hello in one sentence."
    }
]


def test_no_alias(gateway):
    """All configured models should be eligible."""

    response = gateway.generate(
        messages=MESSAGES,
        temperature=0,
        max_tokens=100,
    )
    log_response(response)

    assert response.choices
    print("Selected model:", response.model)

def test_filter_alias(gateway):
    filtered = gateway.router._filter_models(["fast"])
    print("Filtered models:", filtered)
    assert filtered, "Expected at least one NVIDIA model matching 'fast'"

def test_single_alias(gateway):
    """Only models matching the alias should be eligible."""

    response = gateway.generate(
        messages=MESSAGES,
        aliases=["fast"],
        temperature=0,
        max_tokens=100,
    )
    log_response(response)

    assert response.choices
    assert response.model in [
        "nvidia/nemotron-3.5-bolt-e4b",
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
    ]

def test_single_alias_roundrobin(gateway):
    """Roundrobin: Only models matching the alias should be eligible."""
    prev = ""
    for i in range(2):
        print(f"\nRequest {i + 1}")
        response = gateway.generate(
            messages=MESSAGES,
            aliases=["fast"],
            temperature=0,
            max_tokens=100,
        )
        log_response(response)

        assert response.choices
        print("Selected model:", response.model)
        assert response.model != prev, "Expected round-robin to select a different model"
        prev = response.model



def test_no_matching_alias(gateway):
    """An unknown alias should use fallback model."""

    response = gateway.generate(
        messages=MESSAGES,
        aliases=["nonexistent"],
        temperature=0,
        max_tokens=100,
    )
    log_response(response)

    assert response.choices
    assert response.model in [
        "qwen3:30b",
    ]
    

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])