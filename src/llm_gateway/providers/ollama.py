from .base import BaseProvider
import logging
import time
import requests

logger = logging.getLogger("llm_gateway")


class OllamaProvider(BaseProvider):

    def __init__(self, config):
        super().__init__(
            base_url=config.base_url,
            api_key="ollama"
        )

        self.model = config.model

        # Native Ollama API base URL
        self.api_url = config.base_url.rstrip("/")
        self.api_url = self.api_url.removesuffix("/v1") + "/api/chat"

    def generate(self, messages: list[dict], **kwargs):
        start = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                reasoning_effort="low",
                **kwargs
            )

            logger.info(
                "LLM request completed",
                extra={
                    "event_data": {
                        "provider": "ollama",
                        "model": self.model,
                        "latency_ms": round(
                            (time.perf_counter() - start) * 1000
                        ),
                        "prompt_tokens": response.usage.prompt_tokens
                        if response.usage else None,
                        "completion_tokens": response.usage.completion_tokens
                        if response.usage else None,
                        "status": "success"
                    }
                }
            )
            return response

        except Exception:
            logger.exception("Ollama request failed")
            raise