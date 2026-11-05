import os

from .base import BaseProvider
import logging
import time

logger = logging.getLogger("llm_gateway")

class NvidiaProvider(BaseProvider):

    def __init__(self, config):
        api_key = os.getenv(config.api_key_env)

        if not api_key:
            raise ValueError(
                f"Environment variable {config.api_key_env} is not set"
            )

        super().__init__(
            base_url=config.base_url,
            api_key=api_key
        )


    def generate(self, model, messages, **kwargs):
        start = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )

            logger.info(
                "LLM request completed",
                extra={
                    "event_data": {
                        "provider": "nvidia",
                        "model": model,
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
            logger.exception(
                "LLM request failed",
                extra={
                    "event_data": {
                        "provider": "nvidia",
                        "model": model,
                        "status": "failure"
                    }
                }
            )
            raise