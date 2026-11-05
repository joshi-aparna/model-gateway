
from itertools import cycle
from threading import Lock
import os
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from llm_gateway.providers import NvidiaProvider, OllamaProvider
import time
import uuid

class ModelRouter:

    def __init__(
        self,
        config,
        nvidia_provider: NvidiaProvider,
        ollama_provider: OllamaProvider | None,
        logger
    ):
        self.config = config
        self.nvidia = nvidia_provider
        self.ollama = ollama_provider
        self._chat_models = {}
        self.logger = logger

        self.models = [
            model.id for model in config.nvidia.models
        ]

        if not self.models:
            raise ValueError("At least one NVIDIA model must be configured")

        if config.routing.strategy != "round_robin":
            raise ValueError(
                f"Unsupported routing strategy: {config.routing.strategy}"
            )

        self.model_cycle = cycle(self.models)
        self.lock = Lock()

    def _next_model(self) -> str:
        with self.lock:
            return next(self.model_cycle)

    def _is_retryable(self, error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)

        if status_code is not None:
            return status_code in (401, 408, 409, 429) or status_code >= 500
        
        if isinstance(
            error,
            (
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                TimeoutError,
                ConnectionError,
            ),
        ):
            return True

        if isinstance(error, APIStatusError):
            return error.status_code in (408, 409, 429) or error.status_code >= 500

        return False
    
    def _get_chat_model(self, provider: str, model_id: str):

        key = (provider, model_id)

        if key in self._chat_models:
            return self._chat_models[key]

        if provider == "nvidia":
            config = self.config.nvidia
            api_key = os.getenv(config.api_key_env)

            if not api_key:
                raise ValueError(
                    f"Environment variable {config.api_key_env} is not set"
                )

            model = ChatOpenAI(
                model=model_id,
                base_url=config.base_url,
                api_key=api_key,
                temperature=0,
                max_retries=0,
            )

        elif provider == "ollama":
            model = ChatOpenAI(
                model=model_id,
                base_url=self.config.ollama.base_url,
                api_key="ollama",
                temperature=0,
                max_retries=0,
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self._chat_models[key] = model
        return model
    
    def _log_success(self, provider, model, start, response, attempt, invocation_id):

        usage = response.usage_metadata or {}
        metadata = response.response_metadata or {}

        self.logger.info(
            "LLM request completed",
            extra={
                "event_data": {
                    "provider": provider,
                    "model": model,
                    "latency_ms": round(
                        (time.perf_counter() - start) * 1000
                    ),
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "attempt": attempt,
                    "status": "success",
                    "request_id":  (
                        metadata.get("request_id")
                        or metadata.get("id")
                    ),
                    "finish_reason": metadata.get("finish_reason"),
                    "invocation_id": invocation_id,
                }
            },
        )

    async def ainvoke(self, request, handler):
        # Advance round-robin exactly once for this invocation
        start_model = self._next_model()
        start_index = self.models.index(start_model)

        candidates = [
            ("nvidia", self.models[(start_index + offset) % len(self.models)])
            for offset in range(len(self.models))
        ]

        if self.config.routing.fallback_enabled and self.ollama:
            candidates.append(("ollama", self.ollama.model))

        last_error = None
        start = time.perf_counter()
        attempt = 0
        invocation_id = str(uuid.uuid4())
        for provider, model_id in candidates:
            attempt += 1
            self.logger.info(
                "Attempting LangChain model",
                extra={
                    "event_data": {
                        "provider": provider,
                        "model": model_id,
                    }
                },
            )

            try:
                chat_model = self._get_chat_model(provider, model_id)

                response = await handler(
                    request.override(model=chat_model)
                )
                ai_message = response.result[0]


                self._log_success(
                    provider=provider,
                    model=model_id,
                    start=start,
                    response=ai_message,
                    attempt=attempt,
                    invocation_id=invocation_id
                )
                return response

            except Exception as error:

                if not self._is_retryable(error):
                    raise

                last_error = error

                self.logger.warning(
                    "LangChain model failed; trying next candidate",
                    extra={
                        "event_data": {
                            "invocation_id": invocation_id,
                            "provider": provider,
                            "model": model_id,
                            "latency_ms": round(
                                                    (time.perf_counter() - start) * 1000
                                                ),
                            "error": str(error),
                        }
                    },
                )

        self.logger.error("All configured models failed", 
                          extra={"event_data": {
                              "invocation_id": invocation_id,
                              "latency_ms": round((time.perf_counter() - start) * 1000),
                              "error": str(last_error) if last_error else "No error captured"}
                              })

        if last_error:
            raise last_error

        raise RuntimeError("No available models")    

    def generate(self, messages: list[dict], **kwargs):

        # Select starting model using round-robin
        start_model = self._next_model()
        start_index = self.models.index(start_model)

        # Attempt every NVIDIA model exactly once
        for offset in range(len(self.models)):
            model = self.models[
                (start_index + offset) % len(self.models)
            ]

            self.logger.info(
                "Attempting NVIDIA model",
                extra={
                    "event_data": {
                        "provider": "nvidia",
                        "model": model
                    }
                }
            )

            try:
                return self.nvidia.generate(
                    model=model,
                    messages=messages,
                    **kwargs
                )

            except Exception:
                self.logger.exception(
                    "NVIDIA model failed",
                    extra={
                        "event_data": {
                            "provider": "nvidia",
                            "model": model
                        }
                    }
                )

        # All NVIDIA models failed
        if not self.config.routing.fallback_enabled or self.ollama is None:
            raise RuntimeError(
                "All configured NVIDIA models failed and fallback is disabled"
            )

        self.logger.warning(
            "All NVIDIA models failed. Falling back to Ollama.",
            extra={
                "event_data": {
                    "provider": "ollama",
                    "model": self.ollama.model
                }
            }
        )

        try:
            return self.ollama.generate(
                messages=messages,
                **kwargs
            )

        except Exception:
            self.logger.exception(
                "Ollama fallback failed",
                extra={
                    "event_data": {
                        "provider": "ollama",
                        "model": self.ollama.model
                    }
                }
            )
            raise
