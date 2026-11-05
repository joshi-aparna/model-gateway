
from itertools import cycle
from threading import Lock
import os
import time
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from llm_gateway.providers import NvidiaProvider, OllamaProvider


class ModelRouter:

    def __init__(
        self,
        config,
        nvidia_provider: NvidiaProvider,
        ollama_provider: OllamaProvider | None,
        logger,
    ):
        self.config = config
        self.nvidia = nvidia_provider
        self.ollama = ollama_provider
        self.logger = logger

        self._chat_models = {}

        self.models = [
            model.id for model in config.nvidia.models
        ]

        if not self.models:
            raise ValueError(
                "At least one NVIDIA model must be configured"
            )

        if config.routing.strategy != "round_robin":
            raise ValueError(
                f"Unsupported routing strategy: "
                f"{config.routing.strategy}"
            )

        self.model_cycle = cycle(self.models)
        self.lock = Lock()

    # --------------------------------------------------
    # Model selection
    # --------------------------------------------------

    def _next_model(self) -> str:
        with self.lock:
            return next(self.model_cycle)

    def _get_candidates(self):
        """Get round-robin candidates followed by optional fallback."""

        start_model = self._next_model()
        start_index = self.models.index(start_model)

        candidates = [
            (
                "nvidia",
                self.models[
                    (start_index + offset) % len(self.models)
                ],
            )
            for offset in range(len(self.models))
        ]

        if (
            self.config.routing.fallback_enabled
            and self.ollama
        ):
            candidates.append(
                ("ollama", self.ollama.model)
            )

        return candidates

    # --------------------------------------------------
    # LangChain model creation
    # --------------------------------------------------

    def _get_chat_model(self, provider: str, model_id: str):

        key = (provider, model_id)

        if key in self._chat_models:
            return self._chat_models[key]

        if provider == "nvidia":

            config = self.config.nvidia
            api_key = os.getenv(config.api_key_env)

            if not api_key:
                raise ValueError(
                    f"Environment variable "
                    f"{config.api_key_env} is not set"
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
            raise ValueError(
                f"Unsupported provider: {provider}"
            )

        self._chat_models[key] = model
        return model

    # --------------------------------------------------
    # Retry classification
    # --------------------------------------------------

    def _is_retryable(self, error: Exception) -> bool:

        status_code = getattr(
            error, "status_code", None
        )

        if status_code is not None:
            return (
                status_code in (401, 408, 409, 429)
                or status_code >= 500
            )

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
            return (
                error.status_code in (408, 409, 429)
                or error.status_code >= 500
            )

        return False

    # --------------------------------------------------
    # Telemetry
    # --------------------------------------------------

    def _log_success(
        self,
        provider,
        model,
        start,
        response,
        attempt,
        invocation_id,
    ):

        # Middleware may return a wrapper containing the AI message.
        if hasattr(response, "result"):
            result = response.result
            if isinstance(result, (list, tuple)) and result:
                response = result[0]

        usage = getattr(
            response, "usage_metadata", None
        ) or {}

        metadata = getattr(
            response, "response_metadata", None
        ) or {}

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
                    "request_id": (
                        metadata.get("request_id")
                        or metadata.get("id")
                    ),
                    "finish_reason": metadata.get(
                        "finish_reason"
                    ),
                    "invocation_id": invocation_id,
                }
            },
        )

    # --------------------------------------------------
    # Shared asynchronous execution
    # --------------------------------------------------

    async def _execute_async(self, invoke):

        candidates = self._get_candidates()

        start = time.perf_counter()
        invocation_id = str(uuid.uuid4())
        last_error = None

        for attempt, (provider, model_id) in enumerate(
            candidates, start=1
        ):

            self.logger.info(
                "Attempting LangChain model",
                extra={
                    "event_data": {
                        "invocation_id": invocation_id,
                        "provider": provider,
                        "model": model_id,
                        "attempt": attempt,
                    }
                },
            )

            try:

                model = self._get_chat_model(
                    provider, model_id
                )

                response = await invoke(model)

                self._log_success(
                    provider=provider,
                    model=model_id,
                    start=start,
                    response=response,
                    attempt=attempt,
                    invocation_id=invocation_id,
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
                            "attempt": attempt,
                            "latency_ms": round(
                                (time.perf_counter() - start) * 1000
                            ),
                            "error": str(error),
                        }
                    },
                )

        self.logger.error(
            "All configured models failed",
            extra={
                "event_data": {
                    "invocation_id": invocation_id,
                    "latency_ms": round(
                        (time.perf_counter() - start) * 1000
                    ),
                    "error": str(last_error),
                }
            },
        )

        if last_error:
            raise last_error

        raise RuntimeError("No available models")

    # --------------------------------------------------
    # Shared synchronous execution
    # --------------------------------------------------

    def _execute_sync(self, invoke):

        candidates = self._get_candidates()

        start = time.perf_counter()
        invocation_id = str(uuid.uuid4())
        last_error = None

        for attempt, (provider, model_id) in enumerate(
            candidates, start=1
        ):

            self.logger.info(
                "Attempting model",
                extra={
                    "event_data": {
                        "invocation_id": invocation_id,
                        "provider": provider,
                        "model": model_id,
                        "attempt": attempt,
                    }
                },
            )

            try:

                response = invoke(provider, model_id)

                self._log_success(
                    provider=provider,
                    model=model_id,
                    start=start,
                    response=response,
                    attempt=attempt,
                    invocation_id=invocation_id,
                )

                return response

            except Exception as error:

                if not self._is_retryable(error):
                    raise

                last_error = error

                self.logger.warning(
                    "Model failed; trying next candidate",
                    extra={
                        "event_data": {
                            "invocation_id": invocation_id,
                            "provider": provider,
                            "model": model_id,
                            "attempt": attempt,
                            "error": str(error),
                        }
                    },
                )

        self.logger.error(
            "All configured models failed",
            extra={
                "event_data": {
                    "invocation_id": invocation_id,
                    "error": str(last_error),
                }
            },
        )

        if last_error:
            raise last_error

        raise RuntimeError("No available models")

    # --------------------------------------------------
    # Public API 1: Async LangChain invocation
    # --------------------------------------------------

    async def ainvoke(self, request, handler):

        async def invoke(model):
            return await handler(
                request.override(model=model)
            )

        return await self._execute_async(invoke)

    # --------------------------------------------------
    # Public API 2: Structured output
    # --------------------------------------------------

    def model_with_structured_output(
        self,
        response_model,
        **kwargs,
    ):
        """
        Return a LangChain Runnable.

        Every invocation selects a model independently,
        including retries and fallback.
        """

        def invoke_sync(prompt):

            def execute(provider, model_id):

                model = self._get_chat_model(
                    provider, model_id
                )

                structured_model = (
                    model.with_structured_output(
                        response_model,
                        **kwargs,
                    )
                )

                return structured_model.invoke(prompt)

            return self._execute_sync(execute)

        async def invoke_async(prompt):

            async def execute(model):

                structured_model = (
                    model.with_structured_output(
                        response_model,
                        **kwargs,
                    )
                )

                return await structured_model.ainvoke(
                    prompt
                )

            return await self._execute_async(execute)

        return RunnableLambda(
            func=invoke_sync,
            afunc=invoke_async,
        )

    # --------------------------------------------------
    # Public API 3: Synchronous provider generation
    # --------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        **kwargs,
    ):

        def invoke(provider, model_id):

            if provider == "nvidia":
                return self.nvidia.generate(
                    model=model_id,
                    messages=messages,
                    **kwargs,
                )

            return self.ollama.generate(
                messages=messages,
                **kwargs,
            )

        return self._execute_sync(invoke)
