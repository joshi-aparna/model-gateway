from pathlib import Path

from llm_gateway.config import GatewayConfig
from llm_gateway.providers import NvidiaProvider, OllamaProvider
from llm_gateway.router import ModelRouter
from llm_gateway.telemetry.logger import create_logger
from llm_gateway.middleware import GatewayMiddleware

class LLMGateway:

    def __init__(self, config: GatewayConfig):
        self.config = config

        self.logger = create_logger(
            name="llm_gateway",
            level=config.logging.level,
            log_file=config.logging.file,
            max_bytes=config.logging.max_bytes,
            backup_count=config.logging.backup_count
        )

        self.nvidia = NvidiaProvider(config.nvidia)

        self.ollama = (
            OllamaProvider(config.ollama)
            if config.ollama.enabled
            else None
        )

        self.router = ModelRouter(
            config=config,
            nvidia_provider=self.nvidia,
            ollama_provider=self.ollama,
            logger=self.logger
        )

    @classmethod
    def from_config(cls, path: str | Path):
        config = GatewayConfig.from_yaml(path)
        return cls(config)

    def generate(self, messages: list[dict], **kwargs):
        return self.router.generate(messages, **kwargs)
    
    def get_middleware(self):
        return GatewayMiddleware(self.router)
    
    def close(self):
        
        self.router._chat_models.clear()
        self.nvidia.close()

        if self.ollama:
            self.ollama.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()