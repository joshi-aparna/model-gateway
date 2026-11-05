import os
import yaml

from pathlib import Path
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    id: str
    alias: str = "default"


class NvidiaConfig(BaseModel):
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "NVIDIA_API_KEY"
    models: list[ModelConfig]


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    model: str
    enabled: bool = True


class RoutingConfig(BaseModel):
    strategy: str = "round_robin"
    fallback_enabled: bool = True


class GatewayConfig(BaseModel):
    nvidia: NvidiaConfig
    ollama: OllamaConfig
    routing: RoutingConfig = Field(
        default_factory=RoutingConfig
    )

    @classmethod
    def from_yaml(cls, path: str | Path):
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        return cls.model_validate(data)

class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str | None = None
    max_bytes: int = 10485760  # 10 MB
    backup_count: int = 5


class GatewayConfig(BaseModel):
    nvidia: NvidiaConfig
    ollama: OllamaConfig
    routing: RoutingConfig = Field(
        default_factory=RoutingConfig
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GatewayConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return cls.model_validate(data)