from abc import ABC, abstractmethod
from openai import OpenAI


class BaseProvider(ABC):

    def __init__(self, base_url: str, api_key: str):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    @abstractmethod
    def generate(self, model: str, messages: list[dict], **kwargs):
        pass

    def close(self):
        self.client.close()