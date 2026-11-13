from langchain.agents.middleware.types import AgentMiddleware
from dataclasses import dataclass

@dataclass
class GatewayContext:
    aliases: list[str] | None = None

class GatewayMiddleware(AgentMiddleware):

    def __init__(self, router):
        super().__init__()
        self.router = router

    async def awrap_model_call(self, request, handler):
        aliases = request.runtime.context.aliases
        return await self.router.ainvoke(request, handler, aliases=aliases)