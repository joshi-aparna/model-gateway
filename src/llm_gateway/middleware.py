from langchain.agents.middleware.types import AgentMiddleware

class GatewayMiddleware(AgentMiddleware):

    def __init__(self, router):
        super().__init__()
        self.router = router

    async def awrap_model_call(self, request, handler):
        return await self.router.ainvoke(request, handler)