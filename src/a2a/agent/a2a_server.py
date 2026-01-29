from fastapi import FastAPI

class A2AServer:
    """Minimal placeholder A2A server for local testing.
    Provides a Starlette/FastAPI app via `get_starlette_app()` and an agent card via `_get_agent_card()`.
    """
    def __init__(self, httpx_client=None, host: str = "0.0.0.0", port: int = 8001):
        self._httpx = httpx_client
        self._host = host
        self._port = port
        self._app = FastAPI()

        @self._app.get("/agent-card")
        async def agent_card():
            return self._get_agent_card()

        @self._app.get("/agent-card/info")
        async def agent_card_info():
            return {"info": "Zava A2A placeholder agent card"}

    def get_starlette_app(self):
        return self._app

    def _get_agent_card(self):
        return {
            "id": "zava-product-manager",
            "name": "Zava Product Manager",
            "description": "Placeholder A2A agent card for local testing",
        }
