"""HTTP entrypoint for exposing the MCP server through ngrok.

This file reuses the tool definitions from ``server.py`` and runs the same
server with a network transport instead of local stdio.
"""

from __future__ import annotations

import contextlib
import os

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mcp.server.transport_security import TransportSecuritySettings
from medicine_equivalence_mcp.server import mcp


def _normalize_mount_path(path: str) -> str:
    """Normalize the external mount path expected by the reverse proxy/client."""
    normalized = path if path.startswith("/") else f"/{path}"
    if normalized != "/" and normalized.endswith("/"):
        normalized = normalized.rstrip("/")
    return normalized


def build_app() -> Starlette:
    """Build an ASGI app that mounts the MCP streamable HTTP handler."""
    mount_path = _normalize_mount_path(os.getenv("MCP_PATH", "/mcp/"))
    external_host = os.getenv("RENDER_EXTERNAL_HOSTNAME") or os.getenv("MCP_PUBLIC_HOST")

    if external_host:
        mcp.settings.host = external_host
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[external_host, f"{external_host}:*"],
            allowed_origins=[f"https://{external_host}"],
        )

    mcp.settings.streamable_http_path = "/"
    mounted_app = mcp.streamable_http_app()

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        async with mcp.session_manager.run():
            yield

    async def healthcheck(request):
        return JSONResponse({"status": "ok", "service": "medicine-equivalence-mcp"})

    routes = [Route("/", endpoint=healthcheck), Mount(mount_path, app=mounted_app)]
    if mount_path != "/" and not mount_path.endswith("/"):
        routes.append(Mount(f"{mount_path}/", app=mounted_app))

    return Starlette(
        routes=routes,
        lifespan=lifespan,
    )


app = build_app()


def main() -> None:
    """Run the MCP server over Streamable HTTP for remote access."""
    host = os.getenv("MCP_BIND_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("MCP_PORT", "8200")))
    uvicorn.run("medicine_equivalence_mcp.server_http:app", host=host, port=port)


if __name__ == "__main__":
    main()
