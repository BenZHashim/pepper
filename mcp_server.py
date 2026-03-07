"""
mcp_server.py — Entry point
Imports all tool modules (which register tools onto the shared mcp instance),
then starts the FastMCP HTTP server.

Run standalone:  python mcp_server.py
Run via Docker:  docker compose up mcp
"""

import logging
from starlette.requests import Request
from starlette.responses import JSONResponse

# ─── Logging ──────────────────────────────────────────────────────────────────

class _SuppressNoiseRequests(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "/health" in msg:
            return False
        # Suppress MCP session handshake/teardown but keep tool calls
        if "/mcp" in msg and ("DELETE" in msg or "202 Accepted" in msg):
            return False
        return True

logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn.access").addFilter(_SuppressNoiseRequests())
logging.getLogger("mcp").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Shared state (mcp instance, pool, constants) ─────────────────────────────

from mcp_shared import mcp, MCP_PORT

# ─── Tool registration — import order doesn't matter ─────────────────────────
# Each module calls @mcp.tool() at import time, registering onto the shared mcp.

import tools_user       # noqa: F401  get_user, create_user, update_user, update_user_preferences*
import tools_reminders  # noqa: F401  create_reminder, list_reminders, delete_reminder, get_due_reminders
import tools_gmail      # noqa: F401  generate_gmail_auth_url
import tools_memory     # noqa: F401  search_conversation_history, search_messages_by_time, get_source_messages

# ─── Health check ─────────────────────────────────────────────────────────────

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})

# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"Starting FastMCP server on port {MCP_PORT}")
    mcp.run(transport="http", host="0.0.0.0", port=MCP_PORT)
