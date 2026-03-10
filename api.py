"""
FastAPI SMS Gateway
Handles inbound Twilio webhooks, runs the agentic Claude loop,
and connects to the FastMCP server for tools.

Run with:
    uvicorn api:app --reload --port 5000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import deps
from config import get_mcp_url
from services.conversation import ConversationManager
from services.database import get_db_pool
from services.embeddings import EmbeddingManager
from services.reminders import fire_due_reminders
from routes.other import router as other_router
from routes.sms import router as sms_router
from services.skills_agent import SkillsAgent
from services.summarization import SummarizationManager
from services.tools import ToolManager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await get_db_pool()

    deps.embedding_manager     = EmbeddingManager()
    deps.conversation_manager  = ConversationManager(pool, deps.embedding_manager)
    deps.summarization_manager = SummarizationManager(pool, deps.embedding_manager, deps.anthropic_client)
    deps.tool_manager          = ToolManager(pool, deps.embedding_manager, get_mcp_url())
    deps.skills_agent          = SkillsAgent(pool, deps.anthropic_client)

    await deps.tool_manager.sync()

    deps.scheduler.add_job(fire_due_reminders, "interval", minutes=1, id="reminders")
    deps.scheduler.start()
    logger.info("Startup complete — DB pool, managers, and scheduler running")
    yield

    deps.scheduler.shutdown()
    await deps.embedding_manager.close()

    import services.database as database
    if database._db_pool:
        await database._db_pool.close()


app = FastAPI(title="SMS AI Assistant", lifespan=lifespan)
app.include_router(sms_router)
app.include_router(other_router)
