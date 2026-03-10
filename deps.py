"""Shared application state — clients and manager singletons."""
import anthropic
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient

from config import ANTHROPIC_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN

# ─── Clients (constructed once at import time) ────────────────────────────────

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
twilio_client    = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)

# ─── Scheduler ────────────────────────────────────────────────────────────────

scheduler = AsyncIOScheduler()

# ─── Managers (populated during lifespan startup) ─────────────────────────────

embedding_manager     = None
conversation_manager  = None
tool_manager          = None
summarization_manager = None
skills_agent          = None
