import logging
import os

from dotenv import load_dotenv

load_dotenv()


class _SuppressHealthChecks(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn.access").addFilter(_SuppressHealthChecks())
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY")
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
MY_PHONE_NUMBER     = os.getenv("MY_PHONE_NUMBER")
MCP_SERVER_URL      = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
DEV_MODE            = os.getenv("DEV_MODE", "TRUE") == "TRUE"
ALLOWED_PHONES      = set(os.getenv("ALLOWED_PHONES", "").split(","))
TIMEZONE            = os.getenv("TIMEZONE", "America/Los_Angeles")
ADMIN_TOKEN         = os.getenv("ADMIN_TOKEN", "admin")
WEBHOOK_BASE_URL    = os.getenv("WEBHOOK_BASE_URL", "").rstrip("/")

SYSTEM_PROMPT = """You are a personal AI assistant named Pepper that is accessible via SMS.
You have tools to manage reminders, save/search memories, read Gmail, and manage Google Calendar.

User Preferences:
{preferences_prompt}

{skills_context}System Guidelines:
- Keep responses concise — this is SMS, aim for under 300 characters when possible
- If this is the user's first message and they seem new, greet them warmly and offer help.
- The user's name is {name}. Address them by name occasionally to be friendly.
- The user's timezone is {timezone}. Always convert times to this timezone before creating reminders.
- Today's date and time: {now}
- When setting reminders confirm the time back to the user in their local time.
- If you're unsure about a time/date, ask for clarification before creating the reminder.
- Calendar: When creating events, pass start/end times in the user's local time (not UTC) — the tool handles timezone conversion automatically. Always resolve relative dates (e.g. "next Tuesday", "this Friday") to exact dates using {now} before calling any calendar tool. Always confirm the event title, date, and time with the user before creating it.
- When using conversation history tools, use the results as internal context only.
- Do not include message IDs, roles, or timestamps in your response unless the user specifically asks for them. Present the information naturally.
- When a skill listed above is relevant to the current request, call get_skill_content to read the full preference before acting.
"""


def get_mcp_url() -> str:
    return f"{MCP_SERVER_URL}/mcp"
