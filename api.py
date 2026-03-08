"""
FastAPI SMS Gateway
Handles inbound Twilio webhooks, runs the agentic Claude loop,
and connects to the FastMCP server for tools.

Run with:
    uvicorn main:app --reload --port 5000
"""

import asyncio
import os
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
import re
from zoneinfo import ZoneInfo

import anthropic
import httpx
import asyncpg
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Depends, FastAPI, Form, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from fastmcp import Client as MCPClient
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

from conversation import ConversationManager
from embeddings import EmbeddingManager
from summarization import SummarizationManager
from tools import ToolManager
from skills_agent import SkillsAgent
from llm_client import create_message

load_dotenv()


class _SuppressHealthChecks(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()

logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn.access").addFilter(_SuppressHealthChecks())
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY")
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
MY_PHONE_NUMBER     = os.getenv("MY_PHONE_NUMBER")   # Your personal number for outbound reminders
MCP_SERVER_URL      = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
DEV_MODE            = os.getenv("DEV_MODE", "TRUE") == "TRUE"  # Skip Twilio signature validation in dev mode
ALLOWED_PHONES = set(os.getenv("ALLOWED_PHONES", "").split(","))  # Comma-separated list of allowed phone numbers
TIMEZONE           = os.getenv("TIMEZONE", "America/Los_Angeles")
ADMIN_TOKEN        = os.getenv("ADMIN_TOKEN", "admin")

SYSTEM_PROMPT = """You are a personal AI assistant named Pepper that is accessible via SMS.
You have tools to manage reminders, save/search memories, read Gmail, and manage Google Calendar.

User Preferences:
{preferences_prompt}

{skills_context}
System Guidelines:
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

# ─── Clients ──────────────────────────────────────────────────────────────────

anthropic_client    = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
twilio_client       = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
twilio_validator    = RequestValidator(TWILIO_AUTH_TOKEN)

def get_mcp_url() -> str:
    """Helper to construct full MCP tool URLs."""
    return f"{MCP_SERVER_URL}/mcp"

async def get_or_create_user(phone: str) -> dict:
    """Look up user by phone, prompt for name if new."""
    result = await tool_manager.call("get_user", {"phone": phone})
    user = json.loads(result)
    return user

_db_pool: asyncpg.Pool = None

async def get_db_pool() -> asyncpg.Pool:
    global _db_pool
    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL", "postgresql://smsai:smsai_secret@db:5432/smsai"),
            min_size=2,
            max_size=10,
            ssl=False,
        )
    return _db_pool

def _strip_context(message: str) -> str:
    return re.sub(r'^\[.*?\]\s*', '', message)


async def _get_user_skills(phone: str) -> list[dict]:
    """Return raw list of {name, description} for the user's skills."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT s.name, s.description
            FROM skills s
            JOIN users u ON u.id = s.user_id
            WHERE u.phone = $1
            ORDER BY s.name
            """,
            phone,
        )
    return [{"name": r["name"], "description": r["description"]} for r in rows]


def _format_skills_context(skills: list[dict]) -> str:
    """Format skills list as a system prompt section, or empty string if none."""
    if not skills:
        return ""
    lines = ["## Your Personalized Skills", ""]
    for s in skills:
        lines.append(f"- **{s['name']}**: {s['description']}")
    lines.append("")
    return "\n".join(lines)


# ─── Registration Helpers ─────────────────────────────────────────────────────

_REGISTRATION_TOOL = {
    "name": "save_registration_fields",
    "description": "Save the extracted name and timezone from the user's message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": ["string", "null"],
                "description": "The user's name, or null if not provided.",
            },
            "timezone": {
                "type": ["string", "null"],
                "description": "A valid IANA timezone string (e.g. 'America/New_York'), or null if not provided. Infer from city/region names.",
            },
        },
        "required": ["name", "timezone"],
    },
}

async def _extract_registration_fields(message: str) -> dict:
    """
    Use Claude to extract name and/or timezone from a free-form message.
    Enforces schema via tool use — always returns {name, timezone}.
    """
    pool = await get_db_pool()
    response = await create_message(
        anthropic_client, pool,
        purpose="registration",
        user_id=None,
        model="claude-haiku-4-5",
        max_tokens=100,
        tools=[_REGISTRATION_TOOL],
        tool_choice={"type": "tool", "name": "save_registration_fields"},
        messages=[{
            "role": "user",
            "content": f"Extract the name and timezone from this message: {message}",
        }],
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input

    return {"name": None, "timezone": None}  # unreachable in practice, but safe fallback

def _registration_prompt(state: dict) -> str:
    """Return the next question to ask based on what's still missing."""
    if not state.get("name") and not state.get("timezone"):
        return "Hi! I don't recognize your number. What's your name and what timezone are you in?"
    if not state.get("name"):
        return "I have your timezone, but I still don't have your name. What's your name?"
    if not state.get("timezone"):
        return f"Nice to meet you, {state['name']}! What timezone are you in?"
    return None   # Both fields present — registration can proceed


async def _handle_registration(sender: str, message_body: str) -> Response:
    """
    Multi-turn registration flow. Blocks the user from proceeding until
    both name and timezone have been collected and the user is created.
    """
    pending = conversation_manager.get_pending_registration(sender)

    if not pending:
        # First contact — start the flow
        conversation_manager.set_pending_registration(sender)
        pending = conversation_manager.get_pending_registration(sender)

    # Try to extract fields from whatever the user just said
    extracted = await _extract_registration_fields(message_body)
    if extracted.get("name"):
        conversation_manager.update_pending_registration(sender, name=extracted["name"])
    if extracted.get("timezone"):
        conversation_manager.update_pending_registration(sender, timezone=extracted["timezone"])

    # Re-fetch updated state
    pending = conversation_manager.get_pending_registration(sender)
    next_question = _registration_prompt(pending)

    if next_question:
        # Still missing something — ask again
        twiml = MessagingResponse()
        twiml.message(next_question)
        return Response(content=str(twiml), media_type="application/xml")

    # Both fields collected — create the user
    result = await tool_manager.call("create_user", {
        "phone":    sender,
        "name":     pending["name"],
        "timezone": pending["timezone"],
    })
    user_data = json.loads(result)
    conversation_manager.clear_pending_registration(sender)

    if user_data.get("success"):
        twiml = MessagingResponse()
        twiml.message(
            f"You're all set, {pending['name']}! "
            f"I'll use {pending['timezone']} for your reminders. "
            "How can I help you today?"
        )
        return Response(content=str(twiml), media_type="application/xml")
    else:
        # Creation failed — reset so they can try again
        conversation_manager.set_pending_registration(sender)
        twiml = MessagingResponse()
        twiml.message("Sorry, something went wrong. Please try again.")
        return Response(content=str(twiml), media_type="application/xml")


# ─── Agentic Loop ─────────────────────────────────────────────────────────────

def _get_user_system_prompt(user: dict, skills: list[dict]) -> str:
    """Fill the system prompt template with user-specific info and skills context."""
    return SYSTEM_PROMPT.format(
        preferences_prompt=user.get("preferences_prompt", "You are a helpful personal AI assistant."),
        name=user.get("name", "there"),
        timezone=user.get("timezone"),
        now=datetime.now(ZoneInfo(user.get("timezone"))).strftime("%A, %B %d %Y at %I:%M %p %Z"),
        skills_context=_format_skills_context(skills),
    )

async def run_agent(phone: str, user_message: str, user: dict) -> str:
    """
    Run the agentic loop for a specific user.
    Uses the user's custom preferences prompt.
    """
    tools = await tool_manager.get_relevant_tools(user_message)
    await conversation_manager.add_message(phone, "user", user_message)
    messages = await conversation_manager.get_history(phone, user_message)

    skills = await _get_user_skills(phone)
    system = _get_user_system_prompt(user, skills)
    pool   = await get_db_pool()

    for _ in range(5):
        response = await create_message(
            anthropic_client, pool,
            purpose="main_agent",
            user_id=user.get("id"),
            model="claude-haiku-4-5",
            max_tokens=500,
            system=system,
            tools=tools,
            messages=messages,
        )

        text_content = ""
        tool_uses = []

        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_uses.append(block)

        if response.stop_reason == "end_turn" or not tool_uses:
            if not text_content.strip():
                logger.warning(f"Claude returned empty text_content for {phone}")
                text_content = "Done!"
            text_content = _strip_context(text_content)
            await conversation_manager.add_message(phone, "assistant", text_content)
            asyncio.create_task(skills_agent.run(phone, user_message, skills))
            return text_content

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_use in tool_uses:
            logger.info(f"Calling tool: {tool_use.name} with {tool_use.input}")
            # Always inject the phone number so tools know which user this is
            tool_input = {**tool_use.input, "phone": phone}
            result = await tool_manager.call(tool_use.name, tool_input)
            logger.info(f"Tool result: {result}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    return "I ran into an issue processing that. Please try again."


# ─── Scheduler: Outbound Reminders ────────────────────────────────────────────

async def fire_due_reminders():
    """Fires due reminders — sends to each user's own phone number."""
    try:
        result_str = await tool_manager.call("get_due_reminders", {})
        logger.info(f"Raw due reminders result: {result_str}")
        due = json.loads(result_str)
        logger.info(f"Parsed {len(due)} due reminders")

        if len(due) == 0:
            logger.info("No due reminders")
            return

        use_whatsapp = os.getenv("USE_WHATSAPP", "false").lower() == "true"

        for reminder in due:
            phone = reminder["phone"]
            logger.info(f"Firing reminder for {phone}: {reminder['message']}")

            # ── NEW: Personalize the reminder via Claude ──────────────────
            user = await get_or_create_user(phone) 
            if not user.get("found"):
                logger.warning(f"User not found for phone {phone} while firing reminder")
                personalized_message = f"⏰ Reminder: {reminder['message']}"
            else:
                personalized_message = await personalize_reminder(reminder["message"], user)
            # ─────────────────────────────────────────────────────────────

            from_number = f"whatsapp:{TWILIO_PHONE_NUMBER}" if use_whatsapp else TWILIO_PHONE_NUMBER
            to_number   = f"whatsapp:{phone}" if use_whatsapp else phone

            twilio_client.messages.create(
                body=personalized_message,
                from_=from_number,
                to=to_number,
            )

            await conversation_manager.add_message(phone, "assistant", personalized_message)
            logger.info(f"Reminder sent to {to_number}")
    except Exception as e:
        logger.error(f"Scheduler error: {e}", exc_info=True)


async def personalize_reminder(raw_message: str, user: dict) -> str:
    """Run a reminder message through Claude to make it feel natural and personalized."""
    skills = await _get_user_skills(user.get("phone", ""))
    system = _get_user_system_prompt(user, skills)
    pool   = await get_db_pool()

    response = await create_message(
        anthropic_client, pool,
        purpose="personalize_reminder",
        user_id=user.get("id"),
        model="claude-haiku-4-5",
        max_tokens=200,
        system=system,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You're sending a reminder to {user.get('name', 'the user')}. "
                    f"Rewrite this reminder in your natural voice — keep it concise but warm and engaging. "
                    f"Raw reminder: \"{raw_message}\""
                ),
            }
        ],
    )

    text = next((b.text for b in response.content if b.type == "text"), None)
    return f"⏰ {text.strip()}" if text else f"⏰ Reminder: {raw_message}"


# ─── App Lifespan ─────────────────────────────────────────────────────────────

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await get_db_pool()

    global embedding_manager, conversation_manager, tool_manager, summarization_manager, skills_agent
    embedding_manager    = EmbeddingManager()
    conversation_manager = ConversationManager(pool, embedding_manager)
    summarization_manager = SummarizationManager(pool, embedding_manager, anthropic_client)
    tool_manager         = ToolManager(pool, embedding_manager, get_mcp_url())
    skills_agent         = SkillsAgent(pool, anthropic_client)

    await tool_manager.sync()

    scheduler.add_job(fire_due_reminders, "interval", minutes=1, id="reminders")
    scheduler.start()
    logger.info("Startup complete — DB pool, managers, and scheduler running")
    yield

    scheduler.shutdown()
    await embedding_manager.close()
    if _db_pool:
        await _db_pool.close()


app = FastAPI(title="SMS AI Assistant", lifespan=lifespan)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/sms")
async def sms_webhook(
    request: Request,
    From: str = Form(...),
    Body: str = Form(...),
):
    if DEV_MODE == True:
        pass
    else:
        form_data = await request.form()
        signature = request.headers.get("X-Twilio-Signature", "")
        if not twilio_validator.validate(str(request.url), dict(form_data), signature):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    # Normalize sender — strip whatsapp: prefix if present
    sender = From.replace("whatsapp:", "")
    message_body = Body.strip()
    logger.info(f"SMS from {sender}: {message_body}")

    # Allowlist check
    if sender not in ALLOWED_PHONES:
        logger.warning(f"Rejected message from unauthorized number: {sender}")
        twiml = MessagingResponse()
        twiml.message("Sorry, this assistant is private.")
        return Response(content=str(twiml), media_type="application/xml")
    
    # Look up or handle new user
    user = await get_or_create_user(sender)

    # Handle STOP / HELP before anything else
    if message_body.lower() == "stop":
        return handle_stop_command()
    
    if message_body.lower() in ["help", "?"]:
        return handle_help_command(message_body)

    if message_body.lower() in ["reset", "clear"]:
        return await async_handle_reset_command(user.get("phone", sender))

    
    logger.info(f"User found: {user.get('found')}")
    if not user.get("found"):
        return await _handle_registration(sender, message_body)
        

    # Existing user — run the agent
    try:
        reply = await run_agent(sender, message_body, user)
    except Exception as e:
        logger.error(f"Agent error for {sender}: {e}", exc_info=True)
        reply = "Sorry, something went wrong. Please try again."

    twiml = MessagingResponse()
    twiml.message(reply)
    return Response(content=str(twiml), media_type="application/xml")

def handle_help_command(message_body: str) -> Response:
    twiml = MessagingResponse()
    twiml.message(
        "I'm your personal AI assistant.\n"
        "• Ask me anything\n"
        "• 'remind me to [X] at [time]'\n"
        "• 'what reminders do I have?'\n"
        "• 'reset' to clear chat history\n"
        "Reply STOP to unsubscribe."
    )
    return Response(content=str(twiml), media_type="application/xml")

def handle_stop_command() -> Response:
    twiml = MessagingResponse()
    twiml.message("You have been unsubscribed and will receive no further messages.")
    return Response(content=str(twiml), media_type="application/xml")

async def async_handle_reset_command(phone: str) -> Response:
    await conversation_manager.clear_history(phone)
    twiml = MessagingResponse()
    twiml.message("Conversation history cleared. Starting fresh!")
    return Response(content=str(twiml), media_type="application/xml")

def verify_admin(request: Request) -> None:
    if not ADMIN_TOKEN or request.headers.get("X-Admin-Token") != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")


# Admin Endpoints 
@app.post("/admin/summarise")
async def trigger_summarisation(
    phone: str,
    _: None = Depends(verify_admin),
) -> dict:
    async with _db_pool.acquire() as conn:
        user_id = await conn.fetchval("SELECT id FROM users WHERE phone = $1", phone)
    if not user_id:
        raise HTTPException(status_code=404, detail=f"No user found for {phone}")

    candidates = await  summarization_manager.get_pruning_candidates(user_id)
    await summarization_manager.run_summarisation_pass(user_id)
    return {"user_id": user_id, "candidates_found": len(candidates)}

@app.post("/admin/messages/embed")
async def embed_all_messages(
    _: None = Depends(verify_admin),
) -> dict:
    async with _db_pool.acquire() as conn:
        # Fetch all messages that need embedding
        rows = await conn.fetch(
            """
            SELECT m.id, m.content
            FROM messages m
            WHERE m.embedding IS NULL
              AND m.embedding_pruned_at IS NULL
            ORDER BY m.id
            """
        )

        if not rows:
            return {"embedded": 0, "skipped": 0, "errors": 0}

        embedded = 0
        skipped  = 0
        errors   = 0

        for row in rows:
            try:
                # Extract plain text from jsonb content
                content = row["content"]
                if isinstance(content, str):
                    text = content
                elif isinstance(content, dict):
                    text = content.get("text") or content.get("content") or str(content)
                elif isinstance(content, list):
                    # e.g. [{type: "text", text: "..."}]
                    text = " ".join(
                        block.get("text", "") for block in content
                        if isinstance(block, dict)
                    ).strip()
                else:
                    logger.warning(f"Message {row['id']} has unrecognised content shape, skipping")
                    skipped += 1
                    continue

                if not text.strip():
                    skipped += 1
                    continue

                vector = await embedding_manager.embed(text)

                await conn.execute(
                    """
                    UPDATE messages
                    SET
                        embedding     = $1,
                        search_vector = to_tsvector('english', $2)
                    WHERE id = $3
                    """,
                    vector,
                    text,
                    row["id"],
                )
                embedded += 1

            except Exception as e:
                logger.error(f"Failed to embed message {row['id']}: {e}", exc_info=True)
                errors += 1

        logger.info(f"Embedding run complete — embedded={embedded}, skipped={skipped}, errors={errors}")
        return {"embedded": embedded, "skipped": skipped, "errors": errors}

@app.get("/auth/google/callback")
async def google_callback(code: str, state: str):
    """
    Google OAuth2 redirect target.
    `state` is a base64url-encoded JSON payload containing the user's phone
    number and the PKCE code_verifier generated when the auth URL was created.
    Exchanges the auth code for tokens via a plain POST (no Flow object) so
    that google-auth-oauthlib's internal PKCE handling doesn't interfere.
    """
    import base64
    import requests as google_requests
    from fastapi.responses import PlainTextResponse

    # ── Decode state ──────────────────────────────────────────────────────────
    padding       = (4 - len(state) % 4) % 4   # 0 if already aligned
    state_json    = base64.urlsafe_b64decode(state + "=" * padding).decode()
    state_data    = json.loads(state_json)
    phone         = state_data["p"]
    code_verifier = state_data["v"]

    client_id     = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri  = os.getenv("GOOGLE_REDIRECT_URI")

    # ── Exchange code for tokens (plain HTTP — no Flow) ───────────────────────
    # We avoid Flow.fetch_token() because newer google-auth-oauthlib versions
    # generate their own PKCE verifier internally, making ours invalid.
    token_resp = google_requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code":          code,
            "client_id":     client_id,
            "client_secret": client_secret,
            "redirect_uri":  redirect_uri,
            "grant_type":    "authorization_code",
            "code_verifier": code_verifier,
        },
    )

    if not token_resp.ok:
        logger.error(f"Google token exchange failed: {token_resp.text}")
        return PlainTextResponse("Failed to connect Google account. Please try again.", status_code=500)

    token_data = token_resp.json()
    tokens = {
        "token":         token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "token_uri":     "https://oauth2.googleapis.com/token",
        "client_id":     client_id,
        "client_secret": client_secret,
        "scopes":        token_data.get("scope", "").split(),
        "expiry":        None,
    }
    pool  = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET google_tokens = $1 WHERE phone = $2",
            json.dumps(tokens),
            phone,
        )

    message = "Your Google account is now connected! You can ask me to check your emails or manage your calendar."
    await conversation_manager.add_message(phone, "assistant", message)

    try:
        use_whatsapp = os.getenv("USE_WHATSAPP", "false").lower() == "true"
        from_number  = f"whatsapp:{TWILIO_PHONE_NUMBER}" if use_whatsapp else TWILIO_PHONE_NUMBER
        to_number    = f"whatsapp:{phone}" if use_whatsapp else phone
        twilio_client.messages.create(
            body=message,
            from_=from_number,
            to=to_number,
        )
    except Exception as e:
        logger.error(f"Failed to send Gmail confirmation SMS to {phone}: {e}")

    return PlainTextResponse("Gmail connected! You can close this tab and go back to texting.")


@app.get("/health")
async def health():
    mcp_ok = False
    try:
        async with MCPClient(get_mcp_url()) as client:
            await client.list_tools()
            mcp_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "mcp_server": "connected" if mcp_ok else "unreachable",
        "active_conversations": await conversation_manager.active_count(),
        "scheduler_running": scheduler.running,
    }


