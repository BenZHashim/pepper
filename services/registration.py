import json
import logging

from fastapi import Response
from twilio.twiml.messaging_response import MessagingResponse

import deps
from .database import get_db_pool
from .llm_client import create_message

logger = logging.getLogger(__name__)

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
    pool = await get_db_pool()
    response = await create_message(
        deps.anthropic_client, pool,
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
    return {"name": None, "timezone": None}


def _registration_prompt(pending: dict) -> str:
    if not pending.get("name") and not pending.get("timezone"):
        return (
            "Hi! I'm Pepper, your AI assistant. "
            "I can set reminders and manage your Google Calendar & Gmail. Tell me how you like things done and I'll learn. "
            "To get started, what's your name and timezone?"
        )
    if not pending.get("name"):
        return "I have your timezone, but I still don't have your name. What's your name?"
    if not pending.get("timezone"):
        return f"Nice to meet you, {pending['name']}! What timezone are you in?"
    return None


async def handle_registration(sender: str, message_body: str) -> Response:
    pending = deps.conversation_manager.get_pending_registration(sender)

    if not pending:
        deps.conversation_manager.set_pending_registration(sender)
        pending = deps.conversation_manager.get_pending_registration(sender)

    extracted = await _extract_registration_fields(message_body)
    if extracted.get("name"):
        deps.conversation_manager.update_pending_registration(sender, name=extracted["name"])
    if extracted.get("timezone"):
        deps.conversation_manager.update_pending_registration(sender, timezone=extracted["timezone"])

    pending = deps.conversation_manager.get_pending_registration(sender)
    next_question = _registration_prompt(pending)

    if next_question:
        twiml = MessagingResponse()
        twiml.message(next_question)
        return Response(content=str(twiml), media_type="application/xml")

    result = await deps.tool_manager.call("create_user", {
        "phone":    sender,
        "name":     pending["name"],
        "timezone": pending["timezone"],
    })
    user_data = json.loads(result)
    deps.conversation_manager.clear_pending_registration(sender)

    if user_data.get("success"):
        twiml = MessagingResponse()
        twiml.message(
            f"You're all set, {pending['name']}! "
            f"I'll use {pending['timezone']} for your reminders. "
            "How can I help you today?"
        )
        return Response(content=str(twiml), media_type="application/xml")
    else:
        deps.conversation_manager.set_pending_registration(sender)
        twiml = MessagingResponse()
        twiml.message("Sorry, something went wrong. Please try again.")
        return Response(content=str(twiml), media_type="application/xml")
