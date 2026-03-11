import logging

from fastapi import APIRouter, Form, HTTPException, Request, Response
from twilio.twiml.messaging_response import MessagingResponse

import deps
from services.agent import get_or_create_user, run_agent
from config import DEV_MODE, WEBHOOK_BASE_URL
from services.registration import handle_registration

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/sms")
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
        url = f"{WEBHOOK_BASE_URL}/sms" if WEBHOOK_BASE_URL else str(request.url)
        if not deps.twilio_validator.validate(url, dict(form_data), signature):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    sender = From.replace("whatsapp:", "")
    message_body = Body.strip()
    logger.info(f"SMS from {sender}: {message_body}")

    user = await get_or_create_user(sender)

    if message_body.lower() == "stop":
        return _handle_stop()

    if message_body.lower() in ["help", "?"]:
        return _handle_help()

    if message_body.lower() in ["reset", "clear"]:
        return await _handle_reset(user.phone if user else sender)

    logger.info(f"User found: {user is not None}")
    if not user:
        return await handle_registration(sender, message_body)

    try:
        reply = await run_agent(sender, message_body, user)
    except Exception as e:
        logger.error(f"Agent error for {sender}: {e}", exc_info=True)
        reply = "Sorry, something went wrong. Please try again."

    twiml = MessagingResponse()
    twiml.message(reply)
    return Response(content=str(twiml), media_type="application/xml")


def _handle_help() -> Response:
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


def _handle_stop() -> Response:
    twiml = MessagingResponse()
    twiml.message("You have been unsubscribed and will receive no further messages.")
    return Response(content=str(twiml), media_type="application/xml")


async def _handle_reset(phone: str) -> Response:
    await deps.conversation_manager.clear_history(phone)
    twiml = MessagingResponse()
    twiml.message("Conversation history cleared. Starting fresh!")
    return Response(content=str(twiml), media_type="application/xml")
