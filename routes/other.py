import base64
import json
import logging
import os

import requests as google_requests
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from fastmcp import Client as MCPClient

import deps 
from config import ADMIN_TOKEN, TWILIO_PHONE_NUMBER, get_mcp_url
from services.database import get_db_pool

logger = logging.getLogger(__name__)

router = APIRouter()


def verify_admin(request: Request) -> None:
    if not ADMIN_TOKEN or request.headers.get("X-Admin-Token") != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")


@router.post("/admin/summarise")
async def trigger_summarisation(
    phone: str,
    _: None = Depends(verify_admin),
) -> dict:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        user_id = await conn.fetchval("SELECT id FROM users WHERE phone = $1", phone)
    if not user_id:
        raise HTTPException(status_code=404, detail=f"No user found for {phone}")

    candidates = await deps.summarization_manager.get_pruning_candidates(user_id)
    await deps.summarization_manager.run_summarisation_pass(user_id)
    return {"user_id": user_id, "candidates_found": len(candidates)}


@router.post("/admin/messages/embed")
async def embed_all_messages(
    _: None = Depends(verify_admin),
) -> dict:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
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
                content = row["content"]
                if isinstance(content, str):
                    text = content
                elif isinstance(content, dict):
                    text = content.get("text") or content.get("content") or str(content)
                elif isinstance(content, list):
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

                vector = await deps.embedding_manager.embed(text)

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


@router.get("/auth/google/callback")
async def google_callback(code: str, state_param: str = Query(alias="state")):
    """
    Google OAuth2 redirect target.
    `state` is a base64url-encoded JSON payload containing the user's phone
    number and the PKCE code_verifier generated when the auth URL was created.
    """
    padding    = (4 - len(state_param) % 4) % 4
    state_json = base64.urlsafe_b64decode(state_param + "=" * padding).decode()
    state_data = json.loads(state_json)
    phone         = state_data["p"]
    code_verifier = state_data["v"]

    client_id     = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri  = os.getenv("GOOGLE_REDIRECT_URI")

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
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET google_tokens = $1 WHERE phone = $2",
            json.dumps(tokens),
            phone,
        )

    message = "Your Google account is now connected! You can ask me to check your emails or manage your calendar."
    await deps.conversation_manager.add_message(phone, "assistant", message)

    try:
        use_whatsapp = os.getenv("USE_WHATSAPP", "false").lower() == "true"
        from_number  = f"whatsapp:{TWILIO_PHONE_NUMBER}" if use_whatsapp else TWILIO_PHONE_NUMBER
        to_number    = f"whatsapp:{phone}" if use_whatsapp else phone
        deps.twilio_client.messages.create(
            body=message,
            from_=from_number,
            to=to_number,
        )
    except Exception as e:
        logger.error(f"Failed to send Gmail confirmation SMS to {phone}: {e}")

    return PlainTextResponse("Gmail connected! You can close this tab and go back to texting.")


@router.get("/health")
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
        "active_conversations": await deps.conversation_manager.active_count(),
        "scheduler_running": deps.scheduler.running,
    }
