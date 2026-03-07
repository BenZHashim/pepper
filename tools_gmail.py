"""
tools_gmail.py
Gmail integration tools: OAuth URL generation and email reading.

OAuth URL + token exchange are done with plain HTTP (no Flow object) so that
google-auth-oauthlib's internal PKCE handling doesn't interfere with ours.
Token refresh and Gmail API calls use the google-auth library as intended.
"""

import os
import json
import base64
import hashlib
import secrets
import asyncio
import urllib.parse

from mcp_shared import mcp, get_conn_for_phone, release_conn


def _b64url(data: bytes) -> str:
    """URL-safe base64 encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


@mcp.tool()
async def generate_gmail_auth_url(phone: str = None) -> dict:
    """
    Generate a Google OAuth URL so the user can connect their Gmail account.
    Call this when the user asks to connect Gmail, or when read_gmail returns
    a 'gmail_not_connected' error.
    The URL should be sent to the user as a text message — they tap it on their
    phone, sign in with Google, and Gmail gets connected automatically.
    """
    client_id     = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri  = os.getenv("GOOGLE_REDIRECT_URI")

    if not all([client_id, client_secret, redirect_uri]):
        return {
            "success": False,
            "error": (
                "Gmail integration is not configured. "
                "GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI must be set."
            ),
        }

    # ── PKCE (RFC 7636 S256) ──────────────────────────────────────────────────
    code_verifier  = secrets.token_urlsafe(96)
    code_challenge = _b64url(hashlib.sha256(code_verifier.encode("ascii")).digest())

    # Encode phone + verifier in state so the callback can recover both
    # without any server-side session storage.
    state_payload = _b64url(json.dumps({"p": phone, "v": code_verifier}).encode())

    params = {
        "client_id":             client_id,
        "redirect_uri":          redirect_uri,
        "response_type":         "code",
        "scope":                 "https://www.googleapis.com/auth/gmail.readonly",
        "access_type":           "offline",
        "prompt":                "consent",
        "state":                 state_payload,
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
    }
    auth_url = "https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode(params)

    return {"success": True, "url": auth_url}


@mcp.tool()
async def read_gmail(
    phone: str = None,
    query: str = "is:unread",
    max_results: int = 10,
) -> str:
    """
    Read emails from the user's connected Gmail account.
    Use when the user asks to check their email, see unread messages,
    find a specific email, or asks anything about their inbox.
    Call generate_gmail_auth_url first if this returns 'gmail_not_connected'.

    Args:
        query:       Gmail search query. Default is unread emails.
                     Examples: "is:unread", "from:boss@company.com",
                     "subject:invoice", "is:unread newer_than:1d"
        max_results: Maximum number of emails to return (default 10).
    """
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request as GoogleRequest
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    # ── Load tokens from DB ───────────────────────────────────────────────────
    conn = await get_conn_for_phone(phone)
    try:
        row = await conn.fetchrow(
            "SELECT gmail_tokens FROM users WHERE phone = $1", phone
        )
    finally:
        await release_conn(conn)

    if not row or not row["gmail_tokens"]:
        return "gmail_not_connected"

    # asyncpg returns JSONB columns as dicts; handle both just in case
    raw = row["gmail_tokens"]
    tokens = raw if isinstance(raw, dict) else json.loads(raw)

    # ── Build credentials ─────────────────────────────────────────────────────
    creds = Credentials(
        token=tokens.get("token"),
        refresh_token=tokens.get("refresh_token"),
        token_uri=tokens.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=tokens.get("client_id"),
        client_secret=tokens.get("client_secret"),
        scopes=tokens.get("scopes"),
    )

    # ── Refresh if expired ────────────────────────────────────────────────────
    if not creds.valid:
        if not creds.refresh_token:
            return "gmail_not_connected"
        try:
            await asyncio.to_thread(creds.refresh, GoogleRequest())
        except Exception as e:
            return f"gmail_not_connected: token refresh failed ({e})"

        # Persist the refreshed token
        updated = {
            "token":         creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri":     creds.token_uri,
            "client_id":     creds.client_id,
            "client_secret": creds.client_secret,
            "scopes":        list(creds.scopes) if creds.scopes else [],
            "expiry":        creds.expiry.isoformat() if creds.expiry else None,
        }
        conn = await get_conn_for_phone(phone)
        try:
            await conn.execute(
                "UPDATE users SET gmail_tokens = $1 WHERE phone = $2",
                json.dumps(updated), phone,
            )
        finally:
            await release_conn(conn)

    # ── Fetch emails (sync Google client → thread) ────────────────────────────
    max_results = min(max_results, 25)

    def _fetch():
        service = build("gmail", "v1", credentials=creds)

        result = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=max_results,
        ).execute()

        message_refs = result.get("messages", [])
        if not message_refs:
            return []

        emails = []
        for ref in message_refs:
            msg = service.users().messages().get(
                userId="me",
                id=ref["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()

            headers = {
                h["name"]: h["value"]
                for h in msg.get("payload", {}).get("headers", [])
            }
            emails.append({
                "from":    headers.get("From", "Unknown"),
                "subject": headers.get("Subject", "(no subject)"),
                "date":    headers.get("Date", ""),
                "snippet": msg.get("snippet", ""),
                "unread":  "UNREAD" in msg.get("labelIds", []),
            })

        return emails

    try:
        emails = await asyncio.to_thread(_fetch)
    except HttpError as e:
        if e.resp.status == 401:
            return "gmail_not_connected"
        return f"Gmail API error: {e}"

    if not emails:
        return f"No emails found for query: '{query}'"

    lines = [f"{len(emails)} email(s) for '{query}':\n"]
    for i, email in enumerate(emails, 1):
        marker = "●" if email["unread"] else "○"
        lines.append(
            f"{i}. {marker} {email['subject']}\n"
            f"   From: {email['from']}\n"
            f"   Date: {email['date']}\n"
            f"   {email['snippet']}"
        )

    return "\n\n".join(lines)
