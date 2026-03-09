"""
tools_google.py
Google integration tools: Gmail reading and Google Calendar management.

OAuth URL + token exchange are done with plain HTTP (no Flow object) so that
google-auth-oauthlib's internal PKCE handling doesn't interfere with ours.
Token refresh and Google API calls use the google-auth library as intended.
"""

import os
import json
import base64
import hashlib
import secrets
import asyncio
import urllib.parse
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from mcp_shared import mcp, get_conn_for_phone, release_conn, TIMEZONE as SERVER_TIMEZONE


def _b64url(data: bytes) -> str:
    """URL-safe base64 encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _localize_to_utc(dt_str: str, tz_name: str) -> str:
    """
    If dt_str has no timezone designator, interpret it as local time in
    tz_name and convert to UTC. Otherwise return as-is.
    """
    if not dt_str:
        return dt_str
    if dt_str.endswith("Z") or "+" in dt_str[10:] or "-" in dt_str[10:]:
        return dt_str
    user_tz = ZoneInfo(tz_name)
    dt_local = datetime.fromisoformat(dt_str).replace(tzinfo=user_tz)
    return dt_local.astimezone(timezone.utc).isoformat()


async def _get_google_creds(phone: str):
    """
    Load Google OAuth credentials and user timezone from the DB, refreshing if expired.
    Returns (creds, user_tz, None) on success, or (None, None, error_string) on failure.
    """
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request as GoogleRequest

    conn = await get_conn_for_phone(phone)
    try:
        row = await conn.fetchrow(
            "SELECT google_tokens, timezone FROM users WHERE phone = $1", phone
        )
    finally:
        await release_conn(conn)

    user_tz = (row["timezone"] if row and row["timezone"] else None) or SERVER_TIMEZONE

    if not row or not row["google_tokens"]:
        return None, None, "google_not_connected"

    raw = row["google_tokens"]
    tokens = raw if isinstance(raw, dict) else json.loads(raw)

    creds = Credentials(
        token=tokens.get("token"),
        refresh_token=tokens.get("refresh_token"),
        token_uri=tokens.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=tokens.get("client_id"),
        client_secret=tokens.get("client_secret"),
        scopes=tokens.get("scopes"),
    )

    if not creds.valid:
        if not creds.refresh_token:
            return None, None, "google_not_connected"
        try:
            await asyncio.to_thread(creds.refresh, GoogleRequest())
        except Exception as e:
            return None, None, f"google_not_connected: token refresh failed ({e})"

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
                "UPDATE users SET google_tokens = $1 WHERE phone = $2",
                json.dumps(updated), phone,
            )
        finally:
            await release_conn(conn)

    return creds, user_tz, None


# ── Auth ──────────────────────────────────────────────────────────────────────

@mcp.tool()
async def generate_google_auth_url(phone: str = None) -> dict:
    """
    Generate a Google OAuth URL so the user can connect their Google account
    (gives access to Gmail and Google Calendar).
    Call this when the user asks to connect Gmail or Google Calendar, or when
    any Google tool returns a 'google_not_connected' error.
    The URL should be sent to the user as a text message — they tap it on their
    phone, sign in with Google, and their account gets connected automatically.
    """
    client_id     = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri  = os.getenv("GOOGLE_REDIRECT_URI")

    if not all([client_id, client_secret, redirect_uri]):
        return {
            "success": False,
            "error": (
                "Google integration is not configured. "
                "GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI must be set."
            ),
        }

    # ── PKCE (RFC 7636 S256) ──────────────────────────────────────────────────
    code_verifier  = secrets.token_urlsafe(96)
    code_challenge = _b64url(hashlib.sha256(code_verifier.encode("ascii")).digest())

    state_payload = _b64url(json.dumps({"p": phone, "v": code_verifier}).encode())

    scopes = " ".join([
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/calendar",
    ])

    params = {
        "client_id":             client_id,
        "redirect_uri":          redirect_uri,
        "response_type":         "code",
        "scope":                 scopes,
        "access_type":           "offline",
        "prompt":                "consent",
        "state":                 state_payload,
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
    }
    auth_url = "https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode(params)

    return {"success": True, "url": auth_url}


# ── Gmail ─────────────────────────────────────────────────────────────────────

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
    Call generate_google_auth_url first if this returns 'google_not_connected'.

    Args:
        query:       Gmail search query. Default is unread emails.
                     Examples: "is:unread", "from:boss@company.com",
                     "subject:invoice", "is:unread newer_than:1d"
        max_results: Maximum number of emails to return (default 10).
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    creds, _tz, error = await _get_google_creds(phone)
    if error:
        return error

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
            return "google_not_connected"
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


# ── Google Calendar ───────────────────────────────────────────────────────────

@mcp.tool()
async def list_calendar_events(
    phone: str = None,
    max_results: int = 10,
    time_min: str = None,
    time_max: str = None,
) -> str:
    """
    List events from the user's Google Calendar.
    Call generate_google_auth_url first if this returns 'google_not_connected'.

    Args:
        max_results: Maximum number of events to return (default 10).
        time_min:    ISO 8601 datetime string for range start. Defaults to now.
        time_max:    ISO 8601 datetime string for range end. Set this when the
                     user asks about a specific day or week — e.g. for "Wednesday"
                     set time_min to Wednesday 00:00 and time_max to Wednesday 23:59.
                     Leave unset to get all upcoming events from time_min onwards.
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    creds, user_tz, error = await _get_google_creds(phone)
    if error:
        return error

    if not time_min:
        time_min = datetime.now(timezone.utc).isoformat()
    time_min = _localize_to_utc(time_min, user_tz)
    time_max = _localize_to_utc(time_max, user_tz) if time_max else None

    def _fetch():
        service = build("calendar", "v3", credentials=creds)
        params = dict(
            calendarId="primary",
            timeMin=time_min,
            maxResults=min(max_results, 50),
            singleEvents=True,
            orderBy="startTime",
        )
        if time_max:
            params["timeMax"] = time_max
        events_result = service.events().list(**params).execute()

        events = events_result.get("items", [])
        if not events:
            return []

        output = []
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            end   = event["end"].get("dateTime", event["end"].get("date"))
            output.append({
                "id":          event["id"],
                "title":       event.get("summary", "(no title)"),
                "start":       start,
                "end":         end,
                "location":    event.get("location", ""),
                "description": event.get("description", ""),
            })
        return output

    try:
        events = await asyncio.to_thread(_fetch)
    except HttpError as e:
        if e.resp.status in (401, 403):
            return "google_not_connected"
        return f"Calendar API error: {e}"

    if not events:
        return "No upcoming events found."

    lines = [f"{len(events)} upcoming event(s):\n"]
    for i, ev in enumerate(events, 1):
        line = f"{i}. {ev['title']}\n   {ev['start']} → {ev['end']}"
        if ev["location"]:
            line += f"\n   📍 {ev['location']}"
        if ev["description"]:
            line += f"\n   {ev['description']}"
        line += f"\n   ID: {ev['id']}"
        lines.append(line)

    return "\n\n".join(lines)


@mcp.tool()
async def create_calendar_event(
    phone: str = None,
    title: str = "",
    start_datetime: str = "",
    end_datetime: str = "",
    description: str = "",
    timezone: str = None,
    reminder_minutes: list[int] = [],
    reminder_method: str = "popup",
    sms_reminder_minutes: list[int] = [],
) -> str:
    """
    Create a new event on the user's Google Calendar.
    Call generate_google_auth_url first if this returns 'google_not_connected'.

    Args:
        title:                Event title/summary.
        start_datetime:       ISO 8601 datetime string in the user's local time
                              (e.g. '2026-03-10T14:00:00'). Do NOT convert to UTC.
        end_datetime:         ISO 8601 datetime string in the user's local time.
        description:          Optional event description.
        timezone:             IANA timezone (e.g. 'America/New_York'). Omit to use
                              the user's stored timezone automatically.
        reminder_minutes:     List of minutes before event for Google Calendar popup/email
                              reminders (e.g. [10, 60]). These only appear inside the
                              Google Calendar app — they do NOT send SMS messages.
                              If empty, the calendar's default reminder is used.
        reminder_method:      'popup' for on-screen notification or 'email' for email reminder.
        sms_reminder_minutes: List of minutes before event to send an SMS reminder via this
                              app (e.g. [15] for 15 min before, [480, 15] for 8 hrs before
                              AND 15 min before). Use this whenever the user says 'remind me'
                              or wants a text message reminder — this is separate from Google
                              Calendar's own reminders.
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    creds, user_tz, error = await _get_google_creds(phone)
    if error:
        return error

    if not timezone:
        timezone = user_tz

    reminders = (
        {
            "useDefault": False,
            "overrides": [
                {"method": reminder_method, "minutes": m}
                for m in reminder_minutes
            ],
        }
        if reminder_minutes
        else {"useDefault": True}
    )

    event_body = {
        "summary":     title,
        "description": description,
        "start":       {"dateTime": start_datetime, "timeZone": timezone},
        "end":         {"dateTime": end_datetime,   "timeZone": timezone},
        "reminders":   reminders,
    }

    def _create():
        service = build("calendar", "v3", credentials=creds)
        created = service.events().insert(calendarId="primary", body=event_body).execute()
        return created.get("htmlLink", ""), created.get("id", "")

    try:
        link, google_event_id = await asyncio.to_thread(_create)
    except HttpError as e:
        if e.resp.status in (401, 403):
            return "google_not_connected"
        return f"Calendar API error: {e}"

    result = f"Event created: {link}"

    # Always mirror Google Calendar reminders as SMS reminders — union both lists
    # so no Google reminder goes unmatched by an SMS reminder.
    # SMS fires SMS_LEAD_MINS earlier than the matching Google Calendar popup so
    # the text arrives before the on-screen notification.
    SMS_LEAD_MINS = 2
    all_sms_minutes = list(set(list(sms_reminder_minutes) + list(reminder_minutes)))

    # Create SMS reminders
    if all_sms_minutes:
        try:
            start_dt = datetime.fromisoformat(start_datetime)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=ZoneInfo(timezone))

            conn = await get_conn_for_phone(phone)
            try:
                user = await conn.fetchrow("SELECT id FROM users WHERE phone = $1", phone)
                if user:
                    scheduled = 0
                    for mins in all_sms_minutes:
                        remind_dt_utc = (
                            start_dt - timedelta(minutes=mins + SMS_LEAD_MINS)
                        ).astimezone(ZoneInfo("UTC"))
                        sms_message = f"Reminder: {title}"
                        # Skip if an identical reminder already exists (prevents duplicates
                        # if the tool is called more than once for the same event)
                        existing = await conn.fetchval(
                            "SELECT id FROM reminders WHERE user_id = $1 AND google_event_id = $2 AND remind_at = $3",
                            user["id"], google_event_id, remind_dt_utc,
                        )
                        if not existing:
                            await conn.execute(
                                """INSERT INTO reminders (user_id, message, remind_at, google_event_id)
                                   VALUES ($1, $2, $3, $4)""",
                                user["id"], sms_message, remind_dt_utc, google_event_id or None,
                            )
                            scheduled += 1
                    if scheduled:
                        result += f"\n{scheduled} SMS reminder(s) scheduled."
            finally:
                await release_conn(conn)
        except Exception as e:
            result += f"\nWarning: could not schedule SMS reminders: {e}"

    return result


@mcp.tool()
async def update_calendar_event(
    phone: str = None,
    event_id: str = "",
    title: str = None,
    start_datetime: str = None,
    end_datetime: str = None,
    description: str = None,
    timezone: str = None,
) -> str:
    """
    Update an existing Google Calendar event. Only the fields you provide will change.
    Use list_calendar_events to find the event ID first.
    Call generate_google_auth_url first if this returns 'google_not_connected'.

    Args:
        event_id:       ID of the event to update.
        title:          New event title (omit to keep existing).
        start_datetime: New start time in the user's local time, ISO 8601
                        (e.g. '2026-03-10T15:00:00'). Omit to keep existing.
        end_datetime:   New end time in the user's local time. Omit to keep existing.
        description:    New description. Omit to keep existing.
        timezone:       IANA timezone. Omit to use the user's stored timezone.
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    creds, user_tz, error = await _get_google_creds(phone)
    if error:
        return error

    if not timezone:
        timezone = user_tz

    def _update():
        service = build("calendar", "v3", credentials=creds)

        # Fetch current event so we only patch what changed and can capture old start
        existing = service.events().get(calendarId="primary", eventId=event_id).execute()
        old_start_str = (
            existing.get("start", {}).get("dateTime")
            or existing.get("start", {}).get("date")
        )

        patch = {}
        if title is not None:
            patch["summary"] = title
        if description is not None:
            patch["description"] = description
        if start_datetime is not None:
            patch["start"] = {"dateTime": start_datetime, "timeZone": timezone}
        if end_datetime is not None:
            patch["end"] = {"dateTime": end_datetime, "timeZone": timezone}

        # If only one of start/end is changing, carry over the other to keep duration intact
        if start_datetime is not None and end_datetime is None and "end" not in patch:
            patch["end"] = existing.get("end", {})
        if end_datetime is not None and start_datetime is None and "start" not in patch:
            patch["start"] = existing.get("start", {})

        updated = service.events().patch(
            calendarId="primary", eventId=event_id, body=patch
        ).execute()
        return updated.get("htmlLink", ""), old_start_str

    try:
        link, old_start_str = await asyncio.to_thread(_update)
    except HttpError as e:
        if e.resp.status in (401, 403):
            return "google_not_connected"
        if e.resp.status == 404:
            return f"Event {event_id} not found."
        return f"Calendar API error: {e}"

    result = f"Event updated: {link}"

    # If the event start moved, shift all linked SMS reminders by the same delta
    if start_datetime is not None and old_start_str:
        try:
            old_start_dt = datetime.fromisoformat(old_start_str.replace("Z", "+00:00"))
            new_start_dt = datetime.fromisoformat(start_datetime)
            if new_start_dt.tzinfo is None:
                new_start_dt = new_start_dt.replace(tzinfo=ZoneInfo(timezone))
            old_start_utc = old_start_dt.astimezone(ZoneInfo("UTC"))
            new_start_utc = new_start_dt.astimezone(ZoneInfo("UTC"))

            conn = await get_conn_for_phone(phone)
            try:
                user = await conn.fetchrow("SELECT id FROM users WHERE phone = $1", phone)
                if user:
                    linked = await conn.fetch(
                        """SELECT id, remind_at FROM reminders
                           WHERE user_id = $1 AND google_event_id = $2 AND fired = FALSE""",
                        user["id"], event_id,
                    )
                    shifted = 0
                    for reminder in linked:
                        # Preserve the exact offset each reminder had from the old start
                        offset = old_start_utc - reminder["remind_at"].replace(tzinfo=ZoneInfo("UTC"))
                        new_remind_at = new_start_utc - offset
                        await conn.execute(
                            "UPDATE reminders SET remind_at = $1 WHERE id = $2",
                            new_remind_at, reminder["id"],
                        )
                        shifted += 1
                    if shifted:
                        result += f"\n{shifted} SMS reminder(s) rescheduled."
            finally:
                await release_conn(conn)
        except Exception as e:
            result += f"\nWarning: could not reschedule SMS reminders: {e}"

    return result


@mcp.tool()
async def delete_calendar_event(
    phone: str = None,
    event_id: str = "",
) -> str:
    """
    Delete an event from the user's Google Calendar by its event ID.
    Use list_calendar_events to find the event ID first.
    Call generate_google_auth_url first if this returns 'google_not_connected'.
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    creds, _tz, error = await _get_google_creds(phone)
    if error:
        return error

    def _delete():
        service = build("calendar", "v3", credentials=creds)
        service.events().delete(calendarId="primary", eventId=event_id).execute()

    try:
        await asyncio.to_thread(_delete)
    except HttpError as e:
        if e.resp.status in (401, 403):
            return "google_not_connected"
        if e.resp.status == 404:
            return f"Event {event_id} not found."
        return f"Calendar API error: {e}"

    return f"Event {event_id} deleted successfully."
