import logging
import os
from datetime import datetime, timedelta, timezone

import deps
from .agent import get_or_create_user, get_user_skills, get_user_system_prompt
from config import TWILIO_PHONE_NUMBER
from .database import get_db_pool
from .llm_client import create_message

logger = logging.getLogger(__name__)


async def get_due_reminders() -> list[dict]:
    """
    Returns all due reminders across all users for the scheduler.
    Includes the user's phone number so the scheduler knows who to text.
    Only call this from the scheduler.
    """
    now_utc = datetime.now(timezone.utc)

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            due = await conn.fetch(
                """
                SELECT r.id, r.message, r.remind_at, r.recurrence, u.phone
                FROM reminders r
                JOIN users u ON u.id = r.user_id
                WHERE r.fired = FALSE AND r.remind_at <= $1
                FOR UPDATE OF r
                """,
                now_utc,
            )

            due_list = []
            for row in due:
                due_list.append({
                    "id": row["id"],
                    "message": row["message"],
                    "remind_at": row["remind_at"].isoformat(),
                    "recurrence": row["recurrence"],
                    "phone": row["phone"],
                })

                if row["recurrence"] == "daily":
                    next_time = row["remind_at"] + timedelta(days=1)
                    await conn.execute(
                        "UPDATE reminders SET remind_at = $1 WHERE id = $2",
                        next_time, row["id"],
                    )
                elif row["recurrence"] == "weekly":
                    next_time = row["remind_at"] + timedelta(weeks=1)
                    await conn.execute(
                        "UPDATE reminders SET remind_at = $1 WHERE id = $2",
                        next_time, row["id"],
                    )
                elif row["recurrence"] == "monthly":
                    from dateutil.relativedelta import relativedelta
                    next_time = row["remind_at"] + relativedelta(months=1)
                    await conn.execute(
                        "UPDATE reminders SET remind_at = $1 WHERE id = $2",
                        next_time, row["id"],
                    )
                else:
                    await conn.execute(
                        "UPDATE reminders SET fired = TRUE WHERE id = $1", row["id"]
                    )

    return due_list


async def personalize_reminder(raw_message: str, user: dict) -> str:
    """Run a reminder message through Claude to make it feel natural and personalized."""
    skills = await get_user_skills(user.get("phone", ""))
    system = get_user_system_prompt(user, skills)
    pool   = await get_db_pool()

    response = await create_message(
        deps.anthropic_client, pool,
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


async def fire_due_reminders():
    """Fires due reminders — sends to each user's own phone number."""
    try:
        due = await get_due_reminders()
        logger.info(f"Parsed {len(due)} due reminders")

        if len(due) == 0:
            logger.info("No due reminders")
            return

        use_whatsapp = os.getenv("USE_WHATSAPP", "false").lower() == "true"

        for reminder in due:
            phone = reminder["phone"]
            logger.info(f"Firing reminder for {phone}: {reminder['message']}")

            user = await get_or_create_user(phone)
            if not user.get("found"):
                logger.warning(f"User not found for phone {phone} while firing reminder")
                personalized_message = f"⏰ Reminder: {reminder['message']}"
            else:
                personalized_message = await personalize_reminder(reminder["message"], user)

            from_number = f"whatsapp:{TWILIO_PHONE_NUMBER}" if use_whatsapp else TWILIO_PHONE_NUMBER
            to_number   = f"whatsapp:{phone}" if use_whatsapp else phone

            deps.twilio_client.messages.create(
                body=personalized_message,
                from_=from_number,
                to=to_number,
            )

            await deps.conversation_manager.add_message(phone, "assistant", personalized_message)
            logger.info(f"Reminder sent to {to_number}")
    except Exception as e:
        logger.error(f"Scheduler error: {e}", exc_info=True)
