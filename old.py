from mcp.server.fastmcp import FastMCP
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.discovery import Resource
import os, pickle, logging, concurrent.futures
from typing import Optional

logging.basicConfig(
    filename=r'C:\Users\Ben\OneDrive\Desktop\Personal Projects\MCP\mcp_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s'
)

mcp = FastMCP("GoogleCalendar")
SCOPES: list[str] = ['https://www.googleapis.com/auth/calendar']
TOKEN_PATH: str = r'C:\Users\Ben\OneDrive\Desktop\Personal Projects\MCP\token.pickle'


def get_calendar_service() -> Optional[Resource]:
    logging.info("--- get_calendar_service called ---")
    logging.info(f"token.pickle exists: {os.path.exists(TOKEN_PATH)}")

    creds = None
    if os.path.exists(TOKEN_PATH):
        logging.info("Loading token.pickle...")
        with open(TOKEN_PATH, 'rb') as token:
            creds = pickle.load(token)
        logging.info(f"Creds loaded. Valid: {creds.valid}, Expired: {creds.expired}")

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logging.info("Refreshing expired credentials...")
            creds.refresh(Request())
            logging.info("Credentials refreshed.")
        else:
            logging.error("No valid creds — run auth.py manually")
            return None

    logging.info("Building calendar service...")
    service: Resource = build('calendar', 'v3', credentials=creds)
    logging.info("Calendar service built successfully.")
    return service


def run_with_timeout(fn, *args, timeout: int = 10) -> str:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logging.error("TIMED OUT after 10s")
            return "TIMED OUT after 10s. Check mcp_debug.log to see which step it got stuck on."
        except Exception as e:
            logging.error(f"Exception: {type(e).__name__}: {e}")
            return f"FAILED with error: {type(e).__name__}: {e}"


def _list_events_inner(max_results: int, time_min: str) -> str:
    logging.info("Step 1: Getting calendar service...")
    service: Optional[Resource] = get_calendar_service()

    if service is None:
        return "FAILED at Step 1: Could not build calendar service. Run auth.py manually first."

    logging.info("Step 2: Fetching events...")
    events_result: dict = service.events().list(
        calendarId='primary',
        timeMin=time_min,
        maxResults=max_results,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    events: list[dict] = events_result.get('items', [])
    if not events:
        return "No upcoming events found."

    output: list[str] = []
    for event in events:
        start: str = event['start'].get('dateTime', event['start'].get('date'))
        end: str = event['end'].get('dateTime', event['end'].get('date'))
        description: str = event.get('description', 'No description')
        location: str = event.get('location', 'No location')
        reminders: dict = event.get('reminders', {})

        if reminders.get('useDefault'):
            reminder_str: str = "Default reminder"
        else:
            overrides: list[dict] = reminders.get('overrides', [])
            reminder_str = ", ".join(
                [f"{r['minutes']} min via {r['method']}" for r in overrides]
            ) if overrides else "No reminders"

        output.append(
            f"- {event['summary']}\n"
            f"  Start: {start}\n"
            f"  End: {end}\n"
            f"  Location: {location}\n"
            f"  Description: {description}\n"
            f"  Reminders: {reminder_str}\n"
            f"  ID: {event['id']}"
        )

    return "\n\n".join(output)


def _create_event_inner(
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str,
    timezone: str,
    reminder_minutes: list[int],
    reminder_method: str
) -> str:
    logging.info("Step 1: Getting calendar service...")
    service: Optional[Resource] = get_calendar_service()

    if service is None:
        return "FAILED at Step 1: Could not build calendar service. Run auth.py manually first."

    logging.info("Step 2: Building event body...")

    reminders: dict = {
        'useDefault': False,
        'overrides': [
            {'method': reminder_method, 'minutes': minutes}
            for minutes in reminder_minutes
        ]
    } if reminder_minutes else {'useDefault': True}

    event: dict = {
        'summary': title,
        'description': description,
        'start': {'dateTime': start_datetime, 'timeZone': timezone},
        'end':   {'dateTime': end_datetime,   'timeZone': timezone},
        'reminders': reminders
    }

    logging.info("Step 3: Inserting event via API...")
    created: dict = service.events().insert(calendarId='primary', body=event).execute()

    logging.info(f"Step 4: Success: {created.get('htmlLink')}")
    return f"Event created: {created.get('htmlLink')}"

def _delete_event_inner(event_id: str) -> str:
    logging.info("Step 1: Getting calendar service...")
    service: Optional[Resource] = get_calendar_service()

    if service is None:
        return "FAILED at Step 1: Could not build calendar service. Run auth.py manually first."

    logging.info(f"Step 2: Deleting event {event_id}...")
    service.events().delete(calendarId='primary', eventId=event_id).execute()
    logging.info("Step 3: Event deleted successfully.")
    return f"Event {event_id} deleted successfully."



@mcp.tool()
def create_event(
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    timezone: str = "America/New_York",
    reminder_minutes: list[int] = [],
    reminder_method: str = "popup"
) -> str:
    """Schedule a new event on Google Calendar.
    
    reminder_minutes: list of how many minutes before the event to remind (e.g. [10, 60] for 10 min and 1 hour before).
    reminder_method: 'popup' for on-screen notification or 'email' for email reminder.
    If reminder_minutes is empty, the calendar's default reminder will be used.
    """
    logging.info(f"create_event called: {title}, {start_datetime}, {end_datetime}, reminders: {reminder_minutes}")
    return run_with_timeout(_create_event_inner, title, start_datetime, end_datetime, description, timezone, reminder_minutes, reminder_method)


@mcp.tool()
def create_event(
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    timezone: str = "America/New_York"
) -> str:
    """Schedule a new event on Google Calendar."""
    logging.info(f"create_event called: {title}, {start_datetime}, {end_datetime}")
    return run_with_timeout(_create_event_inner, title, start_datetime, end_datetime, description, timezone)


@mcp.tool()
def list_events(
    max_results: int = 10,
    time_min: str = "2025-01-01T00:00:00Z"
) -> str:
    """List upcoming events from Google Calendar. time_min should be an ISO 8601 datetime string."""
    logging.info(f"list_events called: max_results={max_results}, time_min={time_min}")
    return run_with_timeout(_list_events_inner, max_results, time_min)


@mcp.tool()
def delete_event(event_id: str) -> str:
    """Delete an event from Google Calendar by its event ID."""
    logging.info(f"delete_event called: {event_id}")
    return run_with_timeout(_delete_event_inner, event_id)


if __name__ == "__main__":
    mcp.run()