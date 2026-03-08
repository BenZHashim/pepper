"""
llm_client.py
Centralised Anthropic API wrapper with per-call usage tracking.

All Anthropic API calls in the project go through create_message().
It records input/output tokens and estimated cost to the llm_usage table.

Both the api and mcp containers import this module. Each passes its own
asyncpg pool — both connect to the same database, so rows from either
container land in the same llm_usage table.
"""

import asyncio
import logging
from decimal import Decimal

import anthropic
import asyncpg

logger = logging.getLogger(__name__)

# ─── Pricing — USD per million tokens ────────────────────────────────────────
# Update these when Anthropic changes pricing.

_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":  {"input": 1.00,  "output": 5.00},
    "claude-sonnet-4-5": {"input": 3.00,  "output": 15.00},
    "claude-opus-4-5":   {"input": 15.00, "output": 75.00},
}

_FALLBACK_PRICING = {"input": 1.00, "output": 5.00}  # unknown models


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    p = _PRICING.get(model, _FALLBACK_PRICING)
    raw = (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
    return Decimal(str(round(raw, 6)))


async def create_message(
    client: anthropic.Anthropic,
    pool: asyncpg.Pool,
    *,
    purpose: str,
    user_id: int | None = None,
    model: str,
    max_tokens: int,
    messages: list,
    system: str | None = None,
    tools: list | None = None,
    tool_choice: dict | None = None,
) -> anthropic.types.Message:
    """
    Call the Anthropic Messages API and record token usage + estimated cost.

    Args:
        client:      Anthropic client instance.
        pool:        asyncpg pool — works from either the api or mcp container
                     since both connect to the same database.
        purpose:     Call-site label for reporting, e.g. 'main_agent', 'summarization'.
        user_id:     Database user ID, or None for pre-registration / cross-user calls.
        model:       Model name, e.g. 'claude-haiku-4-5'.
        max_tokens:  Maximum tokens to generate.
        messages:    Conversation messages list.
        system:      Optional system prompt string.
        tools:       Optional list of tool definitions.
        tool_choice: Optional tool-choice constraint dict.
    """
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system is not None:
        kwargs["system"] = system
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    response = client.messages.create(**kwargs)

    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost          = _estimate_cost(model, input_tokens, output_tokens)

    logger.info(
        f"[llm:{purpose}] in={input_tokens} out={output_tokens} cost=${cost:.6f}"
    )

    asyncio.create_task(_record_usage(pool, user_id, model, purpose, input_tokens, output_tokens, cost))

    return response


async def _record_usage(
    pool: asyncpg.Pool,
    user_id: int | None,
    model: str,
    purpose: str,
    input_tokens: int,
    output_tokens: int,
    cost: Decimal,
) -> None:
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_usage
                    (user_id, model, purpose, input_tokens, output_tokens, cost_usd)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                user_id, model, purpose, input_tokens, output_tokens, float(cost),
            )
    except Exception as exc:
        logger.warning(f"Failed to record LLM usage for [{purpose}]: {exc}")
