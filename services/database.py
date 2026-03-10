import os

import asyncpg

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
