"""Chrono Postgres connection utilities.

Reads credentials from environment variables (or a full URL) and returns a SQLAlchemy engine.

Env vars:
  - CHRONO_DB_URL (optional, full SQLAlchemy URL)
  - CHRONO_DB_USER
  - CHRONO_DB_PASSWORD
  - CHRONO_DB_HOST
  - CHRONO_DB_PORT (default: 5432)
  - CHRONO_DB_NAME (default: chrono)
"""

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

try:
    # Load .env if present for local development.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Dependency may be absent in some environments; ignore quietly.
    pass


def build_connection_url() -> str:
    """Build SQLAlchemy connection URL from env vars or return CHRONO_DB_URL."""
    db_url = os.getenv("CHRONO_DB_URL")
    if db_url:
        return db_url

    user = os.getenv("CHRONO_DB_USER")
    password = os.getenv("CHRONO_DB_PASSWORD")
    host = os.getenv("CHRONO_DB_HOST")
    port = os.getenv("CHRONO_DB_PORT", "5432")
    database = os.getenv("CHRONO_DB_NAME", "chrono")

    missing = [
        name
        for name, value in {
            "CHRONO_DB_USER": user,
            "CHRONO_DB_PASSWORD": password,
            "CHRONO_DB_HOST": host,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required env vars for DB connection: {', '.join(missing)}")

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def get_engine(echo: bool = False, pool_pre_ping: bool = True) -> Engine:
    """Create a SQLAlchemy engine using Chrono DB env vars."""
    url = build_connection_url()
    engine = create_engine(url, echo=echo, pool_pre_ping=pool_pre_ping)
    return engine


def test_connection(engine: Optional[Engine] = None) -> tuple[bool, Optional[str]]:
    """Execute a simple SELECT 1 to verify connectivity."""
    eng = engine or get_engine()
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:  # noqa: BLE001 - forward message
        return False, str(exc)
