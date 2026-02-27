from logging.config import fileConfig
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy import pool
from alembic import context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# -------------------------------------------------------------------
# Alembic Config object
# -------------------------------------------------------------------
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# -------------------------------------------------------------------
# Import SQLAlchemy Base and models
# -------------------------------------------------------------------
from cdss.heliot.api.db.session import Base
from cdss.heliot.api.models import api_key_db
from cdss.heliot.api.models import project_db

target_metadata = Base.metadata

# target_metadata for 'autogenerate' support
target_metadata = Base.metadata


def _get_database_url() -> str:
    """Read DB URL from env (single source of truth)."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set (required for Alembic migrations).")
    return database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = _get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # useful in case column types change
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    url = _get_database_url()

    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()