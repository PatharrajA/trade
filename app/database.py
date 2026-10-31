from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from .config import config
from .models import Base, RuntimeSetting, Setting


def _sqlite_path(url: str) -> Path | None:
    prefix = "sqlite:///"
    return Path(url[len(prefix) :]) if url.startswith(prefix) else None


db_path = _sqlite_path(config.database_url)
if db_path:
    db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    config.database_url,
    connect_args={"check_same_thread": False, "timeout": 30} if db_path else {},
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(connection, _record) -> None:
    if db_path:
        cursor = connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

DEFAULTS = {
    "paper_capital": "500000",
    "risk_per_trade_pct": "0.25",
    "daily_loss_limit_pct": "0.75",
    "max_trades_per_day": "5",
    "max_consecutive_losses": "2",
    "max_open_positions": "1",
    "live_max_lots": "1",
    "premium_emergency_stop_pct": "25",
    "reward_risk": "1.5",
    "max_spread_pct": "1.5",
    "max_quote_age_seconds": "3",
    "delta_min": "0.45",
    "delta_max": "0.60",
    "paper_slippage_pct": "0.05",
    "square_off_time": "15:20",
    "enabled_indices": "NIFTY,BANK_NIFTY,SENSEX",
    "active_broker": "UPSTOX",
    "default_historical_provider": "UPSTOX",
    "telegram_enabled": "false",
    "telegram_group_chat_id": "",
    "telegram_owner_chat_id": "",
    "telegram_daily_report": "true",
    "telegram_monthly_report": "true",
    "enforce_paper_session_requirement": "true",
    "required_paper_sessions": "20",
    "static_ip_verified": "false",
    "setup_shadow_enabled": "false",
    "setup_shadow_model_path": "",
    "setup_execution_enabled": "false",
    "setup_execution_model_path": "",
}


PROVIDER_TABLES = ("instruments", "candles", "signals", "orders", "trades", "backtest_trades")


def _migrate_provider_columns() -> None:
    """Small idempotent migration for the existing personal SQLite database."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        for table in PROVIDER_TABLES:
            if table not in existing_tables:
                continue
            columns = {column["name"] for column in inspector.get_columns(table)}
            if "provider" not in columns:
                connection.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN provider VARCHAR(16) NOT NULL DEFAULT 'UPSTOX'")
                )
            connection.execute(
                text(f"CREATE INDEX IF NOT EXISTS ix_{table}_provider ON {table} (provider)")
            )


def _migrate_research_columns() -> None:
    """Add role and verification columns without rewriting a personal SQLite database."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        if "candles" in tables:
            columns = {column["name"] for column in inspector.get_columns("candles")}
            if "data_role" not in columns:
                connection.execute(text("ALTER TABLE candles ADD COLUMN data_role VARCHAR(8) NOT NULL DEFAULT 'SPOT'"))
                connection.execute(text("UPDATE candles SET data_role = 'OPTION' WHERE underlying_code IS NULL"))
            connection.execute(
                text(
                    "UPDATE candles SET underlying_code = ("
                    "SELECT instruments.underlying_code FROM instruments "
                    "WHERE instruments.provider = candles.provider "
                    "AND instruments.instrument_key = candles.instrument_key LIMIT 1"
                    ") WHERE data_role = 'OPTION' AND underlying_code IS NULL"
                )
            )
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_candles_data_role ON candles (data_role)"))
        if "instruments" in tables:
            columns = {column["name"] for column in inspector.get_columns("instruments")}
            if "instrument_kind" not in columns:
                connection.execute(text("ALTER TABLE instruments ADD COLUMN instrument_kind VARCHAR(8) NOT NULL DEFAULT 'OPTION'"))
            if "metadata_verified" not in columns:
                connection.execute(text("ALTER TABLE instruments ADD COLUMN metadata_verified BOOLEAN NOT NULL DEFAULT 1"))
            if "metadata_source" not in columns:
                connection.execute(text("ALTER TABLE instruments ADD COLUMN metadata_source VARCHAR(32) NOT NULL DEFAULT 'BROKER'"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_instruments_instrument_kind ON instruments (instrument_kind)"))


def _add_column(connection, table: str, columns: set[str], name: str, definition: str) -> None:
    if name not in columns:
        connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}"))


def _migrate_audit_columns() -> None:
    """Add nullable audit fields safely to existing SQLite installations."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        if "signals" in tables:
            columns = {column["name"] for column in inspector.get_columns("signals")}
            _add_column(connection, "signals", columns, "decision_json", "TEXT")
        if "trades" in tables:
            columns = {column["name"] for column in inspector.get_columns("trades")}
            for name, definition in (
                ("entry_bid", "NUMERIC(14, 2)"),
                ("entry_ask", "NUMERIC(14, 2)"),
                ("entry_slippage", "NUMERIC(14, 4)"),
                ("entry_delta", "FLOAT"),
                ("entry_iv", "FLOAT"),
                ("risk_budget", "NUMERIC(14, 2)"),
            ):
                _add_column(connection, "trades", columns, name, definition)
        if "backtest_trades" in tables:
            columns = {column["name"] for column in inspector.get_columns("backtest_trades")}
            _add_column(connection, "backtest_trades", columns, "effective_slippage_pct", "NUMERIC(10, 4)")
            _add_column(connection, "backtest_trades", columns, "metadata_source", "VARCHAR(32)")
            _add_column(connection, "backtest_trades", columns, "resolution_timeframe", "VARCHAR(16) NOT NULL DEFAULT '5minute'")


def init_db() -> None:
    Base.metadata.create_all(engine)
    _migrate_provider_columns()
    _migrate_research_columns()
    _migrate_audit_columns()
    with SessionLocal.begin() as session:
        if session.get(RuntimeSetting, 1) is None:
            session.add(RuntimeSetting(id=1))
        existing = set(session.scalars(select(Setting.key)).all())
        session.add_all(Setting(key=key, value=value) for key, value in DEFAULTS.items() if key not in existing)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def settings_dict(session: Session) -> dict[str, str]:
    return {row.key: row.value for row in session.scalars(select(Setting)).all()}


def database_health(session: Session) -> tuple[bool, str]:
    try:
        result = session.execute(text("PRAGMA quick_check")).scalar_one()
        return result == "ok", str(result)
    except Exception as exc:
        return False, str(exc)

