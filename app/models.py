from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class RuntimeSetting(Base):
    __tablename__ = "runtime_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    execution_mode: Mapped[str] = mapped_column(String(16), default="STOPPED")
    entries_paused: Mapped[bool] = mapped_column(Boolean, default=True)
    change_reason: Mapped[str] = mapped_column(String(255), default="Initial setup")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Instrument(Base):
    __tablename__ = "instruments"
    __table_args__ = (
        UniqueConstraint("provider", "instrument_key", "valid_from", name="uq_provider_instrument_version"),
        Index("ix_instrument_selection", "provider", "underlying_code", "expiry", "option_type", "strike"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(16), default="UPSTOX", index=True)
    instrument_key: Mapped[str] = mapped_column(String(100))
    underlying_code: Mapped[str] = mapped_column(String(24), index=True)
    exchange: Mapped[str] = mapped_column(String(16))
    trading_symbol: Mapped[str] = mapped_column(String(100))
    option_type: Mapped[str] = mapped_column(String(2))
    strike: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    expiry: Mapped[date] = mapped_column(Date)
    lot_size: Mapped[int] = mapped_column(Integer)
    tick_size: Mapped[Decimal] = mapped_column(Numeric(10, 4), default=Decimal("0.05"))
    valid_from: Mapped[date] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    instrument_kind: Mapped[str] = mapped_column(String(8), default="OPTION", index=True)
    metadata_verified: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_source: Mapped[str] = mapped_column(String(32), default="BROKER")
    downloaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Candle(Base):
    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint("provider", "instrument_key", "timeframe", "timestamp", name="uq_provider_candle"),
        Index("ix_candle_replay", "provider", "instrument_key", "timeframe", "timestamp"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(16), default="UPSTOX", index=True)
    instrument_key: Mapped[str] = mapped_column(String(100))
    underlying_code: Mapped[str | None] = mapped_column(String(24), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), default="5minute")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    open: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    high: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    low: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    close: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    volume: Mapped[int] = mapped_column(Integer, default=0)
    open_interest: Mapped[int] = mapped_column(Integer, default=0)
    bid: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    ask: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    india_vix: Mapped[float | None] = mapped_column(Float)
    futures_basis_pct: Mapped[float | None] = mapped_column(Float)
    pcr: Mapped[float | None] = mapped_column(Float)
    source_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    data_role: Mapped[str] = mapped_column(String(8), default="SPOT", index=True)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(16), default="UPSTOX", index=True)
    mode: Mapped[str] = mapped_column(String(16), index=True)
    underlying_code: Mapped[str] = mapped_column(String(24), index=True)
    candle_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    direction: Mapped[str] = mapped_column(String(16))
    regime: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    selected_instrument: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(24), default="PENDING")
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    decision_json: Mapped[str | None] = mapped_column(Text)
    strategy_version: Mapped[str] = mapped_column(String(32), default="strategy-v1")
    model_version: Mapped[str] = mapped_column(String(32), default="model-v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint("signal_id", "purpose", name="uq_signal_order_purpose"),
        UniqueConstraint("provider", "broker_order_id", name="uq_provider_broker_order_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(16), default="UPSTOX", index=True)
    signal_id: Mapped[str] = mapped_column(ForeignKey("signals.id"), index=True)
    mode: Mapped[str] = mapped_column(String(16), index=True)
    purpose: Mapped[str] = mapped_column(String(8))
    broker_order_id: Mapped[str | None] = mapped_column(String(100))
    tag: Mapped[str] = mapped_column(String(40))
    instrument_key: Mapped[str] = mapped_column(String(100))
    trading_symbol: Mapped[str] = mapped_column(String(100))
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[int] = mapped_column(Integer)
    requested_price: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    average_price: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="CREATED")
    status_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(16), default="UPSTOX", index=True)
    signal_id: Mapped[str] = mapped_column(ForeignKey("signals.id"), unique=True)
    mode: Mapped[str] = mapped_column(String(16), index=True)
    underlying_code: Mapped[str] = mapped_column(String(24), index=True)
    instrument_key: Mapped[str] = mapped_column(String(100))
    trading_symbol: Mapped[str] = mapped_column(String(100))
    option_type: Mapped[str] = mapped_column(String(2))
    moneyness: Mapped[str] = mapped_column(String(8))
    expiry: Mapped[date] = mapped_column(Date)
    quantity: Mapped[int] = mapped_column(Integer)
    lot_size: Mapped[int] = mapped_column(Integer)
    tick_size: Mapped[Decimal] = mapped_column(Numeric(10, 4), default=Decimal("0.05"))
    entry_price: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    entry_bid: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    entry_ask: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    entry_slippage: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    entry_delta: Mapped[float | None] = mapped_column(Float)
    entry_iv: Mapped[float | None] = mapped_column(Float)
    risk_budget: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    current_bid: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    underlying_entry: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    underlying_stop: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    underlying_target: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    premium_emergency_stop: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    gross_pnl: Mapped[Decimal] = mapped_column(Numeric(14, 2), default=0)
    charges: Mapped[Decimal] = mapped_column(Numeric(14, 2), default=0)
    net_pnl: Mapped[Decimal] = mapped_column(Numeric(14, 2), default=0)
    status: Mapped[str] = mapped_column(String(16), default="OPEN", index=True)
    exit_reason: Mapped[str | None] = mapped_column(String(64))
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(24), default="QUEUED", index=True)
    configuration_json: Mapped[str] = mapped_column(Text)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)
    strategy_version: Mapped[str] = mapped_column(String(32), default="strategy-v1")
    model_version: Mapped[str] = mapped_column(String(32), default="model-v1")
    summary_json: Mapped[str | None] = mapped_column(Text)
    warnings_json: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(16), default="UPSTOX", index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("backtest_runs.id"), index=True)
    signal_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    exit_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    underlying_code: Mapped[str] = mapped_column(String(24))
    instrument_key: Mapped[str] = mapped_column(String(100))
    option_type: Mapped[str] = mapped_column(String(2))
    moneyness: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int] = mapped_column(Integer)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    exit_price: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    gross_pnl: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    charges: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    net_pnl: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    exit_reason: Mapped[str] = mapped_column(String(64))
    effective_slippage_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    metadata_source: Mapped[str | None] = mapped_column(String(32))
    resolution_timeframe: Mapped[str] = mapped_column(String(16), default="5minute")


class ResearchSyncRun(Base):
    __tablename__ = "research_sync_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(24), default="QUEUED", index=True)
    configuration_json: Mapped[str] = mapped_column(Text)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)
    current_step: Mapped[str] = mapped_column(String(255), default="Queued")
    result_json: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(16), default="CANDIDATE", index=True)
    model_path: Mapped[str] = mapped_column(Text)
    metadata_path: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class NotificationOutbox(Base):
    __tablename__ = "notification_outbox"
    __table_args__ = (UniqueConstraint("dedupe_key", name="uq_notification_dedupe"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    destination: Mapped[str] = mapped_column(String(16))
    event_type: Mapped[str] = mapped_column(String(40))
    message: Mapped[str] = mapped_column(Text)
    dedupe_key: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(20), default="PENDING", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)


class SystemEvent(Base):
    __tablename__ = "system_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    severity: Mapped[str] = mapped_column(String(16), default="INFO", index=True)
    component: Mapped[str] = mapped_column(String(40), index=True)
    event_type: Mapped[str] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

