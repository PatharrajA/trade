from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd

from app.setup_quality import (
    SETUP_FEATURES,
    SETUP_INDICES,
    SetupContract,
    inspect_setup_training_data,
    option_price_labels,
    relative_index_code,
    select_probability_threshold,
    selection_metrics,
    setup_candidates,
    setup_rule_candidates,
)
from app.models import Candle, Instrument


def _trend_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    start = datetime(2025, 1, 1, 3, 45, tzinfo=UTC)
    spot_rows: list[dict] = []
    future_rows: list[dict] = []
    for day in range(4):
        for bar in range(40):
            stamp = start + timedelta(days=day, minutes=bar * 5)
            base = 100 + day * 3
            price = base if bar < 4 else base + (bar - 3) * 1.5
            spot_rows.append(
                {
                    "timestamp": stamp,
                    "open": price,
                    "high": price + 0.8,
                    "low": price - 0.8,
                    "close": price + 0.3,
                    "volume": 0,
                }
            )
            future_rows.append(
                {
                    "timestamp": stamp,
                    "open": price + 2,
                    "high": price + 2.9,
                    "low": price + 1.1,
                    "close": price + 2.4,
                    "volume": 10_000 + bar * 100,
                }
            )
    return pd.DataFrame(spot_rows), pd.DataFrame(future_rows)


def test_setup_candidates_are_rule_confirmed_and_triple_barrier_labelled():
    spot, futures = _trend_frames()
    candidates = setup_candidates(
        spot,
        futures,
        "NIFTY",
        contract=SetupContract(stop_atr=1.0, target_atr=1.0, timeout_bars=6),
    )

    assert not candidates.empty
    assert set(candidates.direction) == {"CE"}
    assert set(candidates.label) <= {"TAKE", "SKIP"}
    assert set(candidates.exit_reason) <= {"TARGET", "STOP", "TIMEOUT"}
    assert candidates[SETUP_FEATURES].notna().all().all()
    assert (candidates.bars_held <= 6).all()
    assert (candidates.loc[candidates.label == "TAKE", "exit_reason"] == "TARGET").all()
    local_days = pd.to_datetime(candidates.timestamp, utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    assert candidates.groupby(local_days).size().max() == 1


def test_setup_candidates_skip_unresolved_path_after_futures_bars_end():
    spot, futures = _trend_frames()
    last_day = date(2025, 1, 4)
    futures = futures[
        (futures.timestamp.dt.date != last_day)
        | (futures.timestamp <= pd.Timestamp("2025-01-04T06:35:00Z"))
    ]

    candidates = setup_candidates(
        spot, futures, "NIFTY",
        contract=SetupContract(stop_atr=100, target_atr=100, timeout_bars=50),
    )

    assert not (candidates.timestamp.dt.date == last_day).any()
    assert len(candidates) == 2
    assert set(candidates.entry_timestamp).issubset(set(futures.timestamp))
    assert set(candidates.exit_timestamp).issubset(set(futures.timestamp))


def test_setup_candidates_keep_target_reached_before_futures_gap():
    spot, futures = _trend_frames()
    futures = futures[
        (futures.timestamp.dt.date != date(2025, 1, 4))
        | (futures.timestamp <= pd.Timestamp("2025-01-04T05:30:00Z"))
    ]

    candidates = setup_candidates(
        spot, futures, "NIFTY",
        contract=SetupContract(stop_atr=1, target_atr=1, timeout_bars=6),
    )

    last_day_candidate = candidates[
        candidates.timestamp.dt.date == date(2025, 1, 4)
    ].iloc[0]
    assert last_day_candidate.exit_reason == "TARGET"
    assert last_day_candidate.exit_timestamp == pd.Timestamp("2025-01-04T05:30:00Z")


def test_sensex_has_its_own_candidate_stream_and_nifty_relative_reference():
    spot, futures = _trend_frames()

    candidates = setup_candidates(
        spot,
        futures,
        "SENSEX",
        relative_frame=spot,
        contract=SetupContract(stop_atr=1.0, target_atr=1.0, timeout_bars=6),
    )

    assert "SENSEX" in SETUP_INDICES
    assert relative_index_code("SENSEX") == "NIFTY"
    assert not candidates.empty
    assert set(candidates.underlying_code) == {"SENSEX"}


def test_rule_candidates_do_not_start_after_square_off():
    spot, futures = _trend_frames()
    spot["timestamp"] += timedelta(hours=5)
    futures["timestamp"] += timedelta(hours=5)

    candidates = setup_rule_candidates(spot, futures, "NIFTY")

    assert candidates.empty


def test_selection_metrics_compare_take_skip_and_rules_only():
    dataset = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-07-01T04:00:00Z", "2025-07-02T04:00:00Z", "2025-08-01T04:00:00Z", "2025-08-02T04:00:00Z"]
            ),
            "label": ["TAKE", "SKIP", "TAKE", "SKIP"],
            "direction": ["CE", "PE", "CE", "PE"],
            "net_return": [0.02, -0.01, 0.03, -0.02],
        }
    )
    probabilities = np.asarray([0.8, 0.2, 0.75, 0.3])

    result = selection_metrics(dataset, probabilities, 0.60)

    assert result["accepted"] == 2
    assert result["take_precision"] == 1.0
    assert result["selected"]["profit_factor"] is None
    assert result["selected"]["expectancy"] == 0.025
    assert result["rules_only"]["profit_factor"] == 1.6667
    assert result["skipped"]["net_return"] == -0.03


def test_threshold_selection_requires_a_meaningful_accepted_sample():
    size = 100
    dataset = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=size, freq="D", tz="UTC"),
            "label": ["TAKE" if index % 2 == 0 else "SKIP" for index in range(size)],
            "direction": ["CE"] * size,
            "net_return": [0.02 if index % 2 == 0 else -0.01 for index in range(size)],
        }
    )
    probabilities = np.asarray([0.8 if label == "TAKE" else 0.3 for label in dataset.label])

    threshold = select_probability_threshold(dataset, probabilities)

    assert 0.50 <= threshold <= 0.80


def test_option_price_labels_use_verified_contract_and_after_cost_pnl(session):
    signal = pd.Timestamp("2025-07-01T04:00:00Z")
    entry = pd.Timestamp("2025-07-01T04:05:00Z")
    exit_ = pd.Timestamp("2025-07-01T04:20:00Z")
    session.add(
        Instrument(
            provider="GROWW",
            instrument_key="nifty-ce",
            underlying_code="NIFTY",
            exchange="NSE",
            trading_symbol="NIFTY TEST CE",
            option_type="CE",
            strike=Decimal("24000"),
            expiry=date(2025, 7, 31),
            lot_size=65,
            tick_size=Decimal("0.05"),
            valid_from=date(2025, 1, 1),
            instrument_kind="OPTION",
            metadata_verified=True,
        )
    )
    session.add_all(
        [
            Candle(
                provider="GROWW",
                instrument_key="nifty-ce",
                underlying_code="NIFTY",
                timeframe="5minute",
                timestamp=entry.to_pydatetime(),
                open=100,
                high=102,
                low=99,
                close=101,
                volume=100,
                open_interest=100,
                bid=Decimal("99.5"),
                ask=Decimal("100.5"),
                source_verified=True,
                data_role="OPTION",
            ),
            Candle(
                provider="GROWW",
                instrument_key="nifty-ce",
                underlying_code="NIFTY",
                timeframe="5minute",
                timestamp=exit_.to_pydatetime(),
                open=120,
                high=123,
                low=119,
                close=122,
                volume=100,
                open_interest=100,
                bid=Decimal("121.5"),
                ask=Decimal("122.5"),
                source_verified=True,
                data_role="OPTION",
            ),
        ]
    )
    session.add_all([
        Candle(
            provider="GROWW", instrument_key="nifty-spot",
            underlying_code="NIFTY", timeframe="5minute",
            timestamp=stamp.to_pydatetime(), open=price,
            high=price + 2, low=price - 2, close=price,
            source_verified=True, data_role="SPOT",
        )
        for stamp, price in ((entry, 24010), (exit_, 24030))
    ])
    session.commit()
    dataset = pd.DataFrame(
        [
            {
                "timestamp": signal,
                "entry_timestamp": entry,
                "exit_timestamp": exit_,
                "underlying_code": "NIFTY",
                "direction": "CE",
                "signal_close": 24010.0,
                "signal_atr": 10.0,
            }
        ]
    )

    labelled, coverage = option_price_labels(
        session, dataset, provider="GROWW", cost_model="GROWW", slippage_pct=0.05
    )

    assert coverage["coverage"] == 1.0
    assert labelled.iloc[0].label_source == "VERIFIED_OPTION_CANDLES"
    assert labelled.iloc[0].option_moneyness == "ATM"
    assert labelled.iloc[0].option_net_pnl > 0
    assert labelled.iloc[0].label == "TAKE"
    assert labelled.iloc[0].exit_reason == "UNDERLYING_TARGET"
    assert coverage["estimated_fill_rows"] == 0


def test_option_replay_uses_premium_stop_and_profitable_time_exit(session):
    session.add(Instrument(
        provider="GROWW", instrument_key="replay-ce", underlying_code="NIFTY",
        exchange="NSE", trading_symbol="NIFTY REPLAY CE", option_type="CE",
        strike=Decimal("24000"), expiry=date(2025, 7, 31), lot_size=65,
        tick_size=Decimal("0.05"), valid_from=date(2025, 1, 1),
        instrument_kind="OPTION", metadata_verified=True,
    ))
    signals = [
        pd.Timestamp("2025-07-01T04:00:00Z"),
        pd.Timestamp("2025-07-02T08:30:00Z"),
        pd.Timestamp("2025-07-03T08:30:00Z"),
    ]
    option_rows = [
        (pd.Timestamp("2025-07-01T04:05:00Z"), 100, 99, 100),
        (pd.Timestamp("2025-07-01T04:10:00Z"), 70, 69, 70),
        *[
            (stamp, 100 + offset, 99 + offset, 100 + offset)
            for offset, stamp in enumerate(pd.date_range(
                "2025-07-02T08:35:00Z", "2025-07-02T09:50:00Z", freq="5min"
            ))
        ],
        (pd.Timestamp("2025-07-03T08:35:00Z"), 100, 99, 100),
    ]
    for stamp, option_open, option_low, option_close in option_rows:
        session.add_all([
            Candle(
                provider="GROWW", instrument_key="replay-ce",
                underlying_code="NIFTY", timeframe="5minute",
                timestamp=stamp.to_pydatetime(), open=option_open,
                high=option_open + 1, low=option_low, close=option_close,
                volume=100, open_interest=100, source_verified=True,
                data_role="OPTION",
            ),
            Candle(
                provider="GROWW", instrument_key="replay-spot",
                underlying_code="NIFTY", timeframe="5minute",
                timestamp=stamp.to_pydatetime(), open=24000,
                high=24001, low=23999, close=24000,
                source_verified=True, data_role="SPOT",
            ),
        ])
    session.commit()
    dataset = pd.DataFrame([
        {
            "timestamp": stamp, "underlying_code": "NIFTY", "direction": "CE",
            "signal_close": 24000.0, "signal_atr": 1000.0,
        }
        for stamp in signals
    ])

    labelled, coverage = option_price_labels(
        session, dataset, provider="GROWW", cost_model="GROWW"
    )

    assert labelled.label.tolist() == ["SKIP", "TAKE"]
    assert labelled.exit_reason.tolist() == ["PREMIUM_STOP", "TIME_EXIT"]
    assert labelled.iloc[1].exit_timestamp == pd.Timestamp("2025-07-02T09:50:00Z")
    assert coverage["estimated_fill_rows"] == 2
    assert coverage["rejections"] == {"OPTION_TIME_EXIT_BAR_MISSING": 1}


def test_training_data_check_reports_missing_verified_candles(session):
    result = inspect_setup_training_data(
        str(session.get_bind().url), ["SENSEX"],
        date(2023, 1, 1), date(2024, 12, 31),
        date(2025, 6, 30), date(2025, 12, 31), "GROWW",
    )

    assert result["indices"][0]["index"] == "SENSEX"
    assert result["indices"][0]["spot_candles"] == 0
    assert "Fewer than 100" in result["indices"][0]["issue"]


def test_training_data_check_skips_expensive_scan_without_option_candles(session, monkeypatch):
    start = datetime(2024, 1, 1, 3, 45, tzinfo=UTC)
    session.add_all([
        Candle(
            provider="GROWW", instrument_key=f"nifty-{role.lower()}",
            underlying_code="NIFTY", timeframe="5minute",
            timestamp=start + timedelta(minutes=5 * bar),
            open=Decimal("100"), high=Decimal("101"),
            low=Decimal("99"), close=Decimal("100"),
            source_verified=True, data_role=role,
        )
        for role in ("SPOT", "FUTURE") for bar in range(100)
    ])
    session.commit()
    monkeypatch.setattr(
        "app.setup_quality._load_setup_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("expensive scan ran")),
    )

    result = inspect_setup_training_data(
        str(session.get_bind().url), ["NIFTY"],
        date(2023, 1, 1), date(2024, 12, 31),
        date(2025, 6, 30), date(2025, 12, 31), "GROWW",
    )

    summary = result["indices"][0]
    assert summary["spot_candles"] == 100
    assert summary["futures_candles"] == 100
    assert "No verified five-minute option candles" in summary["issue"]
    assert [period["candidates"] for period in summary["periods"]] == [None, None, None]
    assert all(period["labelled"] == 0 and not period["ready"] for period in summary["periods"])

    session.add(Candle(
        provider="GROWW", instrument_key="nifty-unverified-option",
        underlying_code="NIFTY", timeframe="5minute", timestamp=start,
        open=Decimal("100"), high=Decimal("101"),
        low=Decimal("99"), close=Decimal("100"),
        source_verified=True, data_role="OPTION",
    ))
    session.commit()
    result = inspect_setup_training_data(
        str(session.get_bind().url), ["NIFTY"],
        date(2023, 1, 1), date(2024, 12, 31),
        date(2025, 6, 30), date(2025, 12, 31), "GROWW",
    )
    assert "No verified historical option lot-size metadata" in result["indices"][0]["issue"]


def test_option_label_check_identifies_missing_contract_bars(session):
    session.add(Instrument(
        provider="GROWW", instrument_key="sensex-ce", underlying_code="SENSEX",
        exchange="BSE", trading_symbol="SENSEX TEST CE", option_type="CE",
        strike=Decimal("80000"), expiry=date(2025, 7, 31), lot_size=20,
        tick_size=Decimal("0.05"), valid_from=date(2025, 1, 1),
        instrument_kind="OPTION", metadata_verified=True,
    ))
    session.commit()
    dataset = pd.DataFrame([{
        "timestamp": pd.Timestamp("2025-07-01T04:00:00Z"),
        "underlying_code": "SENSEX", "direction": "CE",
        "signal_close": 80010.0, "signal_atr": 50.0,
    }])

    labelled, coverage = option_price_labels(
        session, dataset, provider="GROWW", cost_model="GROWW"
    )

    assert labelled.empty
    assert coverage["rejections"] == {"OPTION_ENTRY_BAR_MISSING": 1}
    assert coverage["missing_examples"][0]["expiry"] == "2025-07-31"
    assert coverage["missing_examples"][0]["atm"] == "80000.00"


def test_option_labels_reject_sparse_chain_instead_of_calling_it_atm(session):
    session.add(Instrument(
        provider="GROWW", instrument_key="sensex-distant-ce",
        underlying_code="SENSEX", exchange="BSE", trading_symbol="SENSEX 91100 CE",
        option_type="CE", strike=Decimal("91100"), expiry=date(2026, 2, 26),
        lot_size=20, tick_size=Decimal("0.05"), valid_from=date(2026, 2, 20),
        instrument_kind="OPTION", metadata_verified=True,
    ))
    session.commit()
    dataset = pd.DataFrame([{
        "timestamp": pd.Timestamp("2026-02-20T06:45:00Z"),
        "underlying_code": "SENSEX", "direction": "CE",
        "signal_close": 83000.0, "signal_atr": 100.0,
    }])

    labelled, coverage = option_price_labels(
        session, dataset, provider="GROWW", cost_model="GROWW"
    )

    assert labelled.empty
    assert coverage["rejections"] == {"NO_NEAR_ATM_CONTRACT": 1}
