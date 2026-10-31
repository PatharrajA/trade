# NiftyPilot

NiftyPilot is a personal Indian index-options bot for NIFTY 50, BANK NIFTY and SENSEX. It supports Upstox and Groww while deliberately keeping the application small: one FastAPI process, one SQLite database, server-rendered pages, no login, no Redis and no Docker.

## Features

- `STOPPED`, `PAPER` and `LIVE` modes stored in SQLite and changed from the dashboard without a restart.
- One selectable execution broker: Upstox or Groww. Broker changes require Stopped mode, no open position and no pending order.
- One active broker across live data, Paper, Live, research download, model training and backtests. Switching broker changes the whole workflow without deleting stored results.
- One common `models/regime.json` model. It can be trained from stored Groww data and later used with the compatible Upstox live feed; broker-specific model folders are not created.
- Completed five-minute candle strategy with spot price structure plus futures VWAP, volume and basis, EMA 9/20/50, ATR, RSI and opening-range features.
- Fail-closed behavior when the model, historical contracts or verified option premiums are missing.
- Dynamic option contracts, non-expiry-day ATM selection with one-strike ITM fallback, and no OTM buying.
- Shared position sizing and risk gates. Initial Live mode is capped at one lot and the portfolio at one open position.
- Paper fills at ask plus slippage; paper exits and open P&L use bid minus slippage.
- Persistent provider-tagged signals, orders and trades with position reconciliation before Live mode.
- Backtests replay verified real option candles and apply selectable Upstox Basic, Upstox Plus or Groww costs.
- Telegram group and optional private-owner alerts through a persistent retry outbox.
- Overview, Orders, Backtests, Telegram and Settings screens.
- Data-readiness matrix with separate spot/futures intraday gaps, first/last missing timestamps, alignment, zero-volume percentage and unmatched historical contract metadata.
- Background, resumable research downloads with persisted progress and a targeted repair mode for known intraday gaps.
- Candidate-model registry with confusion matrix, per-class precision/recall, feature importance and explicit activation.
- Historical contract-metadata CSV import for records that cannot be verified from today's broker instrument master.
- Paired model/rule-only backtests, per-index decision funnels with exact rejection counts, ATM-only versus ATM→ITM1 policies, liquidity/spread rejection, entry delay, adaptive slippage and one-minute stop/target resolution when available.
- Full Paper/Live decision audit including rule checks, bid/ask, slippage, delta, IV, risk budget, stop and target.
- Manual pause and position exit controls, one-minute Live reconciliation, immediate risk shutdown alerts and post-restart Live reconfirmation.

The model predicts regime, direction and confidence. It does not invent stop and target prices: the stop is derived from the underlying ATR, the target uses the configured reward/risk ratio, and a separate premium emergency stop protects against option-price shocks.

## Local setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On macOS, install OpenMP for XGBoost first:

```bash
brew install libomp
```

On Windows, create the environment from PowerShell and install the current Microsoft Visual C++ Redistributable if XGBoost reports a missing runtime DLL:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On Ubuntu, the normal setup above is sufficient. If the XGBoost wheel reports a missing OpenMP library, install the distribution's `libgomp1` package.

Open `http://127.0.0.1:8000`. SQLite tables and default settings are created on first start.

## Broker setup

Open **Settings**, select one broker and configure only the credentials shown for it. Inactive-broker credentials are hidden to avoid using the wrong feed accidentally.

### Upstox

- Analytics Token: read-only market data, contracts, Paper mode and historical downloads.
- Trading Token: order placement, fills, positions and Live reconciliation.

The credentials are intentionally separate. A missing Trading Token disables Live mode but does not interrupt Paper or Backtest. The application validates a replacement token before saving it.

Expired Upstox option contracts and their candles require Upstox Plus. Index candles alone are not enough for this model because cash-index volume is zero. Training now requires aligned, verified index-futures candles for liquidity features. Trustworthy historical CE/PE P&L additionally requires exact expired-option candles and verified historical lot/tick metadata.

### Groww

- Groww Access Token: live data, historical data, Paper mode, orders and reconciliation.
- An active Groww Trading API subscription is required while downloading data or using Groww for Paper/Live. Previously downloaded data does not require the token.
- Groww documents expired F&O candles from 2020. Backtests still require an exact historical contract with its historical lot size and tick size; if those details cannot be verified from the instrument master, NiftyPilot rejects the period instead of using today's values.

Tokens are stored in `data/secrets/` with `0600` permissions. Do not commit or share them.

## Research and backtesting

Use the Backtests page in this order. Download, training and replay automatically use the broker selected in Settings:

1. Download verified five-minute spot, futures and expired-option history for NIFTY, BANK NIFTY and SENSEX.
2. Train the recommended separate TAKE/SKIP setup-quality models. Candidate bundles are kept under `models/candidates/`; training does not silently enable execution.
3. Use **Check training data** with the same indices and Fit/Tune/Holdout dates before training. It counts rule candidates and verified option-price labels for each index and period, shows the first missing contract or option-bar examples, and makes no model or trading changes. Confirm that every period reaches at least 90% option-label coverage and has enough candidates.
4. Review the chronological fit, tuning, untouched holdout and walk-forward results.
5. Run a post-holdout option backtest with **Rule-only comparison** enabled.
6. Review P&L, profit factor, expectancy, drawdown, monthly stability, rejection stages and doubled-slippage stress before enabling shadow mode.

Historical data is always provider-tagged. A Groww backtest cannot accidentally consume Upstox candles and vice versa. Legacy direction-model files and routes remain readable for old records, but they are not shown in the recommended workflow and are not used as a Paper/Live execution fallback.

The engine never synthesizes option premiums from index prices. When historical bid/ask is unavailable, it uses verified option OHLC with configured two-sided slippage and reports the limitation. A historical Groww symbol may be downloaded even when it has disappeared from today's instrument master, but it is marked unverified and excluded from P&L until its historical lot size is known.

Backtests fail with an explicit data reason when the selected period has no option candles linked to verified historical contract metadata. A completed zero-trade run reports the number of evaluated bars, confident model predictions and confirmed signals. Do not lower the confidence gate based only on the number of signals; compare directional precision and after-cost option P&L on unseen dates first.

### Recommended TAKE/SKIP setup-quality model

The Backtests page exposes only the recommended setup-quality trainer. Activation remains an explicit, gated action:

1. The existing futures-VWAP, EMA9/EMA20 and opening-range rules determine CE or PE.
2. Only a newly confirmed rule transition becomes a candidate; consecutive candles in the same active breakout are not duplicated.
3. Each candidate enters on the next five-minute bar and receives a conservative triple-barrier label: 1 ATR stop, 1.5 ATR target and nine-bar timeout. A candle touching stop and target is excluded because the order is unknown without one-minute data.
4. One training job saves a single bundle containing separate NIFTY, BANK NIFTY and SENSEX binary XGBoost components. SENSEX uses NIFTY as its relative-strength reference. The default form uses 2023-2024 for fitting, the first half of 2025 for tuning/calibration and the second half of 2025 as a locked holdout.
5. Features include multi-horizon momentum, EMA slopes, ATR, VWAP/opening-range/day-range distances, futures basis changes, opening gap, NIFTY-versus-BANK-NIFTY relative strength, previous-day position, futures-expiry distance, optional India VIX with an availability flag, cyclical time and time buckets.
6. Holdout reporting compares model-selected TAKE candidates, model SKIP candidates and the rules-only baseline using coverage, TAKE precision, profit factor, expectancy, drawdown, monthly concentration, CE/PE results and confidence bands. Three expanding walk-forward folds are also recorded.

When at least 90% of rule candidates have verified option candles and dated contract metadata, training uses the historical ATM→ITM1 option entry/exit, bid/ask when available, two-sided slippage and the normal Indian charge model. Otherwise the label source is explicitly recorded as `UNDERLYING_TRIPLE_BARRIER_PROXY` and activation remains blocked.

A saved TAKE/SKIP bundle can be selected for a post-holdout option replay from the normal Backtest form. It can also be enabled in **shadow** mode: the live rule/model/contract path records both TAKE and SKIP hypothetical outcomes in `shadow_observations`, but never creates an `Order` or `Trade`. Paper/Live setup execution requires a completed post-holdout option replay paired with rule-only comparison, positive after-cost results for every included index, model profit factor of at least 1.20, positive expectancy, and positive net P&L under double slippage. Shadow validation is assessed separately for every included index: at least 100 closed TAKE outcomes across 20 trading sessions, at least 55% TAKE win rate, profit factor of at least 1.20, positive expectancy, and TAKE profit factor above SKIP when both are available. The shadow dashboard reports all observations for the selected model and broker, with per-index and confidence/time/contract breakdowns. Until these gates pass, Paper/Live model execution remains unavailable.

For pending Groww metadata, export the CSV from **Backtests → Historical contract metadata**, fill lot size and tick size from a trustworthy broker/exchange archive, and import it. The importer only updates already-downloaded matching contracts and records `CSV_IMPORT` as the source.

Research downloads run in the application background. The run and its progress survive page navigation. If the process stops, the run is marked interrupted; submit the same provider, indices and dates again to resume because candle uniqueness prevents duplicate storage. The provider may still be queried for an existing date chunk, but existing rows are not inserted twice.

Repair mode resolves gap dates from the combined spot or futures series, then re-requests those complete trading days for every applicable stored futures contract. A repair run is marked `PARTIAL`, rather than `COMPLETE`, when the provider still omits one or more expected candles. This distinguishes a genuine upstream archive gap from a successful repair.

### Switching after a one-month Groww subscription

1. While the Groww subscription is active, save the Groww token and download the desired spot, futures and expired-option history.
2. Keep Groww selected, train the common model and finish the Groww backtests.
3. Keep the SQLite database and `models/regime*` files in your backups.
4. Stop execution and select Upstox. Upstox then becomes the source for new downloads, training and backtests as well as Paper/Live.
5. The common model trained earlier remains available because it records a normalized, provider-neutral feature schema. Historical Groww candles and old results stay in SQLite, but the simplified UI does not mix them into an Upstox run.

Switching broker does not remove candles, model files or previous backtest results. A future data refresh from Groww requires selecting Groww again and supplying a valid subscription/token.

## Paper and live rollout

1. Keep execution Stopped while configuring the broker, Telegram and risk values.
2. Refresh the selected broker's instrument master.
3. Train the model and complete backtests.
4. Run Paper mode for several sessions and verify signals, option selection, fills, exits, P&L and Telegram messages.
5. Complete the configured Paper-session target (20 distinct trading days by default), verify the static-IP acknowledgement, reconcile broker positions, type `LIVE TRADING`, and begin with the enforced one-lot cap.

Only one broker can execute at a time. Existing trades retain their provider, so exits and reconciliation always route back to the broker that opened them.

Live entries are paused after every application restart and require `LIVE TRADING` again. Broker/database positions are reconciled every minute. Use **Pause new entries** to stop new signals while preserving position monitoring, and use the explicit manual-exit control for an open Paper or Live position. A daily-loss or consecutive-loss breach immediately pauses entries and creates an owner alert.

The Paper-session Live gate is enabled by default at 20 distinct dates containing a closed Paper trade. It can be disabled or changed from **Settings → Live safety gate** without restarting. Disabling it skips only this one gate; model, token, feed, static-IP, open-position and reconciliation checks still apply. Every change is written to system events and queued for the owner-private Telegram destination.

## Telegram

1. Open `@BotFather`, run `/newbot`, choose a name and username, and copy the token.
2. Create a private Telegram group with your friend and add the bot. Send one message such as `/start@your_bot_username` in the group.
3. Open the bot privately, press **Start**, and send one message if you want owner-only failure alerts.
4. Find IDs with `curl "https://api.telegram.org/bot<TOKEN>/getUpdates"`. The group `chat.id` is normally a negative number such as `-100...`; the private `chat.id` is the owner ID.
5. Save `TELEGRAM_BOT_TOKEN=<TOKEN>` in `/opt/niftypilot/.env`, run `chmod 600 /opt/niftypilot/.env`, and restart the service.
6. Open **Telegram** in NiftyPilot, enter the group ID and optional owner ID, enable the destinations, save, and use both test buttons.

The token stays in the server environment; destination IDs and queued delivery state stay in SQLite. The outbox retries temporary Telegram failures and avoids duplicate sends. Treat the token as a password and revoke it in BotFather if it is ever exposed.

## Free deployment and static IP

A computer cannot create its own public static IP. For a home Windows, macOS or Ubuntu machine, either buy a static public IPv4 address from the ISP or run NiftyPilot on a cloud VM with a reserved public IP. Dynamic DNS changes a hostname only; it does not create the fixed outbound IP required for broker orders.

Oracle Cloud Always Free is the practical starting option for Paper mode:

1. Create an Always Free Ubuntu VM in the account home region. An Ampere A1 Flex VM with 1 OCPU and 6 GB RAM is a reasonable small configuration for FastAPI and XGBoost.
2. In OCI, open **Networking → IP management → Reserved public IPs**, reserve an IP, and assign it to the VM's private IP.
3. Allow inbound SSH only. Do not expose port 8000.
4. Copy the project to `/opt/niftypilot`, create the `niftypilot` system user and Python environment, then install `requirements.txt` and Ubuntu's `libgomp1` package.
5. Copy `.env.example` to `/opt/niftypilot/.env`, keep `HOST=127.0.0.1`, set the Telegram token, and run `chmod 600 /opt/niftypilot/.env`.
6. Install and start the included service and backup timer using the commands below.
7. Register the VM's reserved public IP with the active broker. Groww uses the API-key page; Upstox uses **My Apps → Static IPs**. Re-check the broker documentation before enabling Live because IP and token rules can change.

Free compute capacity is not guaranteed and Oracle may reclaim idle Always Free instances. Use it first for backtesting and Paper. For real-money Live trading, monitor availability and move to a paid, reliable VM if interruptions occur.

## Server access

Bind only to `127.0.0.1` and access the dashboard through SSH:

```bash
ssh -L 8000:127.0.0.1:8000 ubuntu@SERVER_IP
```

The included `trading-bot.service` assumes `/opt/niftypilot` and a dedicated `niftypilot` system user. The backup script creates a SQLite-consistent copy, runs `PRAGMA quick_check`, copies the active model and its metadata, and keeps seven days of verified backups.

Install the optional nightly systemd timer:

```bash
sudo cp trading-bot.service niftypilot-backup.service niftypilot-backup.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-bot.service niftypilot-backup.timer
systemctl list-timers niftypilot-backup.timer
```

The Settings screen reports backup freshness and verifies that stored broker token files are not readable by other users. `/health` reports database integrity, active-model status, feed age, backup state, secret-file permissions and the current execution state.

## Verification

```bash
python -m pytest -q
python -m compileall -q app
```

Automated tests never call real order endpoints. Validate both broker integrations in Paper mode, and use a broker sandbox or the smallest allowed quantity before relying on Live execution.
