# ChainScope

ChainScope helps you research cryptocurrencies from one workspace. Choose up to four assets, select
the topics you want to investigate, and run a structured report using current market evidence.

> ChainScope is an educational prototype, not financial advice or a production trading system.

## What you can do

- Research market activity, risk, derivatives, fundamentals, DeFi, on-chain activity, news, and
  price forecasts.
- Analyze current provider evidence instead of relying on prewritten market summaries.
- Compare up to four crypto assets in one report.
- Discover market-wide opportunities using Kraken, Coinbase, or Binance spot data.
- Follow research progress and retry an individual section when its data source fails.
- Review market charts and calculated metrics in the dashboard.
- Save reports in your authenticated workspace and export them as JSON or PDF.
- Run research from the command line and receive structured JSON.

## Data sources

ChainScope keeps each provider's identity and observation time attached to its evidence.

| Research area | Sources |
| --- | --- |
| Spot market data | Kraken, Coinbase, and Binance |
| Fundamentals | CoinGecko |
| DeFi | DefiLlama |
| On-chain activity | Coin Metrics Community |
| Derivatives | Binance USD-M Futures |
| News | Cointelegraph and Google News |

## Live analysis

Each research run collects current evidence for the assets and topics you selected. ChainScope
normalizes records from different providers, removes duplicate news, checks freshness, and keeps
the source and observation time visible in the final report.

Market metrics, risk scores, comparisons, and forecasts are calculated from the collected data.
Each research topic runs in its own section, so an unavailable provider does not remove results
that were completed successfully by the other sections. Failed sections remain visible and can be
retried individually.

### Groq interpretation

Groq can be added as an interpretation layer for the live evidence and calculated results. It
creates written specialist analysis for market, news, fundamentals, on-chain, and forecast
sections while the provider records and numeric calculations remain the source of truth.

Add these settings to `.env` for the CLI or `.streamlit/secrets.toml` for the web interface:

```text
LLM_PROVIDER=groq
GROQ_API_KEY=your-key
GROQ_MODEL=openai/gpt-oss-120b
```

Groq interpretation is isolated by research section. If one interpretation request fails, the
collected live evidence and the other completed sections remain available.

## Requirements

- Python 3.14
- Internet access for live provider data
- An Auth0 Regular Web Application for the Streamlit interface

The CLI does not require Auth0.

## Quick start: CLI

The examples below use PowerShell on Windows.

```powershell
git clone https://github.com/youssef13-del/crypto-research-agent.git
Set-Location crypto-research-agent
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run research from the command line:

```powershell
crypto-research analyze --assets BTC ETH --topics market risk news
crypto-research discover --exchange kraken --timeframe 4h
```

The CLI writes validated, redacted JSON to standard output. Use the global `--output` option to
also save it to a file:

```powershell
crypto-research --output bitcoin-report.json analyze --assets BTC --topics fundamentals defi onchain
```

To see every option:

```powershell
crypto-research --help
crypto-research analyze --help
crypto-research discover --help
```

## Run the web interface

### 1. Create the local secrets file

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Never commit `.env` or `.streamlit/secrets.toml`; both may contain credentials.

### 2. Configure Auth0

Create an **Auth0 Regular Web Application**, then configure an Auth0 database connection in
Universal Login.

Add this exact URL to both **Allowed Callback URLs** and **Allowed Logout URLs**:

```text
http://localhost:8501/oauth2callback
```

Fill in these values in `.streamlit/secrets.toml`:

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "replace-with-a-long-random-secret"

[auth.auth0]
client_id = "replace-with-auth0-client-id"
client_secret = "replace-with-auth0-client-secret"
server_metadata_url = "https://YOUR_AUTH0_DOMAIN/.well-known/openid-configuration"
```

The same Auth0 Universal Login configuration handles both sign-in and sign-up.

### 3. Start ChainScope

```powershell
chainscope
```

You can also use:

```powershell
python -m crypto_research.interfaces.web
```

Open `http://localhost:8501`. After authentication, the app provides Home, Research, Research
Library, Market Dashboard, and Account pages. Guided Research is available at `/research`.

### App pages

- **Home** gives you a quick market overview and discovery shortcuts.
- **Research** lets you choose assets, topics, exchange, timeframe, and forecast settings.
- **Market Dashboard** displays price history, comparisons, and calculated market metrics.
- **Research Library** stores completed reports for viewing, comparison, JSON export, and PDF
  export.
- **Account** shows your workspace details and account controls.

## Forecast options

Forecasts use deterministic models with walk-forward validation and quality gates.

```powershell
crypto-research analyze `
  --assets BTC `
  --topics forecast `
  --forecast-model ridge `
  --forecast-horizon 24 `
  --forecast-confidence 0.9 `
  --forecast-lookback 900
```

- Models: `gradient_boosting_huber` or `ridge`
- Horizons: `4`, `8`, `12`, `24`, or `48` hours
- Confidence: `0.8` or `0.9`
- Lookback: `400` to `2000` candles

## Storage and configuration

All supported settings are explained in `.env.example` and
`.streamlit/secrets.toml.example`.

SQLite is the only database backend. If `DATABASE_URL` is empty, ChainScope creates its database
in the operating system's per-user application-data directory. Saved reports remain scoped to the
authenticated user in the web interface.

Do not commit credentials, local databases, generated reports, `.env`, or
`.streamlit/secrets.toml`.

## How a research run works

1. Enter one to four assets and select the research topics you need.
2. ChainScope collects evidence for each selected topic and calculates the relevant metrics.
3. Results appear in separate report sections, so one unavailable source does not hide the other
   completed work.
4. The completed report is saved to your library, where you can compare, pin, export, or delete it.

Every result keeps its data source, observation time, freshness, and evidence identifier so you can
see where the information came from.

## Troubleshooting

- If `chainscope` is not recognized, activate `.venv` and run `python -m pip install -e ".[dev]"`.
- If sign-in loops back to the login page, verify the Auth0 callback URL and the values in
  `.streamlit/secrets.toml`.
- If one report section is unavailable, check its source warning in the report and retry only that
  section. Other completed sections remain usable.
- If the database cannot be opened, verify that `DATABASE_URL` points to a writable SQLite path.

## Development checks

Run the canonical local verification before opening a pull request:

```powershell
python scripts/check.py
python -m build
git diff --check
```

The check script runs Ruff lint and formatting verification, strict mypy, the test suite,
`pip check`, and import and entry-point smoke checks.

## Project layout

The code is arranged from reusable foundations to user-facing interfaces:

```text
shared → domain → tools / forecasting / llm → agents → orchestration → interfaces
```

- `shared/`: security, formatting, JSON, time, and numeric-grounding helpers.
- `domain/`: validated research, evidence, market, forecast, account, and history contracts.
- `tools/`: external data collection and deterministic calculations.
- `agents/`: one folder per research agent. Each folder owns its public agent, collector,
  analyzer, and prompt.
- `orchestration/`: research planning, evidence ledgers, coverage, events, and runtime.
- `interfaces/`: the CLI and Streamlit web application.

`bootstrap.py` is the only composition root: it wires providers, storage, LLM adapters, and
agents together.

## Technology

Python, Streamlit, Pydantic, Groq, Auth0, SQLite, SQLAlchemy, pandas, NumPy, scikit-learn, CCXT,
pytest, Ruff, and mypy.
