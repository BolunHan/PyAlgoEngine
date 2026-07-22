# PyAlgoEngine

**High-Performance Algorithmic Trading Engine in Python, Cython, and C**

PyAlgoEngine is a low-latency trading engine for HFT (High-Frequency Trading)
systems. It provides C-level market data structures, shared-memory buffers,
an event-driven engine architecture, backtesting with simulated order matching,
exchange calendar profiles, and web-based visualization — all with Cython
acceleration for latency-critical paths.

## Features

| Category | Capabilities |
|----------|-------------|
| **Market Data** | `TickData`, `TickDataLite`, `BarData`, `DailyBar`, `OrderBook`, `TransactionData`, `OrderData`, `TradeData` — all C-backed via Cython |
| **Buffers** | `MarketDataBuffer`, `MarketDataRingBuffer`, `MarketDataConcurrentBuffer`, `MarketDataBufferCache` |
| **Engines** | `EVENT_ENGINE` (pub/sub), `MDS` (Market Data Service), `ALGO_ENGINE` (algo lifecycle), trade engine (DMA, positions, risk) |
| **Exchange Profiles** | CN (A-share) and default global calendars, session phases, holidays via `PROFILE`, `PROFILE_CN`, `PROFILE_DEFAULT` |
| **Backtesting** | `SimpleReplay`, `ProgressReplay`, `SimMatch` (simulated matching), `TradeMetrics` |
| **Strategy Framework** | `StrategyEngine`, `AlgoTemplate`, global singletons (`BALANCE`, `DMA`, `STRATEGY_ENGINE`) |
| **Web Apps** | Flask + Bokeh dashboards, candlestick charts, strategy tester UI |
| **Compile-Time Config** | `BOOK_SIZE`, `ID_SIZE`, `LONG_ID_SIZE`, `MD_BUF_PTR_DEFAULT_CAP`, `MD_BUF_DATA_DEFAULT_CAP` via env vars |

## Quick Install

```bash
git clone https://github.com/BolunHan/PyAlgoEngine.git
cd PyAlgoEngine
./build.sh -i
```

Requires **Python 3.12+** and a C compiler (GCC, Clang, or MSVC).
See the [Setup Guide](docs/source/setup.rst) for
detailed instructions, compile-time configuration, and optional dependencies.

## Quick Verify

```python
import algo_engine
print(algo_engine.__version__)

from algo_engine.base import CONFIG
print(CONFIG)  # compile-time and runtime configuration
```

## Documentation

Build locally:
```bash
cd docs && ./update_docs.sh
# open build/html/index.html
```

### Deploy to Read the Docs

1. Sign up at [readthedocs.org](https://readthedocs.org) and import this repo
2. The included [`.readthedocs.yaml`](.readthedocs.yaml) handles the build —
   it compiles Cython extensions, installs the package, and builds with Sphinx + Furo
3. RTD auto-builds on every push to `main`; enable the GitHub webhook in
   **Admin → Integrations** on your RTD project dashboard

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  apps/          Web visualization (Flask, Bokeh)     │
├──────────────────────────────────────────────────────┤
│  strategy/      Strategy framework, global singletons│
├──────────────────────────────────────────────────────┤
│  engine/        Event, Market, Algo, Trade engines   │
├──────────────────────────────────────────────────────┤
│  backtest/      Replay, SimMatch, Metrics            │
├──────────────────────────────────────────────────────┤
│  base/          Market data types, buffers (Cython/C)│
├──────────────────────────────────────────────────────┤
│  exchange_profile/  Calendars, sessions (CN + global)│
├──────────────────────────────────────────────────────┤
│  monitor/       Synthetic order book, bar monitor    │
├──────────────────────────────────────────────────────┤
│  utils/         Time-series indices, fake data       │
└──────────────────────────────────────────────────────┘
```

## Build System

| Command | Description |
|---------|-------------|
| `./build.sh -i` | Clean, build Cython extensions in-place, install |
| `make build` | Clean and build extensions in-place |
| `make install` | Build + pip install |
| `./build.sh -l` | List all compile-time macros and defaults |
| `make list-args` | Same as above |

Override compile-time constants:
```bash
BOOK_SIZE=20 ./build.sh -i
```

## Dependencies

- **Runtime:** numpy, pandas, exchange_calendars, PyCyBase, PyEventEngine, Cython
- **Web (optional):** flask, waitress, bokeh
- **Docs (optional):** sphinx, sphinx-rtd-theme

## License

MIT — [Han Bolun](https://github.com/BolunHan)
