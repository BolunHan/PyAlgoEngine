# PyAlgoEngine

**High-Performance Algorithmic Trading Engine in Python, Cython, and C**

PyAlgoEngine is a low-latency trading engine designed for HFT (High-Frequency Trading)
systems. It provides C-level market data structures, shared memory buffers,
an event-driven engine architecture, backtesting with simulated order matching,
exchange calendar profiles, and web-based visualization — all with Cython
acceleration for latency-critical paths.

## Features

| Category | Capabilities |
|----------|-------------|
| **Market Data** | Tick, Bar, OrderBook, Transaction, Order, Trade — all C-backed via Cython |
| **Buffers** | Shared-memory `MarketDataBuffer`, lock-free `ConcurrentBuffer`, ring buffer |
| **Engines** | Event engine (pub/sub), Market Data Service (MDS), Algo engine, Trade engine |
| **Exchange Profiles** | CN (A-share) and default global calendars, session phases, holidays |
| **Backtesting** | Data replay (simple/progress), simulated order matching, trade metrics |
| **Strategy Framework** | `StrategyEngineTemplate`, `AlgoTemplate`, global singletons |
| **Web Apps** | Flask + Bokeh dashboards, candlestick charts, strategy tester UI |
| **Compile-Time Config** | `TICKER_SIZE`, `BOOK_SIZE`, `ID_SIZE`, `MAX_WORKERS` via env vars |

## Quick Install

```bash
# Clone and build
git clone https://github.com/BolunHan/PyAlgoEngine.git
cd PyAlgoEngine
./build.sh -i
```

Requires **Python 3.12+** and a C compiler (GCC, Clang, or MSVC).

See the [Setup Guide](https://pyalgoengine.readthedocs.io/en/latest/setup.html) for
detailed instructions, compile-time configuration, and optional dependencies.

## Quick Verify

```python
import algo_engine
print(algo_engine.__version__)  # e.g. "0.12.0"

from algo_engine.base import C_CONFIG
print(C_CONFIG)  # compile-time constants
```

## Documentation

Full documentation at **[pyalgoengine.readthedocs.io](https://pyalgoengine.readthedocs.io/)**.

Build locally:
```bash
cd docs && ./update_docs.sh
# open build/html/index.html
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  apps/          Web visualization (Flask, Bokeh)     │
├──────────────────────────────────────────────────────┤
│  strategy/      Strategy framework                   │
├──────────────────────────────────────────────────────┤
│  engine/        Event, Market, Algo, Trade engines   │
├──────────────────────────────────────────────────────┤
│  backtest/      Replay, SimMatch, Metrics            │
├──────────────────────────────────────────────────────┤
│  base/          Market data types, buffers, allocators│
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
| `make list-args` | List compile-time macros |
| `./build.sh -l` | List compile-time macros |

Override compile-time constants:
```bash
BOOK_SIZE=20 MAX_WORKERS=256 ./build.sh -i
```

## Dependencies

- **Runtime:** numpy, pandas, exchange_calendars, PyCyBase, PyEventEngine, Cython
- **Web (optional):** flask, waitress, bokeh
- **Docs (optional):** sphinx, sphinx-rtd-theme

## License

MIT — [Han Bolun](https://github.com/BolunHan)
