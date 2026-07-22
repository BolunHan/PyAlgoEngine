# Documentation Overhaul — Implementation Plan

> **For agentic workers:** Use inline execution. Tasks are documentation files — verify by building with Sphinx.

**Goal:** Rewrite all PyAlgoEngine docs from sparse/obsolete to comprehensive, accurate, well-structured documentation.

**Architecture:** Curated RST prose guides + `autodoc` API reference, built with Sphinx + sphinx_rtd_theme, verified via `docs/update_docs.sh`.

**Tech Stack:** Sphinx, sphinx-rtd-theme, RST, Python autodoc

## Global Constraints

- Python >= 3.12
- Sphinx build must produce zero warnings
- All pages must render correctly in sphinx_rtd_theme
- README must link to built docs

---

### Task 1: Update conf.py

**Files:**
- Modify: `docs/source/conf.py`

- [ ] **Step 1: Update conf.py with better config**

Replace `docs/source/conf.py` with:

```python
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

from algo_engine import __version__

project = 'PyAlgoEngine'
author = 'Han Bolun'
release = __version__
version = '.'.join(__version__.split('.')[:2])  # e.g. "0.12"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

autosummary_generate = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}
html_static_path = ['_static']
html_context = {
    'display_github': True,
    'github_user': 'BolunHan',
    'github_repo': 'PyAlgoEngine',
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
}

# Suppress Cython .so import warnings during autodoc
suppress_warnings = ['import']


def setup(app):
    from sphinx.builders.html import StandaloneHTMLBuilder
    StandaloneHTMLBuilder.supported_image_types = [
        'image/svg+xml',
        'image/png',
        'image/jpeg',
    ]
```

- [ ] **Step 2: Verify conf.py loads without error**

```bash
cd docs && python -c "import sys; sys.path.insert(0, 'source'); from conf import *; print(f'Config OK: {project} v{release}')"
```

---

### Task 2: Update update_docs.sh

**Files:**
- Modify: `docs/update_docs.sh`

- [ ] **Step 1: Replace update_docs.sh**

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[docs] Generating API stub pages..."
sphinx-apidoc -o source/api/ ../algo_engine/ -f -e -M --tocfile index 2>/dev/null || true

echo "[docs] Cleaning previous build..."
make clean

echo "[docs] Building HTML (warnings as errors)..."
make html SPHINXOPTS="-W --keep-going"

echo "[docs] Done — open build/html/index.html"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x docs/update_docs.sh
```

---

### Task 3: Rewrite README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README.md with accurate project description**

```markdown
# PyAlgoEngine

**High-Performance Algorithmic Trading Engine in Python, Cython, and C**

PyAlgoEngine is a low-latency trading engine designed for HFT (High-Frequency Trading)
systems. It provides C-level market data structures, shared memory buffers,
an event-driven engine, backtesting, exchange calendars, and web-based visualization —
all with Cython acceleration for critical paths.

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

Requires **Python 3.12+** and a C compiler. See the
[Setup Guide](https://pyalgoengine.readthedocs.io/en/latest/setup.html) for
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
┌──────────────────────────────────────────────┐
│  apps/          Web visualization (Flask, Bokeh) │
├──────────────────────────────────────────────┤
│  strategy/      Strategy framework              │
├──────────────────────────────────────────────┤
│  engine/        Event, Market, Algo, Trade engines │
├──────────────────────────────────────────────┤
│  backtest/      Replay, SimMatch, Metrics       │
├──────────────────────────────────────────────┤
│  base/          Market data types, buffers, allocators │
├──────────────────────────────────────────────┤
│  exchange_profile/   Calendars, sessions (CN + global) │
├──────────────────────────────────────────────┤
│  monitor/       Synthetic order book, bar monitor   │
├──────────────────────────────────────────────┤
│  utils/         Time-series indices, fake data   │
└──────────────────────────────────────────────┘
```

## Dependencies

- **Runtime:** numpy, pandas, exchange_calendars, PyCyBase, PyEventEngine, Cython
- **Web (optional):** flask, waitress, bokeh
- **Docs (optional):** sphinx, sphinx-rtd-theme

## License

MIT — [Han Bolun](https://github.com/BolunHan)
```

---

### Task 4: Rewrite index.rst

**Files:**
- Modify: `docs/source/index.rst`

- [ ] **Step 1: Replace with accurate landing page**

```rst
PyAlgoEngine — Algorithmic Trading Engine
==========================================

**PyAlgoEngine** is a high-performance algorithmic trading engine written in
Python, Cython, and C, designed for HFT (High-Frequency Trading) systems.

It provides C-level market data structures with shared-memory buffers, an
event-driven engine architecture, backtesting with simulated order matching,
exchange calendar profiles, and web-based visualization dashboards — all
with Cython acceleration for latency-critical paths.

:Homepage: https://github.com/BolunHan/PyAlgoEngine
:License: MIT
:Version: |release|

Features
--------

- **C-Level Market Data** — Tick, Bar, OrderBook, Transaction, Order, Trade
  types implemented in Cython/C with nanosecond timestamps and 128-bit IDs.
- **High-Performance Buffers** — Shared-memory ``MarketDataBuffer``,
  lock-free ``ConcurrentBuffer``, ring buffer, and buffer cache.
- **Event-Driven Engine** — Pub/sub event engine, market data service (MDS),
  algorithm engine, and trade engine with position/risk management.
- **Exchange Profiles** — CN (A-share) and global exchange calendars with
  session phases, holiday schedules, and timezone handling.
- **Backtesting** — Historical data replay, simulated order matching with
  configurable fees/latency, and trade performance metrics.
- **Strategy Framework** — ``StrategyEngineTemplate`` and ``AlgoTemplate``
  base classes with global singletons for rapid strategy development.
- **Web Visualization** — Flask + Bokeh dashboards with candlestick charts
  and interactive strategy tester.
- **Compile-Time Tuning** — Configure ``BOOK_SIZE``, ``TICKER_SIZE``,
  ``ID_SIZE``, ``MAX_WORKERS`` at build time via environment variables.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   setup
   usage
   architecture
   market_data
   engines
   strategy
   backtest
   web_apps

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

---

### Task 5: Rewrite setup.rst

**Files:**
- Modify: `docs/source/setup.rst`

- [ ] **Step 1: Replace with detailed setup guide**

```rst
Setup & Installation
====================

Requirements
------------

* **Python** 3.12 or higher
* **C compiler** (GCC, Clang, or MSVC)
* **Linux** or **Windows** (macOS may work but is not tested)

Dependencies are declared in ``pyproject.toml`` and installed automatically:

* ``numpy``, ``pandas`` — data handling
* ``exchange_calendars`` — trading calendar support
* ``PyCyBase`` — Cython base utilities (allocators, interned strings)
* ``PyEventEngine`` — event-driven framework
* ``Cython`` — extension compilation

Install from Source
-------------------

Clone the repository and build:

.. code-block:: bash

   git clone https://github.com/BolunHan/PyAlGoEngine.git
   cd PyAlgoEngine

Then choose one of three build methods:

Method 1: build.sh (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ./build.sh -i

This cleans previous artifacts, compiles Cython extensions in-place, and
installs the package. Use ``-v <path>`` to specify a virtual environment.

Method 2: Make
~~~~~~~~~~~~~~~

.. code-block:: bash

   make build && pip install -U . --no-build-isolation

``make build`` runs ``python setup.py build_ext --inplace --verbose --force``
after cleaning.

Method 3: Step-by-step
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python setup.py build_ext --inplace --verbose --force
   pip install -U . --no-build-isolation

Compile-Time Configuration
---------------------------

You can override memory layout constants at compile time via environment
variables. This is useful for tuning the engine to specific market data
requirements without modifying source code.

.. list-table:: Compile-Time Macros
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``TICKER_SIZE``
     - 32
     - Max length of a ticker symbol (bytes)
   * - ``BOOK_SIZE``
     - 10
     - Max depth of the order book (levels)
   * - ``ID_SIZE``
     - 16
     - Max length of ID fields (bytes)
   * - ``MAX_WORKERS``
     - 128
     - Max number of concurrent buffer workers
   * - ``DEBUG``
     - 0
     - Enable debug assertions (set to 1)

Override any of these before building:

.. code-block:: bash

   BOOK_SIZE=20 MAX_WORKERS=256 ./build.sh -i

Verify the compiled values:

.. code-block:: python

   from algo_engine.base import C_CONFIG
   print(C_CONFIG)

Optional Dependencies
---------------------

Install extras for additional features:

.. code-block:: bash

   # Web dashboard support
   pip install PyAlgoEngine[WebApps]

   # Documentation build tools
   pip install PyAlgoEngine[Docs]

Platform Notes
--------------

**Linux** (primary target):
    GCC or Clang with ``-O3 -march=native``. Parallel builds use
    ``os.cpu_count() - 2`` threads.

**Windows**:
    MSVC with ``/Ox /std:c17``. Parallel builds are disabled by default
    (set to 1 thread); override with ``N_THREADS`` environment variable if needed.
```

---

### Task 6: Rewrite usage.rst

**Files:**
- Modify: `docs/source/usage.rst`

- [ ] **Step 1: Replace with quick-start examples**

```rst
Quick Start
===========

This guide walks through basic usage: creating market data, using buffers,
and subscribing to market data events.

Creating Market Data
--------------------

PyAlgoEngine provides several market data types, all backed by C-level
structures via Cython:

.. code-block:: python

   from algo_engine.base import TickData, BarData, OrderBook
   from algo_engine.base import TransactionData, OrderData
   from algo_engine.base import TransactionSide, TransactionDirection, TransactionOffset

   # Create a tick
   tick = TickData(
       ticker="AAPL",
       last=150.25,
       bid=150.20,
       ask=150.30,
       bid_volume=1000,
       ask_volume=500,
       volume=50000,
       timestamp=1711000000000000000,  # nanoseconds
   )

   # Create a bar (candlestick)
   bar = BarData(
       ticker="AAPL",
       open=150.00,
       high=151.00,
       low=149.50,
       close=150.75,
       volume=100000,
       timestamp=1711000000000000000,
   )

   # Create an order
   order = OrderData(
       ticker="AAPL",
       price=150.50,
       volume=100,
       side=TransactionSide.BUY,
       direction=TransactionDirection.LONG,
       offset=TransactionOffset.OPEN,
   )

Using Market Data Buffers
--------------------------

Buffers are the core data storage. They support shared memory for
inter-process communication:

.. code-block:: python

   from algo_engine.base import MarketDataBuffer, MarketDataConcurrentBuffer

   # Create a standard buffer
   buffer = MarketDataBuffer()

   # Push data
   buffer.push_tick(tick)
   buffer.push_bar(bar)
   buffer.push_order(order)

   # Read back
   latest_tick = buffer.get_tick("AAPL")
   print(latest_tick.last)  # 150.25

   # For multi-worker scenarios, use the concurrent buffer
   cbuffer = MarketDataConcurrentBuffer(num_workers=4)
   cbuffer.push_tick(tick)

Subscribing to Market Data
--------------------------

The Market Data Service (MDS) provides a pub/sub interface for market data
events:

.. code-block:: python

   from algo_engine.engine import MDS, MarketDataMonitor

   # Subscribe to tick updates for a ticker
   MDS.subscribe_tick("AAPL")

   # Register a monitor for custom handling
   class MyMonitor(MarketDataMonitor):
       def on_tick(self, tick):
           print(f"Tick: {tick.ticker} @ {tick.last}")

   monitor = MyMonitor()
   MDS.add_monitor(monitor)

Using Exchange Profiles
-----------------------

Exchange profiles handle trading calendars, sessions, and timezone logic:

.. code-block:: python

   from algo_engine.exchange_profile import PROFILE, PROFILE_CN

   # Use the global dispatcher (auto-detects market)
   is_session = PROFILE.is_trading_time(timestamp=1711000000000000000)

   # Or use a specific profile
   cn_calendar = PROFILE_CN.get_calendar()
   is_cn_session = PROFILE_CN.is_trading_time(timestamp=1711000000000000000)

Generating Test Data
--------------------

For development and testing, use the built-in fake data utilities:

.. code-block:: python

   from algo_engine.utils import fake_data, fake_daily_data, ts_indices

   # Generate fake tick data for a date range
   ticks = fake_data(ticker="AAPL", start_date="2024-01-02", end_date="2024-01-05")

   # Generate daily bars
   daily_bars = fake_daily_data(ticker="AAPL", start_date="2024-01-02", end_date="2024-03-29")

   # Generate time-series indices aligned to trading sessions
   indices = ts_indices(freq="1min", start_date="2024-01-02", end_date="2024-01-05")

Next Steps
----------

- :doc:`architecture` — understand the system design
- :doc:`market_data` — deep dive into market data types and buffers
- :doc:`strategy` — build your first trading strategy
- :doc:`backtest` — run backtests on historical data
```

---

### Task 7: Create architecture.rst

**Files:**
- Create: `docs/source/architecture.rst`

- [ ] **Step 1: Write architecture overview**

```rst
Architecture
============

PyAlgoEngine is organized into eight layers, each with a clear
responsibility. Data flows from low-level C structures upward through
engines, strategies, and finally to visualization.

Layer Diagram
-------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │  apps/                                                  │
   │  Web dashboards, strategy tester UI, Bokeh servers      │
   │  Depends on: engine, backtest                           │
   ├─────────────────────────────────────────────────────────┤
   │  strategy/                                              │
   │  Strategy engine template, global singletons            │
   │  Depends on: engine                                     │
   ├─────────────────────────────────────────────────────────┤
   │  engine/                                                │
   │  Event engine, MDS, Algo engine, Trade engine           │
   │  Depends on: base, exchange_profile                     │
   ├─────────────────────────────────────────────────────────┤
   │  backtest/                 │  monitor/                  │
   │  Replay, SimMatch, metrics │  Synthetic OB, bar monitor │
   │  Depends on: engine, base  │  Depends on: engine, base  │
   ├─────────────────────────────────────────────────────────┤
   │  base/                                                 │
   │  Market data types, buffers, allocators, intern strings │
   │  (Cython/C accelerated)                                 │
   ├─────────────────────────────────────────────────────────┤
   │  exchange_profile/                                      │
   │  Trading calendars, sessions, holidays (CN + global)    │
   │  (Cython/C accelerated)                                 │
   ├─────────────────────────────────────────────────────────┤
   │  utils/                                                 │
   │  Time-series indices, fake data generators              │
   └─────────────────────────────────────────────────────────┘

Data Flow
---------

.. code-block:: text

   Market Data Feed
        │
        ▼
   ┌─────────────────┐
   │  MarketDataBuffer │  ◄── C-level, shared memory
   │  (base)           │
   └────────┬────────┘
            │ push events
            ▼
   ┌─────────────────┐
   │  Event Engine    │  ◄── Pub/sub dispatch
   │  (engine)        │
   └────────┬────────┘
            │ notify handlers
            ▼
   ┌─────────────────┐
   │  Market Data     │
   │  Service (MDS)   │  ◄── Subscription management
   └────────┬────────┘
            │ tick/bar/order events
            ▼
   ┌─────────────────┐
   │  Strategy Engine │  ◄── User strategy logic
   │  (strategy)      │
   └────────┬────────┘
            │ trade signals
            ▼
   ┌─────────────────┐
   │  Trade Engine    │  ◄── Order routing, positions
   │  (engine)        │
   └────────┬────────┘
            │ execution reports
            ▼
   ┌─────────────────┐
   │  Backtest / Live │
   └─────────────────┘

Singleton Pattern
-----------------

PyAlgoEngine uses a global singleton pattern for core services. This
simplifies strategy code by providing direct access to engines without
dependency injection:

.. code-block:: python

   from algo_engine.engine import EVENT_ENGINE, MDS, ALGO_ENGINE
   from algo_engine.strategy import BALANCE, DMA, STRATEGY_ENGINE

These singletons are module-level instances created at import time. In
backtesting mode, ``algo_engine.backtest.__main__`` creates isolated
singletons for the backtest context.

Cython Acceleration
-------------------

Performance-critical components are implemented in Cython (``.pyx``) and
compiled to native extensions. The Cython layer provides:

* **C-level structs** for market data — no Python object overhead per tick
* **Shared memory allocators** — lock-free and locking protocols for IPC
* **Interned strings** — ticker symbols stored as integer IDs
* **Direct C function calls** — bypassing Python call overhead in hot paths

The build system compiles 15+ Cython extensions covering allocators,
market data types, buffers, exchange profiles, engines, and event handling.
```

---

### Task 8: Create market_data.rst

**Files:**
- Create: `docs/source/market_data.rst`

- [ ] **Step 1: Write market data types and buffers guide**

```rst
Market Data
===========

This section covers the market data types and buffer systems that form the
foundation of PyAlgoEngine. All types are C-level structures exposed to
Python via Cython — they provide Python-friendly APIs with near-C
performance.

Data Types
----------

TickData
~~~~~~~~

Represents a single tick (trade/quote) with full order book snapshot.

.. code-block:: python

   from algo_engine.base import TickData

   tick = TickData(
       ticker="AAPL",
       last=150.25,           # last traded price
       bid=150.20,            # best bid
       ask=150.30,            # best ask
       bid_volume=1000,       # volume at best bid
       ask_volume=500,        # volume at best ask
       volume=50000,          # cumulative volume
       open_interest=0,       # options/futures OI
       timestamp=1711000000000000000,  # nanoseconds since epoch
   )

   # Access order book levels
   tick.order_book.bid_price[0]   # best bid price
   tick.order_book.bid_volume[0]  # best bid volume
   tick.order_book.ask_price[0]   # best ask price
   tick.order_book.ask_volume[0]  # best ask volume

Key attributes:
* ``ticker`` — instrument symbol
* ``last`` — last traded price
* ``bid`` / ``ask`` — best bid/ask
* ``bid_volume`` / ``ask_volume`` — volume at best levels
* ``volume`` — cumulative trading volume
* ``amount`` — cumulative trading amount (price × volume)
* ``order_book`` — full ``OrderBook`` with ``bid_price``, ``bid_volume``,
  ``ask_price``, ``ask_volume`` arrays
* ``timestamp`` — nanosecond Unix timestamp

OrderBook
~~~~~~~~~

A snapshot of the limit order book at a given depth (compile-time
``BOOK_SIZE``, default 10).

.. code-block:: python

   from algo_engine.base import OrderBook

   ob = tick.order_book
   for i in range(ob.book_size):
       if ob.bid_volume[i] > 0:
           print(f"Bid L{i}: {ob.bid_price[i]} x {ob.bid_volume[i]}")
       if ob.ask_volume[i] > 0:
           print(f"Ask L{i}: {ob.ask_price[i]} x {ob.ask_volume[i]}")

BarData / DailyBar
~~~~~~~~~~~~~~~~~~

Candlestick (OHLCV) bars at any resolution.

.. code-block:: python

   from algo_engine.base import BarData, DailyBar

   bar = BarData(
       ticker="AAPL",
       open=150.00, high=151.00, low=149.50, close=150.75,
       volume=100000, amount=15100000.0,
       timestamp=1711000000000000000,
   )

   daily = DailyBar(
       ticker="AAPL",
       open=149.00, high=152.00, low=148.50, close=151.25,
       volume=5000000, amount=750000000.0,
       timestamp=1711000000000000000,
   )

Transaction Data
~~~~~~~~~~~~~~~~

Orders, trades, and transaction records.

.. code-block:: python

   from algo_engine.base import (
       OrderData, TradeData, TransactionData,
       TransactionSide, TransactionDirection, TransactionOffset
   )

   # An order sent to the market
   order = OrderData(
       ticker="AAPL",
       price=150.50,
       volume=100,
       side=TransactionSide.BUY,
       direction=TransactionDirection.LONG,
       offset=TransactionOffset.OPEN,
       order_id="ord_001",
       timestamp=1711000000000000000,
   )

   # A trade (fill) confirmation
   trade = TradeData(
       ticker="AAPL",
       price=150.48,
       volume=100,
       side=TransactionSide.BUY,
       trade_id="trd_001",
       order_id="ord_001",
       timestamp=1711000000000000000,
   )

   # A generic transaction record
   transaction = TransactionData(
       ticker="AAPL",
       price=150.50,
       volume=100,
       side=TransactionSide.BUY,
       direction=TransactionDirection.LONG,
       offset=TransactionOffset.OPEN,
   )

Enums:
* ``TransactionSide`` — ``BUY``, ``SELL``
* ``TransactionDirection`` — ``LONG``, ``SHORT``
* ``TransactionOffset`` — ``OPEN``, ``CLOSE``, ``CLOSE_TODAY``, ``CLOSE_YESTERDAY``
* ``OrderType`` — ``LIMIT``, ``MARKET``, ``FAK``, ``FOK``, ``GFD``

Trade Utilities
~~~~~~~~~~~~~~~

Track order state and generate trade reports:

.. code-block:: python

   from algo_engine.base import OrderState, TradeReport, TradeInstruction

   # Track an order through its lifecycle
   state = OrderState(order)
   state.filled_volume = 50
   print(state.is_filled)  # False

   # Trade report summarizes execution
   report = TradeReport(order, trade)

   # Instruction for the next action
   instruction = TradeInstruction(
       ticker="AAPL", price=151.00, volume=50,
       side=TransactionSide.SELL, offset=TransactionOffset.CLOSE,
   )

MarketData Base Class
~~~~~~~~~~~~~~~~~~~~~

All market data types inherit from ``MarketData``, which provides:

* ``md_id`` — 128-bit unique identifier
* ``ticker`` — instrument symbol (interned string)
* ``timestamp`` — nanosecond Unix timestamp
* ``data_type`` — ``DataType`` enum (TICK, BAR, ORDER, TRADE, etc.)
* ``exchange`` — exchange identifier string

Data Buffers
------------

MarketDataBuffer
~~~~~~~~~~~~~~~~

The primary buffer for storing and accessing market data. Supports shared
memory for inter-process communication.

.. code-block:: python

   from algo_engine.base import MarketDataBuffer

   buffer = MarketDataBuffer()

   # Push data
   buffer.push_tick(tick)
   buffer.push_bar(bar)
   buffer.push_order(order)

   # Query
   latest_tick = buffer.get_tick("AAPL")
   latest_bar = buffer.get_bar("AAPL")
   all_ticks = buffer.get_all_ticks()

   # Thread-safe snapshot
   snapshot = buffer.snapshot()

MarketDataConcurrentBuffer
~~~~~~~~~~~~~~~~~~~~~~~~~~

Lock-free concurrent buffer for multi-worker scenarios. Uses per-worker
shards to avoid contention.

.. code-block:: python

   from algo_engine.base import MarketDataConcurrentBuffer

   cbuffer = MarketDataConcurrentBuffer(num_workers=4)
   cbuffer.push_tick(tick, worker_id=0)

MarketDataRingBuffer
~~~~~~~~~~~~~~~~~~~~

Fixed-size ring (circular) buffer for streaming data.

.. code-block:: python

   from algo_engine.base import MarketDataRingBuffer

   ring = MarketDataRingBuffer(capacity=10000)
   ring.push_tick(tick)

MarketDataBufferCache
~~~~~~~~~~~~~~~~~~~~~

LRU-style cache wrapper around a buffer.

.. code-block:: python

   from algo_engine.base import MarketDataBufferCache

   cache = MarketDataBufferCache(buffer)
   cache.get_tick("AAPL")  # cached access

Allocator Protocols
-------------------

Memory allocation is controlled by configurable protocols defined in
``algo_engine.base``:

* ``MD_SHARED`` — shared memory allocation (inter-process)
* ``MD_LOCKED`` — mutex-protected allocation
* ``MD_LOCKFREE`` — lock-free allocation (atomic operations)
* ``MD_FREELIST`` — free-list based allocation (object pooling)

The active protocol is set via ``MarketDataBuffer`` configuration.
```

---

### Task 9: Create engines.rst

**Files:**
- Create: `docs/source/engines.rst`

- [ ] **Step 1: Write engine layer guide**

```rst
Engines
=======

The engine layer is the orchestration core of PyAlgoEngine. It provides
event-driven dispatch, market data management, algorithm execution, and
trade order routing.

Event Engine
------------

The event engine implements a pub/sub pattern on typed topics. It is a
Cython wrapper around ``PyEventEngine``.

.. code-block:: python

   from algo_engine.engine import EVENT_ENGINE, TOPIC

   # Register a handler
   def on_tick_event(event):
       print(f"Got tick: {event.data.ticker}")

   EVENT_ENGINE.register_handler(TOPIC.TICK, on_tick_event)

   # Unregister
   EVENT_ENGINE.unregister_handler(TOPIC.TICK, on_tick_event)

Key topics in ``TOPIC`` enum:

* ``TICK`` — new tick data
* ``BAR`` — new bar data
* ``ORDER`` — new order
* ``TRADE`` — trade execution
* ``REPORT`` — trade report
* ``TIMER`` — periodic timer events
* ``MD_BUFFER_UPDATE`` — buffer state change

Market Data Service (MDS)
--------------------------

The MDS manages market data subscriptions, dispatching, and monitors.

.. code-block:: python

   from algo_engine.engine import MDS

   # Subscribe to market data for specific tickers
   MDS.subscribe_tick("AAPL")
   MDS.subscribe_bar("AAPL")
   MDS.subscribe_order("AAPL")

   # Unsubscribe
   MDS.unsubscribe_tick("AAPL")

   # Check subscription status
   is_subscribed = MDS.is_subscribed("AAPL")

   # Get current subscriptions
   tickers = MDS.get_subscribed_tickers()

Market Data Monitors
~~~~~~~~~~~~~~~~~~~~

Monitors process market data events with fine-grained control:

.. code-block:: python

   from algo_engine.engine import MarketDataMonitor, MonitorManager

   class MyMonitor(MarketDataMonitor):
       def on_tick(self, tick):
           # Called for each tick update on subscribed tickers
           print(f"{tick.ticker}: {tick.last}")

       def on_bar(self, bar):
           # Called for each bar update
           print(f"{bar.ticker}: O={bar.open} C={bar.close}")

       def on_order(self, order):
           # Called for each order
           pass

       def on_trade(self, trade):
           # Called for each trade
           pass

   monitor = MyMonitor()
   MDS.add_monitor(monitor)
   MDS.remove_monitor(monitor)

Algo Engine
-----------

The algorithm engine manages the lifecycle of trading algorithms:

.. code-block:: python

   from algo_engine.engine import AlgoTemplate, ALGO_ENGINE, ALGO_REGISTRY

   class MyAlgo(AlgoTemplate):
       \"\"\"Custom trading algorithm.\"\"\"

       def on_start(self):
           \"\"\"Called when the algo is started.\"\"\"
           self.subscribe("AAPL")

       def on_tick(self, tick):
           \"\"\"React to tick data.\"\"\"
           if tick.last > self.reference_price * 1.01:
               self.buy(tick.ticker, price=tick.last, volume=100)

       def on_report(self, report):
           \"\"\"Handle trade execution reports.\"\"\"
           print(f"Filled: {report.filled_volume} @ {report.avg_price}")

       def on_stop(self):
           \"\"\"Called when the algo is stopped.\"\"\"
           pass

   # Register the algo class
   ALGO_REGISTRY.register("my_algo", MyAlgo)

   # Instantiate and run
   algo = ALGO_ENGINE.start_algo("my_algo", reference_price=150.0)
   ALGO_ENGINE.stop_algo(algo.id)

The ``AlgoTemplate`` provides built-in methods:

* ``subscribe(ticker)`` — subscribe to market data
* ``buy(ticker, price, volume)`` — send a buy order
* ``sell(ticker, price, volume)`` — send a sell order
* ``cancel(order_id)`` — cancel an order
* ``get_position(ticker)`` — query current position

Trade Engine
------------

The trade engine handles order routing, position tracking, and risk
management.

.. code-block:: python

   from algo_engine.engine import DirectMarketAccess, Balance
   from algo_engine.engine import PositionManagementService, RiskProfile

   # Direct Market Access — send orders to the market/broker
   dma = DirectMarketAccess()

   # Balance tracks capital and positions
   balance = Balance(initial_capital=1_000_000.0)
   balance.update_position("AAPL", volume=100, price=150.0)
   print(balance.equity)  # current total equity

   # Position management
   pms = PositionManagementService()

   # Risk profile
   risk = RiskProfile(max_position_pct=0.10, max_drawdown=0.05)

Pre-built singletons (imported from ``algo_engine.strategy``):

.. code-block:: python

   from algo_engine.strategy import BALANCE, DMA, POSITION_TRACKER

   # These are global instances shared across the strategy layer
   print(BALANCE.equity)
   DMA.send_order(order)
```

---

### Task 10: Create strategy.rst

**Files:**
- Create: `docs/source/strategy.rst`

- [ ] **Step 1: Write strategy development guide**

```rst
Strategy Development
====================

This guide walks through building trading strategies with PyAlgoEngine's
strategy framework.

Strategy Engine Template
------------------------

The ``StrategyEngineTemplate`` is the base class for strategy engines. It
processes market data, generates signals, and manages order flow:

.. code-block:: python

   from algo_engine.strategy import StrategyEngineTemplate

   class MyStrategyEngine(StrategyEngineTemplate):
       def on_market_data(self, md_data):
           \"\"\"Process incoming market data.\"\"\"
           # md_data could be TickData, BarData, etc.
           pass

       def on_report(self, report):
           \"\"\"Process trade execution reports.\"\"\"
           pass

       def on_order(self, order):
           \"\"\"Process order state changes.\"\"\"
           pass

Global Singletons
-----------------

The strategy layer provides pre-instantiated singletons for common
services:

.. code-block:: python

   from algo_engine.strategy import (
       STRATEGY_ENGINE,   # Strategy engine instance
       BALANCE,           # Account balance / equity tracker
       DMA,               # Direct Market Access (EventDMA)
       POSITION_TRACKER,  # Position tracking service
       INVENTORY,         # Inventory management
       RISK_PROFILE,      # Risk limits and monitoring
   )

These are shared across all strategies in the same process. In
backtesting, separate instances are created automatically.

Building a Strategy
-------------------

Here's a complete example of a moving-average crossover strategy:

.. code-block:: python

   from algo_engine.engine import AlgoTemplate
   from algo_engine.base import TransactionSide, TransactionDirection, TransactionOffset
   from collections import deque

   class MACrossover(AlgoTemplate):
       \"\"\"Moving average crossover strategy.\"\"\"

       def __init__(self, ticker, fast_period=5, slow_period=20):
           super().__init__()
           self.ticker = ticker
           self.fast_period = fast_period
           self.slow_period = slow_period
           self.prices = deque(maxlen=slow_period)
           self.position = 0

       def on_start(self):
           self.subscribe(self.ticker)
           print(f"MA Crossover started for {self.ticker}")

       def on_tick(self, tick):
           if tick.ticker != self.ticker:
               return

           self.prices.append(tick.last)
           if len(self.prices) < self.slow_period:
               return

           fast_ma = sum(list(self.prices)[-self.fast_period:]) / self.fast_period
           slow_ma = sum(self.prices) / self.slow_period

           if fast_ma > slow_ma and self.position <= 0:
               # Go long
               self.buy(self.ticker, price=tick.last, volume=100)
               self.position = 1
           elif fast_ma < slow_ma and self.position >= 0:
               # Go short / exit
               self.sell(self.ticker, price=tick.last, volume=100)
               self.position = -1

       def on_report(self, report):
           print(f"Filled {report.filled_volume} @ {report.avg_price:.2f}")

       def on_stop(self):
           print(f"MA Crossover stopped. Final PnL: ...")

Integration with Algo Engine
----------------------------

Strategies are run via the algo engine:

.. code-block:: python

   from algo_engine.engine import ALGO_ENGINE, ALGO_REGISTRY

   # Register
   ALGO_REGISTRY.register("ma_cross", MACrossover)

   # Start
   algo = ALGO_ENGINE.start_algo("ma_cross", ticker="AAPL", fast_period=5, slow_period=20)

   # Stop
   ALGO_ENGINE.stop_algo(algo.id)

Next Steps
----------

- :doc:`backtest` — backtest your strategy on historical data
- :doc:`engines` — understand the underlying engine architecture
```

---

### Task 11: Create backtest.rst

**Files:**
- Create: `docs/source/backtest.rst`

- [ ] **Step 1: Write backtesting guide**

```rst
Backtesting
===========

PyAlgoEngine provides a complete backtesting framework: data replay,
simulated order matching, and performance metrics.

Data Replay
-----------

Replay engines feed historical market data into the system as if it were
live:

.. code-block:: python

   from algo_engine.backtest import SimpleReplay, ProgressReplay

   # Simple replay — processes all data as fast as possible
   replay = SimpleReplay(
       data_source=your_data_iterator,
       start_date="2024-01-02",
       end_date="2024-01-31",
   )
   replay.run()

   # Progress replay — respects timestamps, replays at realistic pace
   replay = ProgressReplay(
       data_source=your_data_iterator,
       speed=1.0,  # 1.0 = real-time, 10.0 = 10x speed
   )
   replay.run()

Data Playback Scopes
~~~~~~~~~~~~~~~~~~~~

Control what data is replayed with ``PyDataScope`` flags:

.. code-block:: python

   from algo_engine.backtest import PyDataScope

   scope = PyDataScope.TICK | PyDataScope.BAR | PyDataScope.ORDER
   replay = SimpleReplay(data_source=..., scope=scope)

Simulated Order Matching
------------------------

``SimMatch`` simulates exchange order matching with configurable
parameters:

.. code-block:: python

   from algo_engine.backtest import SimMatch

   sim = SimMatch(
       ticker="AAPL",
       fee_rate=0.0003,       # 3 bps
       slippage=0.0001,       # 1 bp
       latency_ms=1.0,        # 1ms delay
   )

   # SimMatch integrates with the backtest event loop
   # Orders from strategies are automatically routed through it

Trade Metrics
-------------

``TradeMetrics`` tracks performance throughout a backtest:

.. code-block:: python

   from algo_engine.backtest import TradeMetrics

   metrics = TradeMetrics()

   # After each trade
   metrics.add_trade(trade_report)

   # Summary statistics
   print(f"Total PnL: {metrics.total_pnl:.2f}")
   print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
   print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
   print(f"Win Rate: {metrics.win_rate:.2%}")
   print(f"Total Trades: {metrics.total_trades}")

Strategy Tester
---------------

The ``StrategyTester`` wraps everything into a one-shot backtest runner:

.. code-block:: python

   from algo_engine.apps import StrategyTester

   tester = StrategyTester(
       strategy_class=MyAlgo,
       strategy_kwargs={"ticker": "AAPL"},
       data_source=your_data_iterator,
       start_date="2024-01-02",
       end_date="2024-06-30",
   )
   results = tester.run()

   print(f"Final PnL: {results.pnl}")
   print(f"Sharpe: {results.sharpe}")

Running Backtests
-----------------

The ``algo_engine.backtest`` module can be run as ``__main__`` for
standalone backtest execution:

.. code-block:: bash

   python -m algo_engine.backtest --config backtest_config.json

The backtest ``__main__`` creates isolated singletons (event engine, MDS,
algo engine, balance, DMA, strategy engine) separate from any live
instances, registers event handlers, and runs the replay loop.

Next Steps
----------

- :doc:`web_apps` — visualize backtest results with Bokeh charts
- :doc:`strategy` — build strategies to backtest
```

---

### Task 12: Create web_apps.rst

**Files:**
- Create: `docs/source/web_apps.rst`

- [ ] **Step 1: Write web apps guide**

```rst
Web Applications
================

PyAlgoEngine includes web-based visualization tools built on Flask and
Bokeh for real-time monitoring and backtest analysis.

Flask Web App
-------------

The ``WebApp`` provides a web interface for backtesting:

.. code-block:: python

   from algo_engine.apps.backtest import WebApp, start_app

   # Create and configure
   app = WebApp()

   # Start the server
   start_app(app, host="127.0.0.1", port=5000)

Bokeh Dashboards
----------------

Interactive Bokeh charts for market data visualization:

.. code-block:: python

   from algo_engine.apps import DocServer, DocTheme

   # Create a documentation server for interactive plots
   server = DocServer()

Candlestick Charts
~~~~~~~~~~~~~~~~~~

Real-time candlestick charts:

.. code-block:: python

   from algo_engine.apps.backtest import CandleStick, StickTheme

   chart = CandleStick(
       ticker="AAPL",
       theme=StickTheme.DARK,
   )
   chart.update_bar(bar_data)  # push new bar, chart updates

Strategy Tester UI
------------------

The ``Tester`` and ``StrategyTester`` classes provide a programmatic
interface for running and visualizing backtests:

.. code-block:: python

   from algo_engine.apps import Tester, StrategyTester

   # Basic tester
   tester = Tester()

   # Full strategy tester with UI
   tester = StrategyTester(
       strategy_class=MyAlgo,
       strategy_kwargs={"ticker": "AAPL"},
       data_source=data_iterator,
       start_date="2024-01-02",
       end_date="2024-06-30",
   )
   tester.run()

Optional Dependencies
---------------------

Web features require additional packages:

.. code-block:: bash

   pip install PyAlgoEngine[WebApps]

This installs: ``flask``, ``waitress`` (production WSGI server), and
``bokeh``.

Simulated Input
---------------

For automated testing and demo scenarios, the ``sim_input`` subpackage
provides programmatic control of mouse and keyboard:

.. code-block:: python

   from algo_engine.apps.sim_input import SimKeyboard, SimMouse

   # Note: these are primarily for demo/testing automation
```

---

### Task 13: Create API reference pages

**Files:**
- Create: `docs/source/api/index.rst`
- Create: `docs/source/api/base.rst`
- Create: `docs/source/api/engine.rst`
- Create: `docs/source/api/backtest.rst`
- Create: `docs/source/api/exchange_profile.rst`
- Create: `docs/source/api/strategy.rst`
- Create: `docs/source/api/apps.rst`
- Create: `docs/source/api/monitor.rst`
- Create: `docs/source/api/utils.rst`

- [ ] **Step 1: Create api/index.rst**

```rst
API Reference
=============

Curated API documentation for each sub-package. For full auto-generated
module listings, see the :ref:`modindex`.

.. toctree::
   :maxdepth: 2

   base
   engine
   exchange_profile
   strategy
   backtest
   monitor
   apps
   utils
```

- [ ] **Step 2: Create api/base.rst**

```rst
algo_engine.base
================

The core layer provides C-level market data types, buffers, and memory
allocation. All types are implemented in Cython and compiled to native
extensions.

Market Data Types
-----------------

.. autoclass:: algo_engine.base.MarketData
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.TickData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.base.OrderBook
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.BarData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.base.DailyBar
   :members:
   :undoc-members:
   :show-inheritance:

Transaction Types
-----------------

.. autoclass:: algo_engine.base.TransactionData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.base.OrderData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.base.TradeData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.base.TransactionSide
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.TransactionDirection
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.TransactionOffset
   :members:
   :undoc-members:

Trade Utilities
---------------

.. autoclass:: algo_engine.base.OrderState
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.TradeReport
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.TradeInstruction
   :members:
   :undoc-members:

Data Buffers
------------

.. autoclass:: algo_engine.base.MarketDataBuffer
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.MarketDataRingBuffer
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.MarketDataConcurrentBuffer
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.MarketDataBufferCache
   :members:
   :undoc-members:

Allocator Protocol
------------------

.. autoclass:: algo_engine.base.c_allocator_protocol.MD_ALLOC
   :members:
   :undoc-members:

Configuration
-------------

.. autodata:: algo_engine.base.CONFIG
.. autodata:: algo_engine.base.C_CONFIG
.. autodata:: algo_engine.base.USE_CYTHON

Financial Utilities
-------------------

.. autoclass:: algo_engine.base.FinancialDecimal
   :members:
   :undoc-members:
```

- [ ] **Step 3: Create api/engine.rst**

```rst
algo_engine.engine
==================

The engine layer provides event-driven dispatch, market data management,
algorithm execution, and trade order routing.

Event Engine
------------

.. autoclass:: algo_engine.engine.c_event_engine.EventEngine
   :members:
   :undoc-members:

.. autodata:: algo_engine.engine.EVENT_ENGINE

.. autoclass:: algo_engine.engine.TOPIC
   :members:
   :undoc-members:

Market Data Service
-------------------

.. autoclass:: algo_engine.engine.MarketDataService
   :members:
   :undoc-members:

.. autodata:: algo_engine.engine.MDS

.. autoclass:: algo_engine.engine.MarketDataMonitor
   :members:
   :undoc-members:

.. autoclass:: algo_engine.engine.MonitorManager
   :members:
   :undoc-members:

Algo Engine
-----------

.. autoclass:: algo_engine.engine.AlgoTemplate
   :members:
   :undoc-members:

.. autoclass:: algo_engine.engine.AlgoEngine
   :members:
   :undoc-members:

.. autoclass:: algo_engine.engine.AlgoRegistry
   :members:
   :undoc-members:

.. autodata:: algo_engine.engine.ALGO_ENGINE
.. autodata:: algo_engine.engine.ALGO_REGISTRY

Trade Engine
------------

.. autoclass:: algo_engine.engine.DirectMarketAccess
   :members:
   :undoc-members:

.. autoclass:: algo_engine.engine.Balance
   :members:
   :undoc-members:

.. autoclass:: algo_engine.engine.PositionManagementService
   :members:
   :undoc-members:

.. autoclass:: algo_engine.engine.RiskProfile
   :members:
   :undoc-members:

Singleton
---------

.. autoclass:: algo_engine.engine.Singleton
   :members:
   :undoc-members:
```

- [ ] **Step 4: Create api/exchange_profile.rst**

```rst
algo_engine.exchange_profile
============================

Exchange profiles provide trading calendars, session times, and holiday
schedules. Both CN (A-share) and global exchange profiles are supported.

Exchange Profile
----------------

.. autoclass:: algo_engine.exchange_profile.ExchangeProfile
   :members:
   :undoc-members:

Global Dispatcher
-----------------

.. autodata:: algo_engine.exchange_profile.PROFILE

Default Profile (Global)
------------------------

.. autoclass:: algo_engine.exchange_profile.PROFILE_DEFAULT
   :members:
   :undoc-members:
   :show-inheritance:

CN Profile (A-Share)
--------------------

.. autoclass:: algo_engine.exchange_profile.PROFILE_CN
   :members:
   :undoc-members:
   :show-inheritance:

Session Types
-------------

.. autoclass:: algo_engine.exchange_profile.SessionDate
   :members:
   :undoc-members:

.. autoclass:: algo_engine.exchange_profile.SessionTime
   :members:
   :undoc-members:

.. autoclass:: algo_engine.exchange_profile.SessionPhase
   :members:
   :undoc-members:
```

- [ ] **Step 5: Create api/strategy.rst**

```rst
algo_engine.strategy
====================

The strategy layer provides the framework for building and running trading
strategies.

Strategy Engine
---------------

.. autoclass:: algo_engine.strategy.StrategyEngineTemplate
   :members:
   :undoc-members:

Singletons
----------

.. autodata:: algo_engine.strategy.STRATEGY_ENGINE
.. autodata:: algo_engine.strategy.BALANCE
.. autodata:: algo_engine.strategy.DMA
.. autodata:: algo_engine.strategy.POSITION_TRACKER
.. autodata:: algo_engine.strategy.INVENTORY
.. autodata:: algo_engine.strategy.RISK_PROFILE

Event DMA
---------

.. autoclass:: algo_engine.strategy.EventDMA
   :members:
   :undoc-members:
```

- [ ] **Step 6: Create api/backtest.rst**

```rst
algo_engine.backtest
====================

The backtesting framework provides data replay, simulated order matching,
and performance metrics.

Replay Engines
--------------

.. autoclass:: algo_engine.backtest.Replay
   :members:
   :undoc-members:

.. autoclass:: algo_engine.backtest.SimpleReplay
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.backtest.ProgressReplay
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.backtest.PyDataScope
   :members:
   :undoc-members:

Simulated Matching
------------------

.. autoclass:: algo_engine.backtest.SimMatch
   :members:
   :undoc-members:

Metrics
-------

.. autoclass:: algo_engine.backtest.TradeMetrics
   :members:
   :undoc-members:
```

- [ ] **Step 7: Create api/monitor.rst**

```rst
algo_engine.monitor
===================

Market data monitors provide real-time extensions for synthetic order
books and bar aggregation.

Monitors
--------

.. autoclass:: algo_engine.monitor.Monitor
   :members:
   :undoc-members:

.. autoclass:: algo_engine.monitor.SyntheticOrderBookMonitor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.monitor.MinuteBarMonitor
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

.. autoclass:: algo_engine.monitor.PyOrderBook
   :members:
   :undoc-members:
```

- [ ] **Step 8: Create api/apps.rst**

```rst
algo_engine.apps
================

Application-layer tools including web dashboards, backtest UI, and
simulated input.

Web Apps
--------

.. autoclass:: algo_engine.apps.WebApp
   :members:
   :undoc-members:

.. autofunction:: algo_engine.apps.start_app

Bokeh Server
------------

.. autoclass:: algo_engine.apps.DocServer
   :members:
   :undoc-members:

.. autoclass:: algo_engine.apps.DocTheme
   :members:
   :undoc-members:

Candlestick Charts
------------------

.. autoclass:: algo_engine.apps.backtest.CandleStick
   :members:
   :undoc-members:

.. autoclass:: algo_engine.apps.backtest.StickTheme
   :members:
   :undoc-members:

Strategy Tester
---------------

.. autoclass:: algo_engine.apps.Tester
   :members:
   :undoc-members:

.. autoclass:: algo_engine.apps.StrategyTester
   :members:
   :undoc-members:
```

- [ ] **Step 9: Create api/utils.rst**

```rst
algo_engine.utils
=================

Utility functions for time-series generation and test data creation.

Time-Series Utilities
---------------------

.. autofunction:: algo_engine.utils.ts_indices

Fake Data Generators
--------------------

.. autofunction:: algo_engine.utils.fake_data

.. autofunction:: algo_engine.utils.fake_daily_data
```

---

### Task 14: Remove obsolete files

**Files:**
- Delete: `docs/source/apis.rst`
- Delete: `docs/source/modules.rst`

- [ ] **Step 1: Delete obsolete RST files**

```bash
rm docs/source/apis.rst
rm docs/source/modules.rst
```

---

### Task 15: Build and verify

- [ ] **Step 1: Create api/ directory**

```bash
mkdir -p docs/source/api
```

- [ ] **Step 2: Build docs**

```bash
./docs/update_docs.sh
```

- [ ] **Step 3: Fix any Sphinx warnings**

Review build output. If there are warnings about missing autodoc targets,
adjust the `.. autoclass::` directives to match actual class names.

- [ ] **Step 4: Verify key pages**

```bash
# Check that key HTML files exist and have content
ls -la docs/build/html/index.html
ls -la docs/build/html/setup.html
ls -la docs/build/html/usage.html
ls -la docs/build/html/architecture.html
ls -la docs/build/html/market_data.html
ls -la docs/build/html/backtest.html
ls -la docs/build/html/strategy.html
ls -la docs/build/html/engines.html
ls -la docs/build/html/web_apps.html
ls -la docs/build/html/api/index.html
```

- [ ] **Step 5: Commit**

```bash
git add README.md docs/
git commit -m "docs: comprehensive documentation overhaul

- Rewrite README.md with accurate project description and architecture
- Replace index.rst with correct HFT engine landing page
- Rewrite setup.rst with 3 build methods, compile-time config, platform notes
- Rewrite usage.rst with real quick-start examples
- Add architecture.rst with layer diagram and data flow
- Add market_data.rst covering all data types and buffers
- Add engines.rst covering event/MDS/algo/trade engines
- Add strategy.rst with strategy development guide
- Add backtest.rst with replay, SimMatch, and metrics
- Add web_apps.rst with Flask/Bokeh visualization
- Add curated API reference (api/*.rst) with autodoc directives
- Update conf.py with intersphinx, autosummary, RTD config
- Update update_docs.sh with venv support and better error handling
- Remove obsolete apis.rst and modules.rst"
```
```

---

### Task 16: Fix build issues (iterative)

After Task 15, build may reveal issues. Common fixes:

- [ ] **Cython import warnings**: Add `suppress_warnings = ['import']` to conf.py (already in Task 1)
- [ ] **Missing autodoc targets**: Check actual class names with `python -c "from algo_engine.base import *; print(dir())"` and update `.. autoclass::` directives
- [ ] **Duplicate labels**: Run `sphinx-apidoc` with `-f` to force-overwrite, or manually clean `api/` directory first
```

---

### Task 17: Final verification

- [ ] **Step 1: Clean build from scratch**

```bash
cd docs && make clean && ./update_docs.sh
```

- [ ] **Step 2: Confirm zero warnings**

The build must complete with zero warnings. If not, fix and rebuild.

- [ ] **Step 3: Verify rendered content**

Open `docs/build/html/index.html` in a browser or use `lynx` / `elinks` to verify:
- All pages are accessible from the navigation
- Toctree is correct
- Code blocks are syntax-highlighted
- API pages show class/function documentation
