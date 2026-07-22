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
singletons for the backtest context, completely separate from any live
instances.

Cython Acceleration
-------------------

Performance-critical components are implemented in Cython (``.pyx``) and
compiled to native extensions. The Cython layer provides:

* **C-level structs** for market data — no Python object overhead per tick
* **Shared memory allocators** — lock-free and locking protocols for IPC
* **Interned strings** — ticker symbols stored as integer IDs for fast
  comparison and hashing
* **Direct C function calls** — bypassing Python call overhead in hot paths

The build system compiles 15+ Cython extensions covering allocators,
market data types, buffers, exchange profiles, engines, and event handling.
See :doc:`setup` for build instructions and compile-time configuration.
