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
- **Strategy Framework** — ``StrategyEngine`` and ``AlgoTemplate``
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
