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
