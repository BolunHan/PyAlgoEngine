Quick Start
===========

This guide walks through basic usage with verified API calls.

Creating Market Data
--------------------

All market data types take ``ticker: str`` and ``timestamp: float``
(Unix timestamp, seconds since epoch) as their first keyword arguments.
Additional fields are type-specific and keyword-only.

.. code-block:: python

   import time
   from algo_engine.base import (
       TickData, TickDataLite, BarData, DailyBar,
       TransactionData, OrderData, TradeData,
       TransactionDirection, TransactionOffset, TransactionSide,
       OrderType, InternalData, MarketDataBuffer,
   )

   ts = time.time()

   # Level-1 tick (no order book)
   tick_lite = TickDataLite(
       ticker="000001.SH",
       timestamp=ts,
       last_price=15.28,
       bid_price=15.27,
       bid_volume=5000.0,
       ask_price=15.29,
       ask_volume=3200.0,
   )
   print(tick_lite.mid_price, tick_lite.spread)

   # Level-2 tick (with order book via kwargs)
   tick = TickData(
       ticker="000001.SH",
       timestamp=ts,
       last_price=15.28,
       # Order book levels passed as keyword arguments:
       bid_price_1=15.27, bid_volume_1=5000.0, bid_n_orders_1=12,
       ask_price_1=15.29, ask_volume_1=3200.0, ask_n_orders_1=8,
       bid_price_2=15.26, bid_volume_2=8000.0,
       ask_price_2=15.30, ask_volume_2=6000.0,
   )
   # Access best prices
   print(tick.best_bid_price, tick.best_ask_price)
   # Access order book sides
   print(tick.bid.at_level(0))   # (15.27, 5000.0, 12)
   print(tick.ask.at_price(15.29))  # lookup by price
   # Extract lite view (zero-copy or copy)
   lite_view = tick.lite(copy=False)

   # Bar (candlestick) — timestamp is bar END time
   bar = BarData(
       ticker="000001.SH",
       timestamp=ts,
       high_price=15.50,
       low_price=15.10,
       open_price=15.20,
       close_price=15.28,
       volume=1500000.0,
       notional=22800000.0,
       trade_count=3421,
       start_timestamp=ts - 300,      # bar start
       bar_span=300,                   # 5 minutes in seconds
   )
   print(bar.vwap, bar.bar_type)

   # Daily bar — uses market_date instead of timestamp
   from datetime import date
   daily = DailyBar(
       ticker="000001.SH",
       market_date=date(2024, 6, 15),
       high_price=15.80,
       low_price=15.00,
       open_price=15.20,
       close_price=15.45,
       volume=50000000.0,
   )
   print(daily.market_date)

   # Transaction — side is built via | operator
   txn = TransactionData(
       ticker="000001.SH",
       timestamp=ts,
       price=15.28,
       volume=1000.0,
       side=TransactionDirection.DIRECTION_LONG | TransactionOffset.OFFSET_OPEN,
       transaction_id="trd_001",
   )
   print(txn.side, txn.side_sign, txn.notional_flow)

   # Order — placed into the order book
   order = OrderData(
       ticker="000001.SH",
       timestamp=ts,
       price=15.25,
       volume=500.0,
       side=TransactionDirection.DIRECTION_LONG | TransactionOffset.OFFSET_ORDER,
       order_id="ord_001",
       order_type=OrderType.ORDER_LIMIT,
   )
   print(order.side, order.order_type)

   # TradeData alias — uses trade_* parameter names
   trade = TradeData(
       ticker="000001.SH",
       timestamp=ts,
       trade_price=15.28,
       trade_volume=1000.0,
       trade_side=TransactionDirection.DIRECTION_LONG | TransactionOffset.OFFSET_OPEN,
   )

   # InternalData — for heartbeats / control messages
   internal = InternalData(
       ticker="__CTRL__",
       timestamp=ts,
       code=0x01,
   )

Allocation Context
------------------

By default, ``MarketData`` allocates from shared memory (``MD_CFG_SHARED=True``)
and uses a freelist (``MD_CFG_FREELIST=True``). You can control this per
block of code with context managers:

.. code-block:: python

   from algo_engine.base import MD_SHARED, MD_LOCKED

   # Thread-safe, shared-memory allocation
   with MD_SHARED | MD_LOCKED:
       data = TickDataLite(ticker="TEST", timestamp=ts,
                           last_price=100.0, bid_price=99.0,
                           bid_volume=10.0, ask_price=101.0, ask_volume=10.0)

   # Book size context — override default book depth
   from algo_engine.base import MD_BOOK5, MD_BOOK20
   with MD_BOOK20:
       tick = TickData(ticker="TEST", timestamp=ts, last_price=100.0,
                       bid_price_1=99.0, bid_volume_1=10.0,
                       ask_price_1=101.0, ask_volume_1=10.0)

Using Market Data Buffers
--------------------------

Buffers store and sort serialized ``MarketData``. The generic ``put()``
accepts any ``MarketData`` subclass.

.. code-block:: python

   from algo_engine.base import (
       MarketDataBuffer, MarketDataBufferCache,
       MarketDataRingBuffer, MarketDataConcurrentBuffer,
   )

   # Create a buffer (default capacities from compile-time constants)
   buf = MarketDataBuffer(ptr_cap=128, data_cap=16384)

   # Put individual entries
   buf.put(tick)
   buf.put(bar)
   buf.put(order)

   # Batch writes via cache (more efficient)
   with buf.cache() as cache:
       cache.put(tick)
       cache.put(bar)
   # cache auto-flushes to buf on __exit__

   # Sort by timestamp, then iterate
   buf.sort()
   for md in buf:
       print(md.ticker, md.timestamp)

   # Indexed access
   first = buf[0]
   print(len(buf))          # number of entries
   print(buf.ptr_tail)      # same as len

   # Serialization
   raw = buf.to_bytes()
   buf2 = MarketDataBuffer.from_bytes(raw)

   # Ring buffer — fixed capacity, blocking reads/writes
   ring = MarketDataRingBuffer(ptr_cap=256, data_cap=32768)
   ring.put(tick, block=True, timeout=1.0)
   next_md = ring.listen(block=True, timeout=1.0)

   # Concurrent buffer — multi-consumer via shared memory
   cbuf = MarketDataConcurrentBuffer(n_workers=4, capacity=1024)
   cbuf.put(tick, block=True, timeout=1.0)
   md_for_worker = cbuf.listen(worker_id=0, block=True, timeout=1.0)
   print(cbuf.is_worker_empty(0))

Market Data Service (MDS)
-------------------------

``MDS`` is a singleton that receives market data, manages subscriptions,
and dispatches to monitors:

.. code-block:: python

   from algo_engine.engine import MDS, MarketDataMonitor

   # Feed data into MDS
   MDS.on_market_data(tick)

   # Get latest price for a ticker
   price = MDS.get_market_price("000001.SH")

   # All known prices
   print(MDS.market_price)       # dict[ticker, float]
   print(MDS.subscriptions)      # dict[ticker, int]

   # Create a custom monitor
   class MyMonitor(MarketDataMonitor):
       def __call__(self, market_data, **kwargs):
           print(f"{market_data.ticker}: {market_data.market_price}")

       def clear(self):
           pass

       @property
       def value(self):
           return 0.0

   monitor = MyMonitor(name="my_monitor")
   MDS.add_monitor(monitor)
   MDS.pop_monitor(monitor_name="my_monitor")

Using Exchange Profiles
-----------------------

Three module-level singletons provide exchange calendars:

.. code-block:: python

   from algo_engine.exchange_profile import PROFILE, PROFILE_DEFAULT, PROFILE_CN

   # PROFILE is a dispatcher that auto-detects the market
   # PROFILE_DEFAULT — global/24h markets (crypto)
   # PROFILE_CN — China A-share market

   # Check trading status
   PROFILE_CN.is_trading_day(date(2024, 6, 15))
   PROFILE_CN.is_market_session(time.time())

   # Get trading calendar
   cal = PROFILE_CN.trade_calendar(date(2024, 6, 1), date(2024, 6, 30))
   for d in cal:
       print(d, d.session_type)

   # Trading day arithmetic
   prev = PROFILE_CN.trading_days_before(date(2024, 6, 15), days=5)
   n_days = PROFILE_CN.trading_days_between(date(2024, 6, 1), date(2024, 6, 15))

   # Session phase at a given time
   from algo_engine.exchange_profile import SessionTime
   st = SessionTime(10, 30, 0)
   print(PROFILE_CN.resolve_session_phase(st))

Generating Test Data
--------------------

.. code-block:: python

   from algo_engine.utils import fake_data, fake_daily_data, ts_indices

   # Generate fake daily OHLC data
   daily_df = fake_daily_data(
       start_date="2024-01-02", end_date="2024-03-29",
       p0=100.0, volatility=0.20,
   )

   # Generate fake intraday data for one day
   intraday_df = fake_data(
       market_date=date(2024, 6, 15),
       p0=15.0, volatility=0.15, interval=60.0,
   )

   # Generate timestamp indices aligned to trading sessions
   indices = ts_indices(
       market_date=date(2024, 6, 15),
       interval=60.0,
       ts_format="timestamp",
   )

Next Steps
----------

- :doc:`architecture` — understand the system design
- :doc:`market_data` — deep dive into all data types and buffers
- :doc:`strategy` — build your first trading strategy
- :doc:`backtest` — run backtests on historical data
