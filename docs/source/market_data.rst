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
  * ``order_book`` — full ``OrderBook`` with price/volume arrays
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
   )

   # A trade (fill) confirmation
   trade = TradeData(
       ticker="AAPL",
       price=150.48,
       volume=100,
       side=TransactionSide.BUY,
       trade_id="trd_001",
       order_id="ord_001",
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

LRU-style cache wrapper around a buffer for faster repeated access.

.. code-block:: python

   from algo_engine.base import MarketDataBufferCache

   cache = MarketDataBufferCache(buffer)
   cache.get_tick("AAPL")  # cached access

Allocator Protocols
-------------------

Memory allocation is controlled by configurable protocols:
  * ``MD_SHARED`` — shared memory allocation (inter-process)
  * ``MD_LOCKED`` — mutex-protected allocation
  * ``MD_LOCKFREE`` — lock-free allocation (atomic operations)
  * ``MD_FREELIST`` — free-list based allocation (object pooling)

The active protocol is set via ``MarketDataBuffer`` configuration.
For API details, see :doc:`api/base`.
