Market Data
===========

All market data types are C-level structures exposed to Python via Cython.
They provide Python-friendly accessors with near-C performance. Every type
inherits from :class:`~algo_engine.base.MarketData`.

Common ``MarketData`` properties: ``ticker``, ``timestamp`` (float, Unix
seconds), ``market_price``, ``market_time``, ``dtype``, ``topic``,
``session_date``, ``session_time``.

Data Types
----------

TickDataLite
~~~~~~~~~~~~

Level-1 snapshot without order book depth.

.. code-block:: python

   from algo_engine.base import TickDataLite

   tick = TickDataLite(
       ticker="000001.SH",
       timestamp=1718400000.0,
       last_price=15.28,
       bid_price=15.27,
       bid_volume=5000.0,
       ask_price=15.29,
       ask_volume=3200.0,
       open_price=15.20,
       prev_close=15.35,
       total_traded_volume=15000000.0,
       total_traded_notional=228000000.0,
       total_trade_count=3421,
   )

Properties: ``last_price``, ``bid_price``, ``bid_volume``, ``ask_price``,
``ask_volume``, ``open_price``, ``prev_close``, ``total_traded_volume``,
``total_traded_notional``, ``total_trade_count``.
Computed: ``mid_price`` ((bid+ask)/2), ``spread`` (ask-bid), ``market_price``.

TickData
~~~~~~~~

Level-2 snapshot with full order book. The order book depth is fixed at
compile time (``BOOK_SIZE``, default 10).

.. code-block:: python

   from algo_engine.base import TickData

   tick = TickData(
       ticker="000001.SH",
       timestamp=1718400000.0,
       last_price=15.28,
       total_traded_volume=15000000.0,
       total_bid_volume=120000.0,
       total_ask_volume=80000.0,
       weighted_bid_price=15.271,
       weighted_ask_price=15.292,
       # Order book via kwargs:
       bid_price_1=15.27, bid_volume_1=5000.0, bid_n_orders_1=12,
       ask_price_1=15.29, ask_volume_1=3200.0, ask_n_orders_1=8,
       bid_price_2=15.26, bid_volume_2=8000.0,
       ask_price_2=15.30, ask_volume_2=6000.0,
   )

   # Best prices
   tick.best_bid_price   # 15.27
   tick.best_ask_price   # 15.29

   # Order book access
   tick.bid.at_level(0)            # (15.27, 5000.0, 12)
   tick.bid.at_price(15.27)        # same, by price lookup
   tick.bid.loc_volume(15.25, 15.28)  # total vol in price range
   tick.bid.price                   # numpy array of prices
   tick.bid.volume                  # numpy array of volumes

   # Extract lite (zero-copy view or owned copy)
   lite = tick.lite(copy=False)    # non-owning view
   lite_copy = tick.lite(copy=True) # independent copy

Additional properties: ``total_bid_volume``, ``total_ask_volume``,
``weighted_bid_price``, ``weighted_ask_price``.

Parse order book data from a dict:
``tick.parse({"bid_price_1": 15.27, "bid_volume_1": 5000.0, ...})``

OrderBook
~~~~~~~~~

Represents one side (bid or ask). Construct standalone or access via
``tick.bid`` / ``tick.ask``.

.. code-block:: python

   from algo_engine.base import OrderBook, TransactionDirection

   ob = OrderBook(
       direction=TransactionDirection.DIRECTION_LONG,  # bid side
       price=[15.27, 15.26, 15.25],
       volume=[5000.0, 8000.0, 3000.0],
       n_orders=[12, 20, 5],
       is_sorted=True,
   )
   ob.sort()           # sort by price (ascending for bids, descending for asks)
   ob.to_numpy()       # 2D numpy array
   raw = ob.to_bytes()  # serialize

Properties: ``price`` (float64 array), ``volume`` (float64 array),
``n_orders`` (uint64 array), ``side`` (TransactionSide), ``direction``,
``size``, ``capacity``, ``sorted``.

BarData
~~~~~~~

OHLCV candlestick. **Timestamp is the END of the bar** to prevent
look-ahead bias.

.. code-block:: python

   from algo_engine.base import BarData

   bar = BarData(
       ticker="000001.SH",
       timestamp=1718400300.0,      # bar end
       high_price=15.50,
       low_price=15.10,
       open_price=15.20,
       close_price=15.28,
       volume=1500000.0,
       notional=22800000.0,
       trade_count=3421,
       start_timestamp=1718400000.0, # bar start
       bar_span=300.0,               # 5 min in seconds
   )

   bar.vwap              # volume-weighted average price
   bar.bar_type          # 'Minute', 'Hourly', 'Sub-Minute', etc.
   bar.bar_span          # timedelta
   bar.bar_start_time    # datetime
   bar.bar_end_time      # datetime

   # Dict-style access and mutation
   bar["close_price"]    # 15.28
   bar["close_price"] = 15.30

DailyBar
~~~~~~~~

Full-day candlestick. Uses ``market_date`` (``datetime.date``) instead of
``timestamp``.

.. code-block:: python

   from algo_engine.base import DailyBar
   from datetime import date

   daily = DailyBar(
       ticker="000001.SH",
       market_date=date(2024, 6, 15),
       high_price=15.80,
       low_price=15.00,
       open_price=15.20,
       close_price=15.45,
       volume=50000000.0,
       notional=760000000.0,
       trade_count=50000,
       bar_span=1,    # number of days
   )
   daily.market_date      # date(2024, 6, 15)
   daily.bar_type         # 'Daily' or 'Daily-Plus'

Transaction / Order / Trade
---------------------------

.. code-block:: python

   from algo_engine.base import (
       TransactionData, OrderData, TradeData,
       TransactionDirection, TransactionOffset, TransactionSide,
       OrderType,
   )

   # side = direction | offset  (int enum via | operator)
   side = TransactionDirection.DIRECTION_LONG | TransactionOffset.OFFSET_OPEN

   txn = TransactionData(
       ticker="000001.SH",
       timestamp=1718400000.0,
       price=15.28,
       volume=1000.0,
       side=side,
       multiplier=1.0,
       transaction_id="trd_001",
       buy_id="bid_001",
       sell_id="ask_001",
   )
   txn.side               # TransactionSide.SIDE_LONG_OPEN
   txn.side_sign          # 1 (long), -1 (short), 0 (cancel/unknown)
   txn.volume_flow        # signed volume (volume * side_sign)
   txn.notional_flow      # signed notional

   # Merge multiple transactions
   merged = TransactionData.merge([txn1, txn2, txn3])

   # Order — placed into the order book
   order = OrderData(
       ticker="000001.SH",
       timestamp=1718400000.0,
       price=15.25,
       volume=500.0,
       side=TransactionDirection.DIRECTION_LONG | TransactionOffset.OFFSET_ORDER,
       order_id="ord_001",
       order_type=OrderType.ORDER_LIMIT,
   )

   # TradeData — alias with trade_* parameter names
   trade = TradeData(
       ticker="000001.SH",
       timestamp=1718400000.0,
       trade_price=15.28,
       trade_volume=1000.0,
       trade_side=side,
   )

Enums:

* ``TransactionDirection``: ``DIRECTION_UNKNOWN``, ``DIRECTION_SHORT``,
  ``DIRECTION_LONG``, ``DIRECTION_NEUTRAL``
* ``TransactionOffset``: ``OFFSET_CANCEL``, ``OFFSET_ORDER``,
  ``OFFSET_OPEN``, ``OFFSET_CLOSE``
* ``OrderType``: ``ORDER_UNKNOWN``, ``ORDER_CANCEL``, ``ORDER_GENERIC``,
  ``ORDER_LIMIT``, ``ORDER_LIMIT_MAKER``, ``ORDER_MARKET``, ``ORDER_FOK``,
  ``ORDER_FAK``, ``ORDER_IOC``

Trade Utilities
~~~~~~~~~~~~~~~

.. code-block:: python

   from algo_engine.base import OrderState, TradeReport, TradeInstruction

   # TradeReport — execution confirmation
   report = TradeReport(
       ticker="000001.SH", timestamp=1718400000.0,
       price=15.28, volume=500.0,
       side=TransactionDirection.DIRECTION_LONG | TransactionOffset.OFFSET_OPEN,
       fee=2.5, order_id="ord_001", trade_id="trd_001",
   )

   # TradeInstruction — order lifecycle tracking
   instr = TradeInstruction(
       ticker="000001.SH", timestamp=1718400000.0,
       side=TransactionDirection.DIRECTION_LONG | TransactionOffset.OFFSET_OPEN,
       volume=1000.0, limit_price=15.25,
       order_type=OrderType.ORDER_LIMIT,
       order_id="ord_001",
   )
   instr.set_order_state(OrderState.STATE_PLACED)
   instr.fill(report)
   instr.cancel_order()
   print(instr.filled_volume, instr.average_price)
   print(instr.is_working, instr.is_done)

Data Buffers
------------

MarketDataBuffer
~~~~~~~~~~~~~~~~

Resizable block buffer for collecting and sorting market data.

.. code-block:: python

   from algo_engine.base import MarketDataBuffer

   buf = MarketDataBuffer(ptr_cap=128, data_cap=16384)

   # Single puts
   buf.put(tick)
   buf.put(bar)

   # Batch writes via cache (reduces reallocations)
   with buf.cache() as cache:
       for md in many_data:
           cache.put(md)

   buf.sort()             # sort by timestamp
   for md in buf:         # iterate chronologically
       print(md.ticker)

   first = buf[0]         # indexed access
   raw = buf.to_bytes()   # serialize
   buf2 = MarketDataBuffer.from_bytes(raw)

Properties: ``ptr_capacity``, ``ptr_tail`` (= len), ``data_capacity``, ``data_tail``.

MarketDataRingBuffer
~~~~~~~~~~~~~~~~~~~~

Fixed-capacity ring buffer with blocking read/write.

.. code-block:: python

   from algo_engine.base import MarketDataRingBuffer

   ring = MarketDataRingBuffer(ptr_cap=256, data_cap=32768)
   ring.put(tick, block=True, timeout=1.0)    # may raise BufferFull, PipeTimeoutError
   next_md = ring.listen(block=True, timeout=1.0)  # may raise BufferEmpty
   print(ring.is_empty)

MarketDataConcurrentBuffer
~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-consumer buffer backed by shared memory.

.. code-block:: python

   from algo_engine.base import MarketDataConcurrentBuffer

   cbuf = MarketDataConcurrentBuffer(n_workers=4, capacity=1024)
   cbuf.put(tick, block=True, timeout=1.0)
   md = cbuf.listen(worker_id=0, block=True, timeout=1.0)
   cbuf.disable_worker(2)
   cbuf.enable_worker(2)       # resets read pointer to current write position
   print(cbuf.is_full, cbuf.is_empty)

Allocator Protocols
-------------------

Context managers control how ``MarketData`` allocates its backing buffer:

.. code-block:: python

   from algo_engine.base import MD_SHARED, MD_LOCKED, MD_LOCKFREE, MD_FREELIST

   with MD_SHARED | MD_LOCKED:
       data = TickDataLite(ticker="TEST", timestamp=ts,
                           last_price=100.0, bid_price=99.0,
                           bid_volume=10.0, ask_price=101.0, ask_volume=10.0)

* ``MD_SHARED`` — allocate in shared memory (default: True)
* ``MD_LOCKED`` — thread-safe allocation (default: False)
* ``MD_FREELIST`` — use freelist on deallocation (default: True)
* ``MD_LOCKFREE`` — lock-free allocation

``FilterMode`` provides bitmask-based filtering for market data streams:

.. code-block:: python

   from algo_engine.base import FilterMode

   f = FilterMode.NO_TICK | FilterMode.NO_CANCEL
   f.mask_data(tick)        # True if data passes filter
   f.freeze()               # prevent further mutation
