Backtesting
===========

Data Replay
-----------

Replay engines feed historical market data into the system as if it were
live. Data loaders are callables that accept ``(market_date, ticker, dtype)``
or ``(market_date, tickers, dtypes)`` and return sequences of ``MarketData``.

.. code-block:: python

   from algo_engine.backtest import SimpleReplay, ProgressReplay, PyDataScope
   from datetime import date

   # Define a data loader (must match MarketDataLoader or MarketDataBulkLoader protocol)
   def my_loader(market_date, ticker, dtype):
       # Return a sequence of MarketData for (date, ticker, dtype)
       if dtype == "TickData":
           return [tick1, tick2, ...]
       return []

   # SimpleReplay — processes all data as fast as possible
   replay = SimpleReplay(
       loader=my_loader,
       start_date=date(2024, 1, 2),
       end_date=date(2024, 1, 31),
   )
   replay.add_subscription("000001.SH", "TickData")

   for md in replay:
       MDS.on_market_data(md)

   # ProgressReplay — adds progress bar (tqdm or native)
   replay = ProgressReplay(
       loader=my_loader,
       start_date=date(2024, 1, 2),
       end_date=date(2024, 1, 31),
   )
   replay.add_subscription("000001.SH", "TickData")
   for md in replay:
       MDS.on_market_data(md)

   # Control what data types are replayed
   scope = PyDataScope.SCOPE_TICK | PyDataScope.SCOPE_TRANSACTION

   # BOD / EOD callbacks
   def on_bod(market_date):
       print(f"Start of day: {market_date}")
   replay.add_bod(on_bod)

Simulated Order Matching
------------------------

``SimMatch`` simulates exchange order matching with configurable
parameters:

.. code-block:: python

   from algo_engine.backtest import SimMatch

   sim = SimMatch(
       ticker="000001.SH",
       fee_rate=0.0003,         # 3 bps
       slippage=0.0001,         # 1 bp
       instant_fill=False,      # require market data to match
       lag=5,                   # min data events before matching
   )

   # SimMatch registers on EVENT_ENGINE to intercept launch/cancel
   sim.register()

   # Feed market data to trigger matching
   sim(tick)    # checks working orders against incoming data

   # Clean up at end of day
   sim.eod()
   sim.unregister()

When a trade occurs, ``SimMatch`` publishes ``TradeReport`` via
``TOPIC.on_report`` and ``TradeInstruction`` updates via ``TOPIC.on_order``.

Trade Metrics
-------------

``TradeMetrics`` from ``algo_engine.backtest.metrics`` tracks performance:

.. code-block:: python

   from algo_engine.backtest.metrics import TradeMetrics

   metrics = TradeMetrics()
   metrics.add_trades(
       side=1,           # long=1, short=-1
       price=15.28,
       volume=1000.0,
       timestamp=1718400000.0,
       trade_id="trd_001",
   )
   metrics.update(market_price=15.50)   # mark-to-market
   print(metrics.summary)               # dict with win_rate, sharpe, etc.

Standalone Backtest
-------------------

``algo_engine.backtest.__main__`` creates isolated engine singletons
(EVENT_ENGINE, MDS, ALGO_ENGINE, BALANCE, DMA, STRATEGY_ENGINE) completely
separate from live instances, then runs the replay loop.

Strategy Tester
---------------

The ``StrategyTester`` in ``algo_engine.apps`` combines replay, matching,
metrics, and web visualization:

.. code-block:: python

   from algo_engine.apps import StrategyTester

   tester = StrategyTester(
       start_date=date(2024, 1, 2),
       end_date=date(2024, 6, 30),
       data_loader=my_loader,
       strategy=strat,
   )
   tester.register_ticker("000001.SH")
   tester.run()

Next Steps
----------

- :doc:`web_apps` — visualize backtest results
- :doc:`strategy` — build strategies to backtest
