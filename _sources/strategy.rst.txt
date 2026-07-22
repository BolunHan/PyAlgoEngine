Strategy Development
====================

This guide walks through building trading strategies with PyAlgoEngine.

Strategy Engine
---------------

``StrategyEngine`` extends ``StrategyEngineTemplate`` with event-engine
integration, handler management, and position operations.

.. code-block:: python

   from algo_engine.strategy import StrategyEngine
   # StrategyEngine is the concrete class; StrategyEngineTemplate is the ABC

   # In live trading, use the pre-built singleton:
   from algo_engine.strategy import STRATEGY_ENGINE

   # In backtesting, backtest/__main__.py creates isolated instances

Key methods on ``StrategyEngine``:
  * ``subscribe(ticker)`` — add ticker to subscription set
  * ``open_pos(ticker, volume, side=None, limit_price=None, algo=None)``
    — open a position via ``position_tracker.open()``
  * ``unwind_pos(ticker, volume, side=None, limit_price=None, algo=None)``
    — close position by opening an offsetting one
  * ``cancel(ticker, side=None, algo_id=None, order_id=None)``
    — cancel working algos by ticker/side or specific algo/order
  * ``stop()`` — cancel all algos and clear subscriptions
  * ``add_handler(on_market_data=..., on_report=..., on_order=...)``
    — register event handlers
  * ``back_test_lite(start_date, end_date, data_loader)``
    — run a lightweight backtest

Global Singletons
-----------------

.. code-block:: python

   from algo_engine.strategy import (
       STRATEGY_ENGINE,   # StrategyEngine instance
       BALANCE,           # Balance (capital, positions, PnL tracking)
       DMA,               # EventDMA (order launch/cancel via event engine)
       POSITION_TRACKER,  # PositionManagementService
       INVENTORY,         # Inventory (security holdings)
       RISK_PROFILE,      # RiskProfile (position/notional limits)
   )

These are created at import time with interlinked dependencies:
``BALANCE`` holds ``INVENTORY``; ``RISK_PROFILE`` references ``MDS`` and
``BALANCE``; ``DMA`` references ``MDS`` and ``RISK_PROFILE``;
``POSITION_TRACKER`` references ``DMA``; ``STRATEGY_ENGINE`` references
``EVENT_ENGINE`` and ``POSITION_TRACKER``.

Writing a Strategy
------------------

Here's a strategy that subscribes to a ticker and opens a position
on first data:

.. code-block:: python

   from algo_engine.strategy import STRATEGY_ENGINE
   from algo_engine.base import TransactionDirection, TransactionOffset

   class FirstTickStrategy:
       """Opens a long position on the first tick received."""
       def __init__(self, ticker, volume):
           self.ticker = ticker
           self.volume = volume
           self.started = False

       def on_market_data(self, market_data, **kwargs):
           if market_data.ticker != self.ticker or self.started:
               return
           self.started = True
           STRATEGY_ENGINE.open_pos(
               ticker=self.ticker,
               volume=self.volume,
               side=TransactionDirection.DIRECTION_LONG | TransactionOffset.OFFSET_OPEN,
           )

       def on_report(self, report, **kwargs):
           print(f"Filled: {report.filled_volume} @ {report.avg_price:.2f}")

       def on_order(self, order, **kwargs):
           print(f"Order state: {order.order_state}")

   # Attach to the strategy engine
   strat = FirstTickStrategy("000001.SH", volume=10000.0)
   STRATEGY_ENGINE.attach_strategy(strat)
   STRATEGY_ENGINE.subscribe("000001.SH")

Integration with Algo Engine
----------------------------

For algo-based execution, use ``STRATEGY_ENGINE.open_pos()`` which
internally calls ``POSITION_TRACKER.open()``, creating an ``AlgoTemplate``
subclass instance (by default ``ALGO_REGISTRY.cast("aggressive_timeout")``).

.. code-block:: python

   # Open a long position of 10,000 shares with a limit 1% below market
   filled, working = STRATEGY_ENGINE.open_pos(
       ticker="000001.SH",
       volume=10000.0,
       side=TransactionDirection.DIRECTION_LONG | TransactionOffset.OFFSET_OPEN,
       limit_price=MDS.get_market_price("000001.SH") * 0.99,
   )
   print(f"Filled: {filled}, Working: {working}")

   # Cancel all working orders for a ticker
   STRATEGY_ENGINE.cancel("000001.SH")

Next Steps
----------

- :doc:`backtest` — backtest your strategy on historical data
- :doc:`engines` — understand the underlying engine architecture
