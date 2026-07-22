Strategy Development
====================

This guide walks through building trading strategies with PyAlgoEngine's
strategy framework.

Strategy Engine
---------------

The ``StrategyEngine`` is the base class for strategy engines. It
processes market data, generates signals, and manages order flow:

.. code-block:: python

   from algo_engine.strategy import StrategyEngine

   class MyStrategyEngine(StrategyEngine):
       def on_market_data(self, md_data):
           """Process incoming market data."""
           # md_data could be TickData, BarData, etc.
           pass

       def on_report(self, report):
           """Process trade execution reports."""
           pass

       def on_order(self, order):
           """Process order state changes."""
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
   from algo_engine.base import TransactionSide
   from collections import deque

   class MACrossover(AlgoTemplate):
       """Moving average crossover strategy."""

       def __init__(self, ticker, fast_period=5, slow_period=20):
           super().__init__()
           self.ticker = ticker
           self.fast_period = fast_period
           self.slow_period = slow_period
           self.prices = deque(maxlen=slow_period)
           self.position = 0

       def on_start(self):
           self.subscribe(self.ticker)

       def on_tick(self, tick):
           if tick.ticker != self.ticker:
               return

           self.prices.append(tick.last)
           if len(self.prices) < self.slow_period:
               return

           fast_ma = sum(list(self.prices)[-self.fast_period:]) / self.fast_period
           slow_ma = sum(self.prices) / self.slow_period

           if fast_ma > slow_ma and self.position <= 0:
               self.buy(self.ticker, price=tick.last, volume=100)
               self.position = 1
           elif fast_ma < slow_ma and self.position >= 0:
               self.sell(self.ticker, price=tick.last, volume=100)
               self.position = -1

       def on_report(self, report):
           print(f"Filled {report.filled_volume} @ {report.avg_price:.2f}")

       def on_stop(self):
           print(f"MA Crossover stopped")

Integration with Algo Engine
----------------------------

Strategies are run via the algo engine:

.. code-block:: python

   from algo_engine.engine import ALGO_ENGINE, ALGO_REGISTRY

   ALGO_REGISTRY.register("ma_cross", MACrossover)
   algo = ALGO_ENGINE.start_algo("ma_cross", ticker="AAPL",
                                  fast_period=5, slow_period=20)
   ALGO_ENGINE.stop_algo(algo.id)

Next Steps
----------

- :doc:`backtest` — backtest your strategy on historical data
- :doc:`engines` — understand the underlying engine architecture
- :doc:`api/strategy` — full strategy API reference
