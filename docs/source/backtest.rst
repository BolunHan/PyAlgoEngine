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

   from algo_engine.backtest.metrics import TradeMetrics

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
instances. This means you can run backtests in the same process without
interfering with production state.

Next Steps
----------

- :doc:`web_apps` — visualize backtest results with Bokeh charts
- :doc:`strategy` — build strategies to backtest
- :doc:`api/backtest` — full backtest API reference
