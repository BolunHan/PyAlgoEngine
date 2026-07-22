Web Applications
================

PyAlgoEngine includes web-based visualization tools built on Flask and
Bokeh for real-time monitoring and backtest analysis.

Flask Web App
-------------

The ``WebApp`` provides a web interface for backtesting:

.. code-block:: python

   from algo_engine.apps.backtest import WebApp, start_app

   # Create and configure
   app = WebApp()

   # Start the server
   start_app(app, host="127.0.0.1", port=5000)

Bokeh Dashboards
----------------

Interactive Bokeh charts for market data visualization:

.. code-block:: python

   from algo_engine.apps import DocServer, DocTheme

   server = DocServer()

Candlestick Charts
~~~~~~~~~~~~~~~~~~

Real-time candlestick charts:

.. code-block:: python

   from algo_engine.apps.backtest import CandleStick, StickTheme

   chart = CandleStick(
       ticker="AAPL",
       theme=StickTheme.DARK,
   )
   chart.update_bar(bar_data)  # push new bar, chart updates

Strategy Tester UI
------------------

The ``Tester`` and ``StrategyTester`` classes provide a programmatic
interface for running and visualizing backtests:

.. code-block:: python

   from algo_engine.apps import Tester, StrategyTester

   # Basic tester
   tester = Tester()

   # Full strategy tester with UI
   tester = StrategyTester(
       strategy_class=MyAlgo,
       strategy_kwargs={"ticker": "AAPL"},
       data_source=data_iterator,
       start_date="2024-01-02",
       end_date="2024-06-30",
   )
   tester.run()

Optional Dependencies
---------------------

Web features require additional packages:

.. code-block:: bash

   pip install PyAlgoEngine[WebApps]

This installs: ``flask``, ``waitress`` (production WSGI server), and
``bokeh``.

Simulated Input
---------------

For automated testing and demo scenarios, the ``sim_input`` subpackage
provides programmatic control of mouse and keyboard:

.. code-block:: python

   from algo_engine.apps.sim_input import SimKeyboard, SimMouse

   # Note: these are primarily for demo/testing automation
