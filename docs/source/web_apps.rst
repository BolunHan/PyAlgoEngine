Web Applications
================

Flask Web App
-------------

.. code-block:: python

   from algo_engine.apps.backtest import WebApp, start_app
   from datetime import date

   app = WebApp(
       start_date=date(2024, 1, 2),
       end_date=date(2024, 6, 30),
       name="My Backtest",
       address="0.0.0.0",
       port=8080,
   )
   app.register("000001.SH")   # creates CandleStick dashboard
   app.serve(blocking=True)

   # Or use the convenience function
   start_app(start_date=date(2024, 1, 2), end_date=date(2024, 6, 30))

``WebApp`` runs a Flask server with embedded Bokeh candlestick charts.
Each registered ticker gets its own dashboard page at ``/<ticker>``.

Bokeh Dashboards
----------------

.. code-block:: python

   from algo_engine.apps import DocServer, DocTheme
   from algo_engine.apps.backtest import CandleStick, StickTheme
   from datetime import date

   stick = CandleStick(
       ticker="000001.SH",
       start_date=date(2024, 1, 2),
       end_date=date(2024, 6, 30),
       interval=60.0,                    # bar interval in seconds
       theme=StickTheme(style="cn_style"),   # red-up/green-down
   )
   # Push market data to update the chart
   stick.update(tick)

   # Export to CSV
   stick.to_csv("000001.SH_bars.csv")

``StickTheme`` supports:
  * ``ws_style`` — green up, red down (Western)
  * ``cn_style`` — red up, green down (China)

Strategy Tester
---------------

.. code-block:: python

   from algo_engine.apps import Tester, StrategyTester
   from datetime import date

   tester = StrategyTester(
       start_date=date(2024, 1, 2),
       end_date=date(2024, 6, 30),
       data_loader=my_loader,
       strategy=my_strategy,
   )
   tester.register_ticker("000001.SH")
   # Automatically creates SimMatch, TradeMetrics, and optional WebApp
   tester.run()

``Tester`` is the abstract base; ``StrategyTester`` adds event-engine
integration, strategy dispatch, and position management.

Optional Dependencies
---------------------

.. code-block:: bash

   pip install PyAlgoEngine[WebApps]

Installs: ``flask``, ``waitress`` (production WSGI), ``bokeh``.
