Engines
=======

The engine layer is the orchestration core of PyAlgoEngine. It provides
event-driven dispatch, market data management, algorithm execution, and
trade order routing.

Event Engine
------------

The event engine implements a pub/sub pattern on typed topics. It is a
Cython wrapper around ``PyEventEngine``.

.. code-block:: python

   from algo_engine.engine import EVENT_ENGINE, TOPIC

   # Register a handler
   def on_tick_event(event):
       print(f"Got tick: {event.data.ticker}")

   EVENT_ENGINE.register_handler(TOPIC.TICK, on_tick_event)

   # Unregister
   EVENT_ENGINE.unregister_handler(TOPIC.TICK, on_tick_event)

Key topics in ``TOPIC`` enum:
  * ``TICK`` — new tick data
  * ``BAR`` — new bar data
  * ``ORDER`` — new order
  * ``TRADE`` — trade execution
  * ``REPORT`` — trade report
  * ``TIMER`` — periodic timer events
  * ``MD_BUFFER_UPDATE`` — buffer state change

Market Data Service (MDS)
--------------------------

The MDS manages market data subscriptions, dispatching, and monitors.

.. code-block:: python

   from algo_engine.engine import MDS

   # Subscribe to market data for specific tickers
   MDS.subscribe_tick("AAPL")
   MDS.subscribe_bar("AAPL")
   MDS.subscribe_order("AAPL")

   # Unsubscribe
   MDS.unsubscribe_tick("AAPL")

   # Check subscription status
   is_subscribed = MDS.is_subscribed("AAPL")

   # Get current subscriptions
   tickers = MDS.get_subscribed_tickers()

Market Data Monitors
~~~~~~~~~~~~~~~~~~~~

Monitors process market data events with fine-grained control:

.. code-block:: python

   from algo_engine.engine import MarketDataMonitor

   class MyMonitor(MarketDataMonitor):
       def on_tick(self, tick):
           print(f"{tick.ticker}: {tick.last}")

       def on_bar(self, bar):
           print(f"{bar.ticker}: O={bar.open} C={bar.close}")

       def on_order(self, order):
           pass

       def on_trade(self, trade):
           pass

   monitor = MyMonitor()
   MDS.add_monitor(monitor)
   MDS.remove_monitor(monitor)

Algo Engine
-----------

The algorithm engine manages the lifecycle of trading algorithms:

.. code-block:: python

   from algo_engine.engine import AlgoTemplate, ALGO_ENGINE, ALGO_REGISTRY

   class MyAlgo(AlgoTemplate):
       """Custom trading algorithm."""

       def on_start(self):
           self.subscribe("AAPL")

       def on_tick(self, tick):
           if tick.last > self.reference_price * 1.01:
               self.buy(tick.ticker, price=tick.last, volume=100)

       def on_report(self, report):
           print(f"Filled: {report.filled_volume} @ {report.avg_price}")

       def on_stop(self):
           pass

   # Register and run
   ALGO_REGISTRY.register("my_algo", MyAlgo)
   algo = ALGO_ENGINE.start_algo("my_algo", reference_price=150.0)
   ALGO_ENGINE.stop_algo(algo.id)

The ``AlgoTemplate`` provides built-in methods:
  * ``subscribe(ticker)`` — subscribe to market data
  * ``buy(ticker, price, volume)`` — send a buy order
  * ``sell(ticker, price, volume)`` — send a sell order
  * ``cancel(order_id)`` — cancel an order
  * ``get_position(ticker)`` — query current position

Trade Engine
------------

The trade engine handles order routing, position tracking, and risk
management.

.. code-block:: python

   from algo_engine.engine import DirectMarketAccess, Balance
   from algo_engine.engine import PositionManagementService, RiskProfile

   # Direct Market Access — send orders to the market/broker
   dma = DirectMarketAccess()

   # Balance tracks capital and positions
   balance = Balance(initial_capital=1_000_000.0)
   balance.update_position("AAPL", volume=100, price=150.0)
   print(balance.equity)

   # Risk profile
   risk = RiskProfile(max_position_pct=0.10, max_drawdown=0.05)

Pre-built singletons (imported from ``algo_engine.strategy``):

.. code-block:: python

   from algo_engine.strategy import BALANCE, DMA, POSITION_TRACKER

   print(BALANCE.equity)
   DMA.send_order(order)
