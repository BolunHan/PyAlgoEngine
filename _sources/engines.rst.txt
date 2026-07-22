Engines
=======

The engine layer provides event-driven dispatch, market data management,
algorithm execution, and trade order routing.

Event Engine
------------

``EVENT_ENGINE`` is a singleton ``EventEngineEx`` (from ``PyEventEngine``)
providing pub/sub dispatch. ``TOPIC`` is a ``TopicSet`` helper that builds
and parses event topics.

.. code-block:: python

   from algo_engine.engine import EVENT_ENGINE, TOPIC

   # Register a handler for real-time market data
   def on_realtime(event):
       md = event.data
       print(f"{md.ticker}: {md.market_price}")

   topic = TOPIC.push(tick)   # resolves to "realtime.000001.SH.TickData"
   EVENT_ENGINE.register_handler(topic, on_realtime)
   EVENT_ENGINE.unregister_handler(topic, on_realtime)

Key ``TOPIC`` attributes:
  * ``on_order`` — order submission events
  * ``on_report`` — trade report events
  * ``bod`` / ``bod_done`` — start-of-day
  * ``eod`` / ``eod_done`` — end-of-day
  * ``launch_order`` — template: ``'launch_order.{ticker}'``
  * ``cancel_order`` — template: ``'cancel_order.{ticker}'``
  * ``realtime`` — template: ``'realtime.{ticker}.{dtype}'``

Market Data Service (MDS)
--------------------------

``MDS`` is a singleton ``MarketDataService`` that receives market data,
manages subscriptions automatically, and dispatches to monitors.

.. code-block:: python

   from algo_engine.engine import MDS

   # Feed market data — subscriptions tracked automatically
   MDS.on_market_data(tick)

   # Query state
   MDS.get_market_price("000001.SH")   # float (NaN if unknown)
   MDS.market_price                     # dict[ticker, float]
   MDS.subscriptions                    # dict[ticker, int]
   MDS.market_time                      # datetime | None
   MDS.n_subscribed                     # number of tickers with data
   MDS.timestamp                        # latest timestamp (float)

Market Data Monitors
~~~~~~~~~~~~~~~~~~~~

Monitors process incoming data and generate custom indices:

.. code-block:: python

   from algo_engine.engine import MarketDataMonitor, MarketDataService

   class PriceMonitor(MarketDataMonitor):
       def __init__(self, name="price_monitor", monitor_id=None):
           super().__init__(name, monitor_id)
           self._last_price = {}

       def __call__(self, market_data, **kwargs):
           self._last_price[market_data.ticker] = market_data.market_price

       def clear(self):
           self._last_price.clear()

       @property
       def value(self):
           return self._last_price

   monitor = PriceMonitor()
   MDS.add_monitor(monitor)
   # Access by ID: MDS[monitor.monitor_id]
   MDS.pop_monitor(monitor_name="price_monitor")

   # Get aggregated monitor values
   values = MDS.monitor_manager.get_values()

Algo Engine
-----------

Manages the lifecycle of ``AlgoTemplate`` subclasses:

.. code-block:: python

   from algo_engine.engine import AlgoTemplate, ALGO_ENGINE, ALGO_REGISTRY

   class MyAlgo(AlgoTemplate):
       """A simple algo that buys immediately."""
       def work(self):
           pass

       def launch(self, **kwargs):
           from algo_engine.base import TradeInstruction, OrderType, TransactionDirection, TransactionOffset
           return [TradeInstruction(
               ticker=self.ticker,
               timestamp=self.timestamp,
               side=TransactionDirection.DIRECTION_LONG | TransactionOffset.OFFSET_OPEN,
               volume=self.target_volume,
               limit_price=self.market_price * 0.99,
               order_type=OrderType.ORDER_LIMIT,
           )]

       def cancel(self, **kwargs):
           for oid in list(self.working_order.keys()):
               self._cancel_order(self.working_order[oid])

   # Register and run
   ALGO_REGISTRY.add_algo("my_algo", handler=MyAlgo)
   # ALGO_ENGINE handles algo lifecycle via the PositionManagementService

Pre-built algo templates (from ``ALGO_REGISTRY``):
  * ``passive`` — single limit order with adjustable price
  * ``passive_timeout`` — passive with time-based cancellation
  * ``aggressive`` — re-launches on fill/cancel until target reached
  * ``aggressive_timeout`` — aggressive with time-based cancellation

Trade Engine
------------

The trade engine handles order routing, position tracking, and risk:

.. code-block:: python

   from algo_engine.engine import DirectMarketAccess, Balance
   from algo_engine.engine import PositionManagementService, RiskProfile

   # These are typically accessed via strategy-level singletons:
   from algo_engine.strategy import (
       BALANCE,           # Balance singleton
       DMA,               # EventDMA (DirectMarketAccess subclass)
       POSITION_TRACKER,  # PositionManagementService
       INVENTORY,         # Inventory singleton
       RISK_PROFILE,      # RiskProfile singleton
   )

   # Check positions and PnL
   print(BALANCE.working_volume)
   print(BALANCE.exposure_volume)
   print(BALANCE.info)     # DataFrame summary

   # Risk checks (RiskProfile can be called with orders)
   RISK_PROFILE.set_rule("max_exposure_long", 100000.0, ticker="000001.SH")
