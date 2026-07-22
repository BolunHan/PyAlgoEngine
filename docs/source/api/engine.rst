algo_engine.engine
====================

The engine layer provides event-driven dispatch, market data management,
algorithm execution, and trade order routing.

Event Engine
------------

.. autodata:: algo_engine.engine.EVENT_ENGINE

.. autoclass:: algo_engine.engine.TOPIC
   :members:
   :undoc-members:

Market Data Service
-------------------

.. autoclass:: algo_engine.engine.MarketDataService
   :members:
   :undoc-members:

.. autodata:: algo_engine.engine.MDS

.. autoclass:: algo_engine.engine.MarketDataMonitor
   :members:
   :undoc-members:

.. autoclass:: algo_engine.engine.MonitorManager
   :members:
   :undoc-members:

Algo Engine
-----------

.. autoclass:: algo_engine.engine.AlgoTemplate
   :members:
   :undoc-members:

.. autoclass:: algo_engine.engine.algo_engine.AlgoEngine
   :members:
   :undoc-members:

.. autoclass:: algo_engine.engine.algo_engine.AlgoRegistry
   :members:
   :undoc-members:

.. autodata:: algo_engine.engine.ALGO_ENGINE

.. autodata:: algo_engine.engine.ALGO_REGISTRY

Trade Engine
------------

.. autoclass:: algo_engine.engine.DirectMarketAccess
   :members:
   :undoc-members:

.. autoclass:: algo_engine.engine.Balance
   :members:
   :undoc-members:

.. autoclass:: algo_engine.engine.PositionManagementService
   :members:
   :undoc-members:

.. autoclass:: algo_engine.engine.RiskProfile
   :members:
   :undoc-members:

Singleton
---------

.. autoclass:: algo_engine.engine.Singleton
   :members:
   :undoc-members:
