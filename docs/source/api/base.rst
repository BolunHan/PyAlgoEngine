algo_engine.base
================

The core layer provides C-level market data types, buffers, and memory
allocation. All types are implemented in Cython and compiled to native
extensions.

Market Data Types
-----------------

.. autoclass:: algo_engine.base.MarketData
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.TickData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.base.OrderBook
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.BarData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.base.DailyBar
   :members:
   :undoc-members:
   :show-inheritance:

Transaction Types
-----------------

.. autoclass:: algo_engine.base.TransactionData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.base.OrderData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.base.TradeData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: algo_engine.base.TransactionSide
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.TransactionDirection
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.TransactionOffset
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.OrderType
   :members:
   :undoc-members:

Trade Utilities
---------------

.. autoclass:: algo_engine.base.OrderState
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.TradeReport
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.TradeInstruction
   :members:
   :undoc-members:

Data Buffers
------------

.. autoclass:: algo_engine.base.MarketDataBuffer
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.MarketDataRingBuffer
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.MarketDataConcurrentBuffer
   :members:
   :undoc-members:

.. autoclass:: algo_engine.base.MarketDataBufferCache
   :members:
   :undoc-members:

Configuration
-------------

.. autodata:: algo_engine.base.CONFIG

.. autodata:: algo_engine.base.USE_CYTHON

Financial Utilities
-------------------

.. autoclass:: algo_engine.base.FinancialDecimal
   :members:
   :undoc-members:
