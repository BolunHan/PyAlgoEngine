algo_engine.exchange_profile
==============================

Exchange profiles provide trading calendars, session times, and holiday
schedules. Both CN (A-share) and global exchange profiles are supported.

Exchange Profile
----------------

.. autoclass:: algo_engine.exchange_profile.ExchangeProfile
   :members:
   :undoc-members:

Global Dispatcher
-----------------

.. autodata:: algo_engine.exchange_profile.PROFILE

Default Profile (Global)
------------------------

.. autodata:: algo_engine.exchange_profile.PROFILE_DEFAULT

CN Profile (A-Share)
--------------------

.. autodata:: algo_engine.exchange_profile.PROFILE_CN

Session Types
-------------

.. autoclass:: algo_engine.exchange_profile.SessionDate
   :members:
   :undoc-members:

.. autoclass:: algo_engine.exchange_profile.SessionTime
   :members:
   :undoc-members:

.. autoclass:: algo_engine.exchange_profile.SessionPhase
   :members:
   :undoc-members:
