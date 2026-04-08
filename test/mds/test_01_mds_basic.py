import datetime
from math import isnan

from algo_engine.base import TransactionData, TransactionSide
from algo_engine.engine import MDS
from algo_engine.exchange_profile import PROFILE_CN

PROFILE_CN.activate()

td_1 = TransactionData(
    ticker='600010.SH',
    price=102.34,
    volume=125,
    timestamp=datetime.datetime(2024, 11, 11, 9, 45).timestamp(),
    side=TransactionSide.LongFilled
)

td_2 = TransactionData(
    ticker='600010.SH',
    price=105.20,
    volume=210,
    timestamp=datetime.datetime(2024, 11, 11, 11, 35).timestamp(),
    side=TransactionSide.ShortFilled
)

td_3 = TransactionData(
    ticker='600010.SH',
    price=102.13,
    volume=210,
    timestamp=datetime.datetime(2024, 11, 11, 9, 21).timestamp(),
    side=TransactionSide.SIDE_LONG_CANCEL
)

assert isnan(MDS.timestamp)
assert MDS.market_date is None
assert MDS.n_subscribed == 0
MDS.on_market_data(td_1)

assert MDS.get_market_price('600010.SH') == 102.34
assert MDS.timestamp == td_1.timestamp
assert MDS.market_time.replace(tzinfo=None) == datetime.datetime(2024, 11, 11, 9, 45)
assert MDS.market_date == datetime.date(2024, 11, 11)
assert MDS.n_subscribed == 1

MDS.on_market_data(td_2)
assert MDS.get_market_price('600010.SH') == 105.20
assert MDS.timestamp == td_2.timestamp
assert MDS.market_time.replace(tzinfo=None) == datetime.datetime(2024, 11, 11, 11, 35)
assert MDS.market_date == datetime.date(2024, 11, 11)
assert MDS.n_subscribed == 1

# this is expected to be non-monotonic in ts field, as the filter is not implemented.
MDS.on_market_data(td_3)
assert MDS.get_market_price('600010.SH') == 102.13
assert MDS.timestamp == td_3.timestamp
assert MDS.market_time.replace(tzinfo=None) == datetime.datetime(2024, 11, 11, 9, 21)
assert MDS.market_date == datetime.date(2024, 11, 11)
assert MDS.n_subscribed == 1
