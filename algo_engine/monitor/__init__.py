from ..engine import MDS, MarketDataMonitor as Monitor


def add_synthetic_orderbook():
    # init synthetic orderbook monitor
    monitor = SyntheticOrderBookMonitor(mds=MDS)
    MDS.add_monitor(monitor=monitor)
    # override current orderbook
    MDS.bid = monitor.bid
    MDS.ask = monitor.ask


from .advanced_data_interface import *

__all__ = ['Monitor', 'SyntheticOrderBookMonitor', 'MinuteBarMonitor']
