from .c_market_data cimport md_variant, MarketData


cdef class BarData(MarketData):
    pass


cdef class DailyBar(BarData):
    pass


cdef inline object bar_from_header(md_variant* market_data, bint owner):
    cdef ts = market_data.bar_data.meta_info.timestamp
    cdef DailyBar daily_bar
    cdef BarData bar_data
    if ts < 1_0000_00_00:
        # The DailyBar.timestamp is repurposed to store datetime, in format YYYYMMDD
        # if the ts is less than 100M, this is highly likely to be a DailyBar, not BarData.
        daily_bar = DailyBar.__new__(DailyBar)
        daily_bar.header = market_data
        daily_bar.owner = owner
        return daily_bar
    else:
        bar_data = BarData.__new__(BarData)
        bar_data.header = market_data
        bar_data.owner = owner
        return bar_data
