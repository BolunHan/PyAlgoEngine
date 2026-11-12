from .c_market_data cimport md_variant, MarketData


cdef class BarData(MarketData):
    pass


cdef class DailyBar(BarData):
    pass


cdef inline object bar_from_header(const md_variant* market_data, bint owner):
    cdef DailyBar daily_bar
    cdef BarData bar_data
    if market_data.bar_data.meta_info.dt.unix_ts == 0. and market_data.bar_data.meta_info.dt.date.year != 0:
        # A date-only record is a DailyBar: dt.date carries the market date and
        # every other dt field is zero.
        daily_bar = DailyBar.__new__(DailyBar)
        daily_bar.header = market_data
        daily_bar.owner = owner
        return daily_bar
    else:
        bar_data = BarData.__new__(BarData)
        bar_data.header = market_data
        bar_data.owner = owner
        return bar_data
