from .c_market_data cimport md_variant, md_orderbook, MarketData


cdef class TickDataLite(MarketData):
    pass


cdef class OrderBook:
    cdef md_orderbook* header
    cdef bint owner
    cdef size_t iter_index

    cdef void c_sort(self)

    cdef double c_loc_volume(self, double p0, double p1)

    cdef bytes c_to_bytes(self)

    @staticmethod
    cdef OrderBook c_from_bytes(const char* data)

    cpdef tuple at_price(self, double price)

    cpdef tuple at_level(self, ssize_t idx)


cdef class TickData(MarketData):
    cdef readonly OrderBook bid
    cdef readonly OrderBook ask

    cpdef void parse(self, dict kwargs)

    cpdef TickDataLite lite(self)


cdef inline object tick_lite_from_header(md_variant* market_data, bint owner):
    cdef TickDataLite instance = TickDataLite.__new__(TickDataLite)
    instance.header = market_data
    instance.owner = owner
    return instance


cdef inline object tick_from_header(md_variant* market_data, bint owner):
    cdef TickData instance = TickData.__new__(TickData)
    instance.header = market_data

    instance.bid = OrderBook.__new__(OrderBook)
    instance.ask = OrderBook.__new__(OrderBook)
    instance.bid.header = market_data.tick_data_full.bid
    instance.ask.header = market_data.tick_data_full.ask

    instance.owner = owner
    return instance
