from .c_market_data cimport md_variant, MarketData


cdef class TransactionData(MarketData):
    pass


cdef class OrderData(MarketData):
    pass


cdef class TradeData(TransactionData):
    pass


cdef inline object transaction_from_header(md_variant* market_data, bint owner):
    cdef TransactionData instance = TransactionData.__new__(TransactionData)
    instance.header = market_data
    instance.owner = owner
    return instance


cdef inline object order_from_header(md_variant* market_data, bint owner):
    cdef OrderData instance = OrderData.__new__(OrderData)
    instance.header = market_data
    instance.owner = owner
    return instance
