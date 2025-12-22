from .c_market_data cimport md_variant, md_order_state, MarketData
from .c_transaction cimport TransactionData


cdef class TradeReport(MarketData):

    cdef dict c_to_json(self)

    @staticmethod
    cdef TradeReport c_from_json(dict json_dict)

    cpdef TradeReport reset_order_id(self, object order_id=?)

    cpdef TradeReport reset_trade_id(self, object trade_id=?)

    cpdef TransactionData to_trade(self)


cdef class TradeInstruction(MarketData):
    cdef readonly dict trades

    cdef dict c_to_json(self)

    @staticmethod
    cdef TradeInstruction c_from_json(dict json_dict)

    cpdef TradeInstruction reset(self)

    cpdef TradeInstruction reset_order_id(self, object order_id=?)

    cpdef TradeInstruction set_order_state(self, md_order_state order_state, double timestamp=?)

    cpdef TradeInstruction fill(self, TradeReport trade_report)

    cpdef TradeInstruction add_trade(self, TradeReport trade_report)

    cpdef TradeInstruction cancel_order(self, double timestamp=?)

    cpdef TradeInstruction canceled(self, double timestamp=?)


cdef inline object report_from_header(md_variant* market_data, bint owner):
    cdef TradeReport instance = TradeReport.__new__(TradeReport)
    instance.header = market_data
    instance.owner = owner
    return instance


cdef inline object instruction_from_header(md_variant* market_data, bint owner):
    cdef TradeInstruction instance = TradeInstruction.__new__(TradeInstruction)
    instance.header = market_data
    instance.owner = owner
    return instance
