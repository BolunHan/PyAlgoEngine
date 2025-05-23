# cython: language_level=3
from libc.stdint cimport uint8_t

from .c_market_data cimport _MarketDataBuffer, _TradeReportBuffer, _TradeInstructionBuffer
from .c_transaction cimport TransactionData


cdef class OrderStateHelper:
    @staticmethod
    cdef bint is_working(int order_state)

    @staticmethod
    cdef bint is_done(int order_state)


cdef class TradeReport:
    cdef dict __dict__
    cdef _MarketDataBuffer* _data_ptr
    cdef _TradeReportBuffer _data

    cpdef TradeReport reset_order_id(self, object order_id=?)

    cpdef TradeReport reset_trade_id(self, object trade_id=?)

    cpdef TransactionData to_trade(self)

    cdef bytes c_to_bytes(self)

    @staticmethod
    cdef TradeReport c_from_bytes(bytes data)


cdef class TradeInstruction:
    cdef dict __dict__
    cdef _MarketDataBuffer* _data_ptr
    cdef _TradeInstructionBuffer _data
    cdef public dict trades

    cpdef TradeInstruction reset(self)

    cpdef TradeInstruction reset_order_id(self, object order_id=?)

    cpdef TradeInstruction set_order_state(self, uint8_t order_state, double timestamp=?)

    cpdef TradeInstruction fill(self, TradeReport trade_report)

    cpdef TradeInstruction add_trade(self, TradeReport trade_report)

    cpdef TradeInstruction cancel_order(self, double timestamp=?)

    cpdef TradeInstruction canceled(self, double timestamp=?)

    cdef bytes c_to_bytes(self)

    @staticmethod
    cdef TradeInstruction c_from_bytes(bytes data)
