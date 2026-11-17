from libc.stdint cimport uint32_t
from libc.string cimport memset

from algo_engine.base.c_market_data.c_market_data cimport md_trade_instruction, md_trade_report, md_variant
from algo_engine.engine cimport MDS
from algo_engine.engine.c_balance cimport dma_balance, dma_balance_flag
from algo_engine.engine.c_dma cimport DirectMarketAccess


def market_price(bytes ticker):
    return MDS.c_get_market_price(ticker)


cdef class CythonDMA(DirectMarketAccess):
    """An access whose protocol is implemented entirely at the Cython level."""

    cdef public int launch_count
    cdef public int cancel_count
    cdef public int balance_count
    cdef public int rejected_count
    cdef public int filled_count

    def __init__(self):
        self.launch_count = 0
        self.cancel_count = 0
        self.balance_count = 0
        self.rejected_count = 0
        self.filled_count = 0
        DirectMarketAccess.__init__(self)

    cdef void c_launch_order_handler(self, md_trade_instruction* order, uint32_t seq_id):
        self.launch_count += 1

    cdef void c_cancel_order_handler(self, const md_trade_instruction* order, uint32_t seq_id):
        self.cancel_count += 1

    cdef int c_balance_check_handler(self, const void* DMA, dma_balance* out):
        self.balance_count += 1
        out.available_cash = 123.
        out.flags |= dma_balance_flag.DMA_BALANCE_F_AVAILABLE_CASH
        return 0

    cdef void c_rejected_order_callback(self, const md_trade_instruction* order):
        self.rejected_count += 1

    cdef void c_filled_order_callback(self, const md_trade_instruction* order, md_trade_report* report):
        self.filled_count += 1

    def fire_launch(self):
        cdef md_variant order
        memset(&order, 0, sizeof(md_variant))
        self.header.protocol.handlers.order_launch_fn(&order.trade_instruction, 1, self.header.protocol.user_data)

    def fire_cancel(self):
        cdef md_variant order
        memset(&order, 0, sizeof(md_variant))
        self.header.protocol.handlers.order_cancel_fn(&order.trade_instruction, 1, self.header.protocol.user_data)

    def fire_reject(self):
        cdef md_variant order
        memset(&order, 0, sizeof(md_variant))
        self.header.protocol.callbacks.order_rejected_fn(&order.trade_instruction, self.header.protocol.user_data)

    def fire_filled(self):
        cdef md_variant order
        cdef md_variant report
        memset(&order, 0, sizeof(md_variant))
        memset(&report, 0, sizeof(md_variant))
        self.header.protocol.callbacks.order_filled_fn(&order.trade_instruction, &report.trade_report, self.header.protocol.user_data)

    def fire_balance(self):
        cdef dma_balance balance
        memset(&balance, 0, sizeof(dma_balance))
        cdef int ret_code = self.header.protocol.handlers.balance_check_fn(NULL, &balance, self.header.protocol.user_data)
        return ret_code, balance.available_cash, balance.flags
