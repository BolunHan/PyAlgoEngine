import enum
import json
import uuid

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.unicode cimport PyUnicode_FromString
from libc.math cimport NAN

from cbase.bytemap cimport bytemap_entry, c_bytemap_ex_len

from algo_engine.base import TransactionSide
from algo_engine.base.c_market_data.c_trade_utils cimport report_from_header
from algo_engine.engine.c_dma_algo cimport dma_algo_ret_code, dma_order_ctx


class AlgoStatus(enum.IntEnum):
    idle = dma_algo_status.DMA_ALGO_STATUS_IDLE
    preparing = dma_algo_status.DMA_ALGO_STATUS_PREPARING
    ready = dma_algo_status.DMA_ALGO_STATUS_READY
    working = dma_algo_status.DMA_ALGO_STATUS_WORKING
    done = dma_algo_status.DMA_ALGO_STATUS_DONE
    closed = dma_algo_status.DMA_ALGO_STATUS_CLOSED
    stopping = dma_algo_status.DMA_ALGO_STATUS_STOPPING
    rejected = dma_algo_status.DMA_ALGO_STATUS_REJECTED
    error = dma_algo_status.DMA_ALGO_STATUS_ERROR

    Idle = idle
    Preparing = preparing
    Ready = ready
    Working = working
    Done = done
    Closed = closed
    Stopping = stopping
    Rejected = rejected
    Error = error


cdef class AlgoTemplate(CCPType):
    cdef void __ccp_dealloc__(self):
        self.orders_map = None
        self.working_orders_map = None

    @staticmethod
    cdef AlgoTemplate c_from_header(dma_algo* header):
        cdef AlgoTemplate instance = AlgoTemplate.__new__(AlgoTemplate)
        instance.algo_meta = header
        instance.ccp_bind(&instance.algo_meta)
        return instance

    # === Cython Internal Binding ===

    cdef int c_launch(self):
        cdef int ret_code = c_dma_algo_launch(self.algo_meta)

        if ret_code == dma_algo_ret_code.DMA_ALGO_OK:
            return ret_code
        elif ret_code == dma_algo_ret_code.DMA_ALGO_ERR_INVALID_INPUT:
            raise ValueError(f'<{self.__class__.__name__}> has no launch hook.')
        elif ret_code == dma_algo_ret_code.DMA_ALGO_ERR_OOM:
            raise MemoryError(f'<{self.__class__.__name__}> failed to allocate its launch order.')
        else:
            raise RuntimeError(f'c_dma_algo_launch failed with err code: {ret_code}')

    cdef int c_stop(self):
        cdef int ret_code = c_dma_algo_stop(self.algo_meta)

        if ret_code == dma_algo_ret_code.DMA_ALGO_OK:
            return ret_code
        elif ret_code == dma_algo_ret_code.DMA_ALGO_ERR_INVALID_INPUT:
            raise ValueError(f'<{self.__class__.__name__}> has no stop hook.')
        else:
            raise RuntimeError(f'c_dma_algo_stop failed with err code: {ret_code}')

    # === Python Dunders ===

    def __repr__(self):
        if not self.algo_meta:
            return f'<{self.__class__.__name__}> Unbound'
        return f'<{self.__class__.__name__} {self.ticker} {self.side.name} target={self.target_volume} status={self.status.name}>'

    # === Python Interfaces ===

    def launch(self, **kwargs):
        self.c_launch()

    def cancel(self, **kwargs):
        self.c_stop()

    def to_json(self, fmt='str'):
        if not self.algo_meta:
            raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

        cdef dict json_dict = {
            'algo_type': self.__class__.__name__,
            'ticker': self.ticker,
            'side': self.side.name,
            'target_volume': self.target_volume,
            'algo_id': self.algo_id.hex,
            'algo_reg_id': self.algo_reg_id,
            'status': self.status.name,
            'target_progress': self.filled_progress,
            'ts_started': self.ts_started,
            'ts_finished': self.ts_finished,
            'order': {order_id: order.to_json(fmt='dict') for order_id, order in self.orders.items()},
        }

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    # === Python Properties ===

    property algo_id:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return uuid.UUID(bytes=PyBytes_FromStringAndSize(<const char*> self.algo_meta.algo_id, 16))

    property algo_reg_id:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.algo_meta.algo_reg_id

    property ticker:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not self.algo_meta.ticker:
                return None
            return PyUnicode_FromString(self.algo_meta.ticker)

    property side:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return TransactionSide(<int> self.algo_meta.side)

    property status:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return AlgoStatus(self.algo_meta.status)

    property target_volume:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.algo_meta.target_volume

    property working_volume:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.algo_meta.working_volume

    property exposure_volume:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.algo_meta.exposure_volume

    property filled_volume:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.algo_meta.filled_volume

    property filled_notional:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.algo_meta.filled_notional

    property fee:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.algo_meta.fee

    property cash_flow:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.algo_meta.cash_flow

    property multiplier:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.algo_meta.multiplier

    property n_orders:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return c_bytemap_ex_len(self.algo_meta.orders)

    property n_working_orders:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return c_bytemap_ex_len(self.algo_meta.working_orders)

    property is_active:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.algo_meta.status == dma_algo_status.DMA_ALGO_STATUS_WORKING

    property average_price:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if self.algo_meta.filled_volume == 0.:
                return NAN
            return self.algo_meta.filled_notional / self.algo_meta.filled_volume

    property filled_progress:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.algo_meta.filled_volume / self.algo_meta.target_volume

    property placed_progress:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return abs(self.algo_meta.working_volume / self.algo_meta.target_volume) + self.filled_progress

    property ts_started:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not self.algo_meta.ts_started.date.year:
                return None
            return self.algo_meta.ts_started.unix_ts

    property ts_finished:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not self.algo_meta.ts_finished.date.year:
                return None
            return self.algo_meta.ts_finished.unix_ts

    property orders:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            if self.orders_map is None:
                self.orders_map = OrderMapView.c_from_header(self.algo_meta.orders, False)
            return self.orders_map

    property working_orders:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            if self.working_orders_map is None:
                self.working_orders_map = OrderMapView.c_from_header(self.algo_meta.working_orders, False)
            return self.working_orders_map

    property trades:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef bytemap_entry* order_entry = self.algo_meta.orders.first
            cdef bytemap_entry* report_entry
            cdef dma_order_ctx* ctx
            cdef md_trade_report* report

            while order_entry:
                ctx = <dma_order_ctx*> (<md_trade_instruction**> order_entry.value)[0]

                if ctx and ctx.reports:
                    report_entry = ctx.reports.first

                    while report_entry:
                        report = (<md_trade_report**> report_entry.value)[0]
                        if report:
                            yield report_from_header(<const md_variant*> report, False)
                        report_entry = report_entry.next
                order_entry = order_entry.next
