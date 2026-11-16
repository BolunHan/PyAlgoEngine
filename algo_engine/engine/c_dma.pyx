import abc
import traceback

from cpython.object cimport PyObject
from cpython.unicode cimport PyUnicode_AsUTF8
from libc.math cimport NAN, isnan

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR
from algo_engine.base.c_market_data.c_market_data cimport md_variant, instruction_from_header, report_from_header
from algo_engine.base.c_market_data.c_trade_utils cimport TradeInstruction, TradeReport

from . import LOGGER
from .c_event_engine import EVENT_ENGINE, TOPIC_SET
from .c_market_engine cimport MDS


class DirectMarketAccessProtocol(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def launch_order_handler(self, order: TradeInstruction):
        ...

    @abc.abstractmethod
    def cancel_order_handler(self, order: TradeInstruction):
        ...

    def balance_check_handler(self, balance: Balance):
        ...

    def rejected_order_callback(self, order: TradeInstruction):
        ...

    def filled_order_callback(self, order: TradeInstruction, report: TradeReport):
        ...


cdef class DirectMarketAccess(CCPType):
    def __init__(self):
        self.header = c_dma_new(MDS.header, MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True
        self.ccp_bind(&self.header)

        self.mds = MDS
        self.algo_engine = AlgoEngine.c_from_header(self.header.ALGO_ENGINE, False)

        self.risk_profile = RiskProfile.c_from_header(self.header.RISK_PROFILE, False)
        self.balance = Balance.c_from_header(self.header.BALANCE, False)

        if self.__class__ is not DirectMarketAccess:
            self.c_bind_protocol()

    def __dealloc__(self):
        if not self.owner:
            return

        if not self.header:
            return

        c_dma_detach_protocol_handler(self.header)
        c_dma_detach_protocol_callback(self.header)
        c_dma_free(self.header)
        self.header = NULL

    def __repr__(self):
        if not self.header:
            return f'<{self.__class__.__name__}: unbound>'
        return f'<{self.__class__.__name__}>(id={id(self)})'

    # === DMA Protocol ===

    def launch_order_handler(self, order: TradeInstruction):
        raise NotImplementedError()

    def cancel_order_handler(self, order: TradeInstruction):
        raise NotImplementedError()

    def balance_check_handler(self, balance: Balance):
        raise NotImplementedError()

    def rejected_order_callback(self, order: TradeInstruction):
        raise NotImplementedError()

    def filled_order_callback(self, order: TradeInstruction, report: TradeReport):
        raise NotImplementedError()

    cdef int c_bind_protocol(self):
        cdef dma_balance_query balance_query = DirectMarketAccess.c_balance_check_adaptor

        if <void*> self.c_launch_order_handler != <void*> DirectMarketAccess.c_launch_order_handler:
            pass
        elif type(self).launch_order_handler is not getattr(DirectMarketAccess, 'launch_order_handler'):
            pass
        else:
            return dma_ret_code.DMA_OK_NOOP

        if <void*> self.c_cancel_order_handler != <void*> DirectMarketAccess.c_cancel_order_handler:
            pass
        elif type(self).cancel_order_handler is not getattr(DirectMarketAccess, 'cancel_order_handler'):
            pass
        else:
            return dma_ret_code.DMA_OK_NOOP

        if <void*> self.c_balance_check_handler != <void*> DirectMarketAccess.c_balance_check_handler:
            pass
        elif type(self).balance_check_handler is not getattr(DirectMarketAccess, 'balance_check_handler'):
            pass
        else:
            balance_query = NULL

        cdef int ret_code = c_dma_attach_protocol_handler(
            self.header,
            DirectMarketAccess.c_launch_order_adaptor,
            DirectMarketAccess.c_cancel_order_adaptor,
            balance_query,
            <void*> <PyObject*> self
        )

        if ret_code != dma_ret_code.DMA_OK:
            return ret_code

        cdef dma_order_rejected_func rejected_fn = DirectMarketAccess.c_rejected_order_adaptor
        cdef dma_order_filled_func filled_fn = DirectMarketAccess.c_filled_order_adaptor

        if <void*> self.c_rejected_order_callback != <void*> DirectMarketAccess.c_rejected_order_callback:
            pass
        elif type(self).rejected_order_callback is not getattr(DirectMarketAccess, 'rejected_order_callback'):
            pass
        else:
            rejected_fn = NULL

        if <void*> self.c_filled_order_callback != <void*> DirectMarketAccess.c_filled_order_callback:
            pass
        elif type(self).filled_order_callback is not getattr(DirectMarketAccess, 'filled_order_callback'):
            pass
        else:
            filled_fn = NULL

        return c_dma_attach_protocol_callback(self.header, rejected_fn, filled_fn)

    # === Cython Internal Binding ===

    @staticmethod
    cdef inline void c_launch_order_adaptor(md_trade_instruction* order, uint32_t seq_id, void* user_data) noexcept with gil:
        cdef DirectMarketAccess self = <DirectMarketAccess> <object> <PyObject*> user_data
        self.c_launch_order_handler(order, seq_id)

    @staticmethod
    cdef inline void c_cancel_order_adaptor(const md_trade_instruction* order, uint32_t seq_id, void* user_data) noexcept with gil:
        cdef DirectMarketAccess self = <DirectMarketAccess> <object> <PyObject*> user_data
        self.c_cancel_order_handler(order, seq_id)

    @staticmethod
    cdef inline int c_balance_check_adaptor(const void* DMA, dma_balance* out, void* user_data) noexcept with gil:
        cdef DirectMarketAccess self = <DirectMarketAccess> <object> <PyObject*> user_data
        return self.c_balance_check_handler(DMA, out)

    @staticmethod
    cdef inline void c_rejected_order_adaptor(const md_trade_instruction* order, void* user_data) noexcept with gil:
        cdef DirectMarketAccess self = <DirectMarketAccess> <object> <PyObject*> user_data
        self.c_rejected_order_callback(order)

    @staticmethod
    cdef inline void c_filled_order_adaptor(const md_trade_instruction* order, md_trade_report* report, void* user_data) noexcept with gil:
        cdef DirectMarketAccess self = <DirectMarketAccess> <object> <PyObject*> user_data
        self.c_filled_order_callback(order, report)

    cdef void c_launch_order_handler(self, md_trade_instruction* order, uint32_t seq_id):
        cdef TradeInstruction py_order = instruction_from_header(<md_variant*> order, False)

        try:
            self.launch_order_handler(py_order)
        except Exception:
            LOGGER.error(traceback.format_exc())

    cdef void c_cancel_order_handler(self, const md_trade_instruction* order, uint32_t seq_id):
        cdef TradeInstruction py_order = instruction_from_header(<md_variant*> order, False)

        try:
            self.cancel_order_handler(py_order)
        except Exception:
            LOGGER.error(traceback.format_exc())

    cdef int c_balance_check_handler(self, const void* DMA, dma_balance* out):
        cdef Balance balance = Balance.c_from_header(out, False)

        try:
            return dma_ret_code.DMA_OK if self.balance_check_handler(balance) else dma_ret_code.DMA_ERR_INVALID_INPUT
        except Exception:
            LOGGER.error(traceback.format_exc())
            return dma_ret_code.DMA_ERR_INVALID_INPUT

    cdef void c_rejected_order_callback(self, const md_trade_instruction* order):
        cdef TradeInstruction py_order = instruction_from_header(<md_variant*> order, False)

        try:
            self.rejected_order_callback(py_order)
        except Exception:
            LOGGER.error(traceback.format_exc())

    cdef void c_filled_order_callback(self, const md_trade_instruction* order, md_trade_report* report):
        cdef TradeInstruction py_order = instruction_from_header(<md_variant*> order, False)
        cdef TradeReport py_report = report_from_header(<md_variant*> report, False)

        try:
            self.filled_order_callback(py_order, py_report)
        except Exception:
            LOGGER.error(traceback.format_exc())

    # === Python Interfaces ===

    cpdef int process_queue(self):
        cdef int ret_code = c_dma_process_queue(self.header)

        if ret_code < 0:
            raise RuntimeError(f'Failed to process the request queue, err code: {ret_code}')

        return ret_code

    cpdef object calculate_limit(self, str ticker, md_side side, object limit_price=None, object limit_adjust_factor=None, dma_algo_limit_mode mode=dma_algo_limit_mode.DMA_ALGO_LIMIT_STRICT):
        cdef double price = NAN if limit_price is None else limit_price
        cdef double factor = NAN if limit_adjust_factor is None else limit_adjust_factor

        cdef double result = c_dma_calculate_limit(
            self.header,
            PyUnicode_AsUTF8(ticker),
            side,
            price,
            factor,
            mode
        )

        if isnan(result):
            return None

        return result

    # === Python Properties ===

    property protocol_bound:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.header.protocol.handlers.order_launch_fn != NULL

    property timestamp:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.mds.timestamp

    property market_price:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.mds.market_price

    property market_time:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'Uninitialized <{self.__class__.__name__}>')
            return self.mds.market_time


DirectMarketAccessProtocol.register(DirectMarketAccess)


class EventDMA(DirectMarketAccess):
    def __init__(self, event_engine=None):
        self.event_engine = EVENT_ENGINE if event_engine is None else event_engine
        DirectMarketAccess.__init__(self)

    def launch_order_handler(self, order):
        self.event_engine.put(topic=TOPIC_SET.launch_order(ticker=order.ticker), order=order)

    def cancel_order_handler(self, order):
        self.event_engine.put(topic=TOPIC_SET.cancel_order(ticker=order.ticker), order_id=order.order_id)

    def rejected_order_callback(self, order):
        pass


cdef DirectMarketAccess DMA = <DirectMarketAccess> EventDMA()
cdef RiskProfile RISK_PROFILE = DMA.risk_profile
cdef Balance BALANCE = DMA.balance
globals()['DMA'] = DMA
globals()['RISK_PROFILE'] = RISK_PROFILE
globals()['BALANCE'] = BALANCE
