import enum
import inspect
import json
import uuid
from typing import Protocol

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.object cimport PyObject
from cpython.ref cimport Py_XDECREF
from cpython.unicode cimport PyUnicode_AsUTF8, PyUnicode_FromString, PyUnicode_FromStringAndSize
from libc.math cimport fabs
from libc.stdint cimport uintptr_t
from libc.string cimport strcmp

from cbase.bytemap cimport bytemap_entry, c_bytemap_ex_len

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR
from algo_engine.base.c_market_data.c_market_data cimport c_md_side_name, md_side, md_trade_instruction, md_trade_report, md_variant
from algo_engine.base.c_market_data.c_trade_utils cimport TradeInstruction, TradeReport, instruction_from_header, report_from_header

from . import LOGGER
from .algo.c_algo_template cimport AlgoTemplate, c_dma_algo_stop, dma_algo
from .c_dma cimport DMA
from .c_dma_algo cimport ALGO_ENGINE, AlgoMapView


class PositionCallbackEvent(enum.IntEnum):
    canceled = dma_callback_event.on_canceled
    rejected = dma_callback_event.on_rejected
    listed = dma_callback_event.on_listed
    filled = dma_callback_event.on_filled

    Canceled = canceled
    Rejected = rejected
    Listed = listed
    Filled = filled


class PositionOrderCallbackProtocol(Protocol):
    def __call__(self, position: PositionManagementService, algo: AlgoTemplate, order: TradeInstruction):
        ...


class PositionTradeCallbackProtocol(Protocol):
    def __call__(self, position: PositionManagementService, algo: AlgoTemplate, order: TradeInstruction, trade: TradeReport):
        ...


cdef class PositionManagementService(CCPType):
    def __init__(self, str default_algo=None):
        self.dma = DMA
        self.algo_engine = ALGO_ENGINE

        self.header = c_dma_position_new(
            self.dma.header,
            self.algo_engine.header,
            DMA_POSITION_DEFAULT_ALGO if default_algo is None else PyUnicode_AsUTF8(default_algo),
            MD_DEFAULT_ALLOCATOR
        )

        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.ccp_bind(&self.header)

        self.on_canceled_callback = []
        self.on_rejected_callback = []
        self.on_listed_callback   = []
        self.on_filled_callback   = []

        self.callbacks = {
            'on_canceled': self.on_canceled_callback,
            'on_rejected': self.on_rejected_callback,
            'on_listed': self.on_listed_callback,
            'on_filled': self.on_filled_callback,
        }

    def __dealloc__(self):
        if not self.header:
            return

        if self.on_canceled_callback_id:
            c_dma_position_unregister_callback(self.header, self.on_canceled_callback_id)
        if self.on_rejected_callback_id:
            c_dma_position_unregister_callback(self.header, self.on_rejected_callback_id)
        if self.on_listed_callback_id:
            c_dma_position_unregister_callback(self.header, self.on_listed_callback_id)
        if self.on_filled_callback_id:
            c_dma_position_unregister_callback(self.header, self.on_filled_callback_id)

        self.ccp_unbind()

    cdef void __ccp_dealloc__(self):
        self.algos_map = None
        self.working_algos_map = None
        self.callbacks.clear()

    @staticmethod
    cdef inline int c_canceled_callback_adaptor(
            dma_position* position,
            const dma_algo* algo,
            const md_trade_instruction* order,
            void* user_data
    ) noexcept with gil:
        cdef PositionManagementService self = <PositionManagementService> <object> <PyObject*> user_data
        cdef AlgoTemplate py_algo = AlgoTemplate.c_from_header(<dma_algo*> algo)
        cdef TradeInstruction py_order = instruction_from_header(<const md_variant*> order, False)
        cdef PyObject* res = NULL
        cdef PyObject* exc = NULL

        cdef object callback
        for callback in self.on_canceled_callback:
            res = PyObject_CallFunctionObjArgs(<PyObject*> callback, <PyObject*> self, <PyObject*> py_algo, <PyObject*> py_order, NULL)
            if not res:
                exc = PyErr_GetRaisedException()
                if exc:
                    PyErr_DisplayException(exc)
                    Py_XDECREF(exc)
            else:
                Py_XDECREF(res)

        return position_ret_code.POSITION_OK

    @staticmethod
    cdef inline int c_rejected_callback_adaptor(
            dma_position* position,
            const dma_algo* algo,
            const md_trade_instruction* order,
            void* user_data
    ) noexcept with gil:
        cdef PositionManagementService self = <PositionManagementService> <object> <PyObject*> user_data
        cdef AlgoTemplate py_algo = AlgoTemplate.c_from_header(<dma_algo*> algo)
        cdef TradeInstruction py_order = instruction_from_header(<const md_variant*> order, False)
        cdef PyObject* res = NULL
        cdef PyObject* exc = NULL

        cdef object callback
        for callback in self.on_rejected_callback:
            res = PyObject_CallFunctionObjArgs(<PyObject*> callback, <PyObject*> self, <PyObject*> py_algo, <PyObject*> py_order, NULL)
            if not res:
                exc = PyErr_GetRaisedException()
                if exc:
                    PyErr_DisplayException(exc)
                    Py_XDECREF(exc)
            else:
                Py_XDECREF(res)

        return position_ret_code.POSITION_OK

    @staticmethod
    cdef inline int c_listed_callback_adaptor(
            dma_position* position,
            const dma_algo* algo,
            const md_trade_instruction* order,
            void* user_data
    ) noexcept with gil:
        cdef PositionManagementService self = <PositionManagementService> <object> <PyObject*> user_data
        cdef AlgoTemplate py_algo = AlgoTemplate.c_from_header(<dma_algo*> algo)
        cdef TradeInstruction py_order = instruction_from_header(<const md_variant*> order, False)
        cdef PyObject* res = NULL
        cdef PyObject* exc = NULL

        cdef object callback
        for callback in self.on_listed_callback:
            res = PyObject_CallFunctionObjArgs(<PyObject*> callback, <PyObject*> self, <PyObject*> py_algo, <PyObject*> py_order, NULL)
            if not res:
                exc = PyErr_GetRaisedException()
                if exc:
                    PyErr_DisplayException(exc)
                    Py_XDECREF(exc)
            else:
                Py_XDECREF(res)

        return position_ret_code.POSITION_OK

    @staticmethod
    cdef inline int c_filled_callback_adaptor(
            dma_position* position,
            const dma_algo* algo,
            const md_trade_instruction* order,
            const md_trade_report* trade,
            void* user_data
    ) noexcept with gil:
        cdef PositionManagementService self = <PositionManagementService> <object> <PyObject*> user_data
        cdef AlgoTemplate py_algo = AlgoTemplate.c_from_header(<dma_algo*> algo)
        cdef TradeInstruction py_order = instruction_from_header(<const md_variant*> order, False)
        cdef TradeReport py_trade = report_from_header(<const md_variant*> trade, False)
        cdef PyObject* res = NULL
        cdef PyObject* exc = NULL

        cdef object callback
        for callback in self.on_filled_callback:
            res = PyObject_CallFunctionObjArgs(<PyObject*> callback, <PyObject*> self, <PyObject*> py_algo, <PyObject*> py_order, <PyObject*> py_trade, NULL)
            if not res:
                exc = PyErr_GetRaisedException()
                if exc:
                    PyErr_DisplayException(exc)
                    Py_XDECREF(exc)
            else:
                Py_XDECREF(res)

        return position_ret_code.POSITION_OK

    def __repr__(self):
        return f'<{self.__class__.__name__}>{self.algos!r}'

    # === Cython Internal Binding ===

    cdef object c_register_callback(self, dma_callback_event event, object callback):
        cdef dict callbacks = self.callbacks
        cdef list handlers
        cdef object handler
        cdef object registered
        cdef dma_position_order_func order_func
        cdef dma_position_trade_func trade_func
        cdef void* callback_func = NULL
        cdef uintptr_t callback_id = 0
        cdef int ret_code

        if event == dma_callback_event.on_canceled:
            handlers = self.on_canceled_callback
        elif event == dma_callback_event.on_rejected:
            handlers = self.on_rejected_callback
        elif event == dma_callback_event.on_listed:
            handlers = self.on_listed_callback
        elif event == dma_callback_event.on_filled:
            handlers = self.on_filled_callback
        else:
            raise ValueError(f'Invalid position callback event <{event}>.')

        for registered in callbacks.values():
            for handler in registered:
                if handler is callback:
                    raise ValueError(f'Callback already registered for <{event!r}>.')

        handlers.append(callback)

        if len(handlers) == 1:
            if event == dma_callback_event.on_canceled:
                order_func    = PositionManagementService.c_canceled_callback_adaptor
                callback_func = <void*> order_func
            elif event == dma_callback_event.on_rejected:
                order_func    = PositionManagementService.c_rejected_callback_adaptor
                callback_func = <void*> order_func
            elif event == dma_callback_event.on_listed:
                order_func    = PositionManagementService.c_listed_callback_adaptor
                callback_func = <void*> order_func
            else:
                trade_func    = PositionManagementService.c_filled_callback_adaptor
                callback_func = <void*> trade_func

            ret_code = c_dma_position_register_callback(self.header, event, callback_func, <void*> <PyObject*> self, &callback_id)

            if ret_code != position_ret_code.POSITION_OK:
                handlers.pop()
                raise RuntimeError(f'Failed to register the callback, err code: {ret_code}')

            if event == dma_callback_event.on_canceled:
                self.on_canceled_callback_id = callback_id
            elif event == dma_callback_event.on_rejected:
                self.on_rejected_callback_id = callback_id
            elif event == dma_callback_event.on_listed:
                self.on_listed_callback_id = callback_id
            else:
                self.on_filled_callback_id = callback_id

        return None

    # === Python Interfaces ===

    def register_callback(self, dma_callback_event event, object callback):
        cdef object signature
        cdef int arity

        if not callable(callback):
            raise TypeError(f'<{self.__class__.__name__}> requires a callable callback.')

        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            arity = 4 if event == dma_callback_event.on_filled else 3

            try:
                signature.bind(*([None] * arity))
            except TypeError as error:
                raise TypeError(f'<{self.__class__.__name__}> <{event!r}> requires a callable accepting {arity} positional arguments: {error}') from error

        self.c_register_callback(event, callback)

    def unregister_callback(self, dma_callback_event event, object callback):
        cdef list handlers
        cdef int ret_code

        if event == dma_callback_event.on_canceled:
            handlers = self.on_canceled_callback
        elif event == dma_callback_event.on_rejected:
            handlers = self.on_rejected_callback
        elif event == dma_callback_event.on_listed:
            handlers = self.on_listed_callback
        elif event == dma_callback_event.on_filled:
            handlers = self.on_filled_callback
        else:
            raise ValueError(f'Invalid position callback event <{event}>.')

        if callback not in handlers:
            raise KeyError(f'Callback <{callback!r}> is not registered for <{event!r}>.')

        handlers.remove(callback)

        if handlers:
            return

        if event == dma_callback_event.on_canceled:
            ret_code = c_dma_position_unregister_callback(self.header, self.on_canceled_callback_id)
            self.on_canceled_callback_id = 0
        elif event == dma_callback_event.on_rejected:
            ret_code = c_dma_position_unregister_callback(self.header, self.on_rejected_callback_id)
            self.on_rejected_callback_id = 0
        elif event == dma_callback_event.on_listed:
            ret_code = c_dma_position_unregister_callback(self.header, self.on_listed_callback_id)
            self.on_listed_callback_id = 0
        else:
            ret_code = c_dma_position_unregister_callback(self.header, self.on_filled_callback_id)
            self.on_filled_callback_id = 0

        if ret_code != position_ret_code.POSITION_OK:
            raise RuntimeError(f'Failed to unregister the callback, err code: {ret_code}')

    cpdef void add_algo(self, AlgoTemplate algo):
        cdef int ret_code = c_dma_position_add_algo(self.header, algo.algo_meta)
        if ret_code == position_ret_code.POSITION_ERR_INVALID_INPUT:
            raise KeyError(f'Algo <{algo.algo_id}> already registered.')
        elif ret_code != position_ret_code.POSITION_OK:
            raise RuntimeError(f'Failed to register algo <{algo.algo_id}>, err code: {ret_code}')

    def open(self, str ticker, double target_volume, md_side trade_side, str algo=None, **kwargs):
        if target_volume <= 0:
            raise ValueError('target_volume must be positive!')

        algo_handler = self.algo_engine.registry.cast(name=algo or self.default_algo)(
            ticker=ticker,
            side=trade_side,
            target_volume=target_volume,
            **kwargs
        )

        LOGGER.info(f'{algo_handler} opening {ticker} {PyUnicode_FromString(c_md_side_name(trade_side))} {target_volume} position!')
        self.add_algo(<AlgoTemplate> algo_handler)
        algo_handler.launch(**kwargs)
        return algo_handler

    cpdef AlgoTemplate unwind(self, str ticker):
        LOGGER.info(f'fully cancel and unwind {ticker} position!')

        cdef const char* ticker_ptr = PyUnicode_AsUTF8(ticker)
        cdef bytemap_entry* entry    = self.header.working_algos.first
        cdef dma_algo* algo
        while entry:
            algo = (<dma_algo**> entry.value)[0]
            if algo.ticker:
                if strcmp(algo.ticker, ticker_ptr) == 0 and c_bytemap_ex_len(algo.working_orders):
                    c_dma_algo_stop(algo)
            entry = entry.next

        cdef double exposure = c_dma_position_get_exposure(self.header, ticker_ptr)
        cdef double working_long = 0.
        cdef double working_short = 0.
        c_dma_position_get_working(self.header, ticker_ptr, &working_long, &working_short)

        if not exposure:
            LOGGER.info(f'No exposure for {ticker}, no unwind actions!')
            return None
        elif working_long and working_short:
            LOGGER.info(f'Multiple trade actions for {ticker}, skip unwind actions! Try again later!')
            return None
        elif (exposure > 0 and working_short) or (exposure < 0 and working_long):
            LOGGER.info(f'Unwinding actions exists for {ticker}, skip unwind actions! Try again later!')
            return None

        cdef md_side side = md_side.SIDE_SHORT_CLOSE if exposure > 0 else md_side.SIDE_LONG_CLOSE
        return self.open(ticker=ticker, target_volume=fabs(exposure), trade_side=side)

    cpdef void add_exposure(self, str ticker, double volume, double notional, double multiplier, md_side side):
        cdef int ret_code = c_dma_position_add_exposure(self.header, PyUnicode_AsUTF8(ticker), volume, notional, multiplier, side)
        if ret_code != position_ret_code.POSITION_OK:
            raise RuntimeError(f'Failed to add exposure, err code: {ret_code}')

    cpdef void unwind_all(self):
        cdef int ret_code = c_dma_position_fully_unwind(self.header)
        if ret_code != position_ret_code.POSITION_OK:
            raise RuntimeError(f'Failed to unwind <{self.__class__.__name__}>, err code: {ret_code}')

    cpdef void cancel_all(self):
        cdef int ret_code = c_dma_position_fully_stop(self.header)
        if ret_code != position_ret_code.POSITION_OK:
            raise RuntimeError(f'Failed to stop <{self.__class__.__name__}>, err code: {ret_code}')

    def to_json(self, fmt='str'):
        cdef dict json_dict = {}
        cdef str map_id = str(self.position_id)

        json_dict[map_id] = {}

        for algo_id, algo in self.algos.items():
            if algo is not None:
                json_dict[map_id][algo_id] = algo.to_json(fmt='dict')

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    cpdef dict pnl(self):
        cdef dict pnl = {}
        cdef bytemap_entry* entry = self.header.tickers.first
        while entry:
            pnl[PyUnicode_FromStringAndSize(entry.key, entry.key_length)] = c_dma_position_pnl(self.header, entry.key)
            entry = entry.next
        return pnl

    # === Python Properties ===

    property position_id:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return uuid.UUID(bytes=PyBytes_FromStringAndSize(<const char*> self.header.position_id, 16))

    property default_algo:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return PyUnicode_FromString(self.header.default_algo)

    property algos:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if self.algos_map is None:
                self.algos_map = AlgoMapView.c_from_header(self.header.algos, False)
            return self.algos_map

    property working_algos:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if self.working_algos_map is None:
                self.working_algos_map = AlgoMapView.c_from_header(self.header.working_algos, False)
            return self.working_algos_map

    property notional:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef dict notional = {}

            cdef dma_position_ticker* cache
            cdef bytemap_entry* entry = self.header.tickers.first
            while entry:
                cache = <dma_position_ticker*> entry.value
                if cache.filled_notional:
                    notional[PyUnicode_FromStringAndSize(<const char*> entry.key, entry.key_length)] = cache.filled_notional
                entry = entry.next

            return notional

    property working_volume:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef dict working_long = {}
            cdef dict working_short = {}

            cdef dma_position_ticker* cache
            cdef bytemap_entry* entry = self.header.tickers.first
            while entry:
                cache = <dma_position_ticker*> entry.value
                if cache.working_long or cache.working_short:
                    if cache.working_long:
                        working_long[PyUnicode_FromStringAndSize(<const char*> entry.key, entry.key_length)] = cache.working_long
                    if cache.working_short:
                        working_short[PyUnicode_FromStringAndSize(<const char*> entry.key, entry.key_length)] = cache.working_short
                entry = entry.next

            return {'Long': working_long, 'Short': working_short}

    property exposure_volume:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef dict exposure = {}

            cdef dma_position_ticker* cache
            cdef bytemap_entry* entry = self.header.tickers.first
            while entry:
                cache = <dma_position_ticker*> entry.value
                if cache.exposure_volume:
                    exposure[PyUnicode_FromStringAndSize(<const char*> entry.key, entry.key_length)] = cache.exposure_volume
                entry = entry.next

            return exposure

    property working_volume_net:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef dict working = {}

            cdef dma_position_ticker* cache
            cdef double value
            cdef bytemap_entry* entry = self.header.tickers.first
            while entry:
                cache = <dma_position_ticker*> entry.value
                value = cache.working_long - cache.working_short
                if value:
                    working[PyUnicode_FromStringAndSize(<const char*> entry.key, entry.key_length)] = value
                entry = entry.next

            return working

    property orders:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef dict orders = {}

            cdef AlgoTemplate algo
            for algo in self.algos.values():
                orders.update(algo.orders)

            return orders

    property working_order:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef dict working_order = {}

            cdef AlgoTemplate algo
            for algo in self.algos.values():
                working_order.update(algo.working_orders)

            return working_order

    property trades:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef dict trades = {}

            cdef AlgoTemplate algo
            for algo in self.algos.values():
                for trade in algo.trades:
                    trades[trade.trade_id] = trade

            return trades

    property order_mapping:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef dict order_mapping = {}

            cdef AlgoTemplate algo
            for algo in self.algos.values():
                order_mapping[algo.algo_id] = algo.orders

            return order_mapping

    property reversed_order_mapping:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef dict reversed_order_mapping = {}

            cdef AlgoTemplate algo
            for algo in self.algos.values():
                for order_id in list(algo.orders):
                    reversed_order_mapping[order_id] = algo

            return reversed_order_mapping
