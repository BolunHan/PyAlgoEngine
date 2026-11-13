import enum

from cpython.unicode cimport PyUnicode_AsUTF8AndSize, PyUnicode_FromStringAndSize

from cbase.bytemap cimport bytemap_entry, bytemap_ret_code, c_bytemap_ex_get_ptr, c_bytemap_gen_seq_id

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR
from algo_engine.base.c_market_data.c_market_data cimport MarketData, c_get_long_id
from algo_engine.base.c_market_data.c_trade_utils cimport TradeInstruction, TradeReport, instruction_from_header


class AlgoLimitMode(enum.IntEnum):
    strict = dma_algo_limit_mode.DMA_ALGO_LIMIT_STRICT
    loose = dma_algo_limit_mode.DMA_ALGO_LIMIT_LOOSE


cdef class OrderMapView(BoundByteMap):
    @staticmethod
    cdef OrderMapView c_from_header(bytemap* header, bint owner=False):
        if owner:
            raise PermissionError('Must not assign the order map ownership to <OrderMapView>')

        cdef OrderMapView instance = OrderMapView.__new__(OrderMapView)
        instance.owner = False
        instance.seq_id = c_bytemap_gen_seq_id(<void*> instance)
        instance.c_bind(header)
        return instance

    cdef const char* c_serialize_key(self, object obj, size_t* key_len):
        raise PermissionError(f'Must not assign order to {self.__class__.__name__}')

    cdef object c_deserialize_key(self, const char* key, size_t key_len):
        return c_get_long_id(<const long_md_id*> key)

    cdef const char* c_serialize_value(self, object obj, size_t* value_len):
        raise PermissionError(f'Must not assign order to {self.__class__.__name__}')

    cdef object c_deserialize_value(self, const char* value, size_t value_len):
        cdef md_trade_instruction* order = (<md_trade_instruction**> value)[0]
        if not order:
            return None
        return instruction_from_header(<const md_variant*> order, False)


cdef class AlgoRegistry(CCPType):
    @staticmethod
    cdef AlgoRegistry c_from_header(dma_algo_registry* header, bint owner):
        cdef AlgoRegistry instance = AlgoRegistry.__new__(AlgoRegistry)
        instance.header = header
        instance.owner = owner
        instance.classes = {}
        instance.ccp_bind(&instance.header)
        return instance

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_dma_algo_registry_free(self.header)

    cdef void __ccp_dealloc__(self):
        self.classes.clear()

    def __contains__(self, str name):
        if not self.header or not self.header.entries:
            raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
        cdef Py_ssize_t key_len
        cdef const char* key = PyUnicode_AsUTF8AndSize(name, &key_len)
        return c_bytemap_ex_get_ptr(self.header.entries, key, key_len, NULL, NULL) == bytemap_ret_code.BYTEMAP_OK

    cpdef void add_algo(self, str name, object handler):
        self.classes[name] = handler

    cpdef object cast(self, str name):
        cdef object handler = self.classes.get(name)
        if handler is None:
            raise ValueError(f'Invalid name {name}')
        return handler

    property names:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef bytemap_entry* entry = self.header.entries.first
            while entry:
                yield PyUnicode_FromStringAndSize(entry.key, entry.key_length)
                entry = entry.next

    property reversed_registry:
        def __get__(self):
            return {handler.__name__: name for name, handler in self.classes.items()}


cdef class AlgoEngine(CCPType):
    def __init__(self):
        self.header = c_dma_algo_engine_new(MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True
        self.ccp_bind(&self.header)
        self.registry = AlgoRegistry.c_from_header(self.header.registry, False)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_dma_algo_engine_free(self.header)
            self.header = NULL

    cdef void __ccp_dealloc__(self):
        self.registry = None
        self.orders_map = None
        self.working_orders_map = None

    def __repr__(self):
        if not self.header:
            return f'<{self.__class__.__name__} Unbound>'
        return f'<{self.__class__.__name__}>{self.registry!r}>'

    # === Cython Internal Binding ===

    cdef int c_on_order(self, const md_trade_instruction* order):
        cdef int ret_code = c_dma_algo_engine_on_order(self.header, order)

        if ret_code == dma_algo_ret_code.DMA_ALGO_OK:
            return 1
        elif ret_code == dma_algo_ret_code.DMA_ALGO_ERR_NOT_FOUND:
            return 0
        elif ret_code == dma_algo_ret_code.DMA_ALGO_ERR_INVALID_INPUT:
            raise ValueError('The engine requires an order.')
        else:
            raise RuntimeError(f'c_dma_algo_engine_on_order failed with err code: {ret_code}')

    cdef int c_on_trade(self, const md_trade_report* trade):
        cdef int ret_code = c_dma_algo_engine_on_trade(self.header, trade)

        if ret_code == dma_algo_ret_code.DMA_ALGO_OK:
            return 1
        elif ret_code == dma_algo_ret_code.DMA_ALGO_ERR_NOT_FOUND:
            return 0
        elif ret_code == dma_algo_ret_code.DMA_ALGO_ERR_INVALID_INPUT:
            raise ValueError('The engine rejected the report for the tracked order.')
        else:
            raise RuntimeError(f'c_dma_algo_engine_on_trade failed with err code: {ret_code}')

    cdef void c_on_md(self, const md_variant* market_data):
        c_dma_algo_engine_on_md(self.header, market_data)

    # === Python Interfaces ===

    def on_order(self, TradeInstruction order):
        return self.c_on_order(&order.header.trade_instruction)

    def on_trade(self, TradeReport report):
        return self.c_on_trade(&report.header.trade_report)

    def on_market_data(self, MarketData market_data):
        self.c_on_md(market_data.header)

    # === Python Properties ===

    property orders:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            if self.orders_map is None:
                self.orders_map = OrderMapView.c_from_header(self.header.orders, False)
            return self.orders_map

    property working_orders:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            if self.working_orders_map is None:
                self.working_orders_map = OrderMapView.c_from_header(self.header.working_orders_by_id, False)
            return self.working_orders_map


cdef AlgoEngine ALGO_ENGINE = AlgoEngine()
cdef AlgoRegistry ALGO_REGISTRY = ALGO_ENGINE.registry
globals()['ALGO_REGISTRY'] = ALGO_REGISTRY
globals()['ALGO_ENGINE'] = ALGO_ENGINE
