import traceback

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.ref cimport Py_XDECREF, Py_XINCREF
from libc.stdint cimport uintptr_t
from libc.string cimport memcmp

from cbase.allocator_protocol cimport ap_ret_code, c_ap_protocol_from_ptr, c_ap_register_callback
from event_engine.capi cimport DEFAULT_MQ_SPIN_LIMIT, EventEngineEx, evt_topic

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR, MD_SHM_ALLOCATOR
from algo_engine.base.c_intern_string cimport C_POOL
from algo_engine.base.c_market_data.c_market_data_buffer cimport c_md_copy_to_shm

from . import LOGGER
from .c_event_engine cimport EVENT_ENGINE, TOPIC_SET, TopicSetEx


cdef class DMAProtocol:
    def __init__(self, event_engine=None, topic_set=None):
        cdef EventEngineEx engine = EVENT_ENGINE if event_engine is None else event_engine
        cdef TopicSetEx topics = TOPIC_SET if topic_set is None else topic_set

        self.header = c_dma_protocol_new(engine.header, topics.header, MD_SHM_ALLOCATOR, C_POOL, MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True
        self.event_engine = engine
        self.topic_set = topics

    def __dealloc__(self):
        if not self.header:
            return

        cdef dma_protocol_pycallback_ctx* ctx = self.callbacks
        cdef dma_protocol_pycallback_ctx* next

        while ctx:
            next = ctx.next
            c_dma_protocol_unregister_callback(self.header, ctx.topic, DMAProtocol.c_callback_adaptor, <void*> ctx)
            Py_XDECREF(ctx.py_callback_fn)
            PyMem_Free(ctx)
            ctx = next

        if self.owner:
            c_dma_protocol_free(self.header)

    # === Cython Internal Binding ===

    @staticmethod
    cdef void c_callback_adaptor(const md_variant* market_data, const evt_topic* topic, void* user_data) noexcept with gil:
        cdef dma_protocol_pycallback_ctx* ctx = <dma_protocol_pycallback_ctx*> user_data
        cdef DMAProtocol self = <DMAProtocol> <object> ctx.self
        cdef MarketData py_md = self.c_construct_md(market_data)
        cdef object callback = <object> ctx.py_callback_fn

        try:
            callback(py_md)
        except Exception:
            LOGGER.error(traceback.format_exc())

    @staticmethod
    cdef void c_detach_md(ap_callback_event event, void* buf, void* user_data) noexcept with gil:
        if event != ap_callback_event.AP_CALLBACK_EVENT_FREE:
            return
        cdef MarketData instance = <MarketData> <object> user_data
        instance.header = NULL
        instance.data_addr = 0

    @staticmethod
    cdef inline bint c_topic_key_equal(const evt_topic* left, const evt_topic* right) noexcept nogil:
        if not left or not right:
            return False
        if left.key_len != right.key_len:
            return False
        if left.key_len == 0:
            return left.key == right.key
        return memcmp(left.key, right.key, left.key_len) == 0

    cdef MarketData c_construct_md(self, const md_variant* market_data):
        if <void*> market_data == <void*> self.ws_md:
            return self.ws_instance

        cdef MarketData instance = MarketData.c_from_header(market_data, False)
        cdef allocator_protocol* allocator = c_ap_protocol_from_ptr(market_data)
        cdef int ret_code = c_ap_register_callback(allocator, DMAProtocol.c_detach_md, <void*> <PyObject*> instance, NULL)
        if ret_code != ap_ret_code.AP_OK:
            LOGGER.error(f'DMAProtocol failed to register the MarketData nullifier, code: {ret_code}')

        self.ws_md = market_data
        self.ws_instance = instance
        return instance

    cpdef int send(self, MarketData market_data):
        self.ws_md = c_md_copy_to_shm(market_data.header, MD_SHM_ALLOCATOR, C_POOL)
        self.ws_instance = market_data

        if not self.ws_md:
            return dma_protocol_ret_code.DMA_PROTOCOL_ERR_NO_SHM

        cdef int ret_code = c_dma_protocol_send_nocopy(self.header, <md_variant*> self.ws_md)

        if ret_code != dma_protocol_ret_code.DMA_PROTOCOL_OK:
            self.ws_md = NULL
            self.ws_instance = None

        return ret_code

    cpdef int register_callback(self, object callback, Topic topic):
        if not callable(callback):
            raise TypeError(f'callback must be callable, got {type(callback).__name__}')

        cdef evt_topic* topic_ptr = topic.header

        cdef dma_protocol_pycallback_ctx* node = self.callbacks
        while node:
            if DMAProtocol.c_topic_key_equal(node.topic, topic_ptr) and (<object> node.py_callback_fn) == callback:
                return dma_protocol_ret_code.DMA_PROTOCOL_DUPLICATED
            node = node.next

        cdef dma_protocol_pycallback_ctx* ctx = <dma_protocol_pycallback_ctx*> PyMem_Malloc(sizeof(dma_protocol_pycallback_ctx))
        if not ctx:
            raise MemoryError(f'Failed to allocate a callback context for <{self.__class__.__name__}>.')

        ctx.self = <PyObject*> self
        ctx.py_callback_fn = <PyObject*> callback
        ctx.topic = topic_ptr
        ctx.prev = NULL
        ctx.next = self.callbacks
        Py_XINCREF(ctx.py_callback_fn)

        cdef int ret_code = c_dma_protocol_register_callback(self.header, topic_ptr, DMAProtocol.c_callback_adaptor, <void*> ctx)

        if ret_code != dma_protocol_ret_code.DMA_PROTOCOL_OK:
            Py_XDECREF(ctx.py_callback_fn)
            PyMem_Free(ctx)

            if ret_code == dma_protocol_ret_code.DMA_PROTOCOL_ERR_OOM:
                raise MemoryError('Failed to register the callback with the event engine.')
            elif ret_code == dma_protocol_ret_code.DMA_PROTOCOL_ERR_INVALID_INPUT:
                raise ValueError('The protocol, its engine, the topic, or the callback is not initialized.')
            else:
                raise RuntimeError(f'c_dma_protocol_register_callback failed with err code: {ret_code}')

        if self.callbacks:
            self.callbacks.prev = ctx
        self.callbacks = ctx

        return 0

    cpdef int unregister_callback(self, object callback, Topic topic):
        cdef evt_topic* topic_ptr = (<Topic> topic).header
        cdef dma_protocol_pycallback_ctx* ctx = self.callbacks
        cdef int ret_code

        while ctx:
            if DMAProtocol.c_topic_key_equal(ctx.topic, topic_ptr) and (<object> ctx.py_callback_fn) == callback:
                break
            ctx = ctx.next

        if not ctx:
            return dma_protocol_ret_code.DMA_PROTOCOL_ERR_NOT_FOUND

        ret_code = c_dma_protocol_unregister_callback(self.header, ctx.topic, DMAProtocol.c_callback_adaptor, <void*> ctx)

        if ret_code == dma_protocol_ret_code.DMA_PROTOCOL_OK or ret_code == dma_protocol_ret_code.DMA_PROTOCOL_ERR_NOT_FOUND:
            if ctx.prev:
                ctx.prev.next = ctx.next
            else:
                self.callbacks = ctx.next
            if ctx.next:
                ctx.next.prev = ctx.prev

            Py_XDECREF(ctx.py_callback_fn)
            PyMem_Free(ctx)
            return dma_protocol_ret_code.DMA_PROTOCOL_OK
        elif ret_code == dma_protocol_ret_code.DMA_PROTOCOL_ERR_INVALID_INPUT:
            raise ValueError('The protocol, the topic, or the callback is not initialized.')
        else:
            raise RuntimeError(f'c_dma_protocol_unregister_callback failed with err code: {ret_code}')

    cpdef void config(self, bint block, size_t max_spin=DEFAULT_MQ_SPIN_LIMIT, double timeout=0.0):
        cdef int ret_code = c_dma_protocol_config(self.header, block, max_spin, timeout)

        if ret_code == dma_protocol_ret_code.DMA_PROTOCOL_OK:
            return
        elif ret_code == dma_protocol_ret_code.DMA_PROTOCOL_ERR_INVALID_INPUT:
            raise ValueError(f'<{self.__class__.__name__}> is not initialized.')
        else:
            raise RuntimeError(f'c_dma_protocol_config failed with err code: {ret_code}')

    # === Python Properties ===

    property address:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return <uintptr_t> self.header


cdef DMAProtocol DMA_PROTOCOL = DMAProtocol()
globals()['DMA_PROTOCOL'] = DMA_PROTOCOL
