from cpython.object cimport PyObject
from libc.stdint cimport uint32_t
from libcpp cimport bool as c_bool

from cbase.allocator_protocol cimport allocator_protocol, ap_callback_event
from cbase.intern_string cimport istr_map
from event_engine.capi cimport EventEngineEx, Topic, evt_engine, evt_message_payload, evt_topic

from algo_engine.base.c_market_data.c_market_data cimport MarketData, md_trade_instruction, md_variant

from .c_event_engine cimport TopicSetEx, evt_topic_set


cdef extern from "algo_engine/engine/c_dma_protocol.h":
    ctypedef enum dma_protocol_ret_code:
        DMA_PROTOCOL_OK
        DMA_PROTOCOL_ERR_INVALID_INPUT
        DMA_PROTOCOL_ERR_OOM
        DMA_PROTOCOL_ERR_NO_SHM
        DMA_PROTOCOL_ERR_NO_TOPIC
        DMA_PROTOCOL_ERR_PUBLISH
        DMA_PROTOCOL_ERR_ENGINE
        DMA_PROTOCOL_ERR_NOT_FOUND
        DMA_PROTOCOL_ERR_BUSY
        DMA_PROTOCOL_DUPLICATED

    ctypedef void (*dma_protocol_callback_func)(const md_variant* market_data, const evt_topic* topic, void* user_data) noexcept

    ctypedef struct dma_protocol_ctx:
        evt_engine* ENGINE
        evt_topic_set* TOPICS
        allocator_protocol* SHM_ALLOCATOR
        istr_map* SHM_POOL
        c_bool block
        size_t max_spin
        double timeout

    dma_protocol_ctx* c_dma_protocol_new(evt_engine* engine, evt_topic_set* topics, allocator_protocol* shm_allocator, istr_map* shm_pool, allocator_protocol* allocator) noexcept nogil
    int c_dma_protocol_init(dma_protocol_ctx* protocol, evt_engine* engine, evt_topic_set* topics, allocator_protocol* shm_allocator, istr_map* shm_pool) noexcept nogil
    void c_dma_protocol_dealloc(dma_protocol_ctx* protocol) noexcept nogil
    void c_dma_protocol_free(dma_protocol_ctx* protocol) noexcept nogil

    void c_dma_protocol_cleanup(evt_message_payload* payload) noexcept nogil

    int c_dma_protocol_config(dma_protocol_ctx* protocol, c_bool block, size_t max_spin, double timeout) noexcept nogil
    int c_dma_protocol_send(dma_protocol_ctx* protocol, md_variant* market_data) noexcept nogil
    int c_dma_protocol_send_nocopy(dma_protocol_ctx* protocol, md_variant* market_data) noexcept nogil
    int c_dma_protocol_register_callback(dma_protocol_ctx* protocol, evt_topic* topic, dma_protocol_callback_func fn, void* user_data) noexcept nogil
    int c_dma_protocol_unregister_callback(dma_protocol_ctx* protocol, evt_topic* topic, dma_protocol_callback_func fn, void* user_data) noexcept nogil
    void c_dma_protocol_launch_order(md_trade_instruction* order, uint32_t seq_id, void* user_data) noexcept nogil
    void c_dma_protocol_cancel_order(const md_trade_instruction* order, uint32_t seq_id, void* user_data) noexcept nogil


ctypedef struct dma_protocol_pycallback_ctx:
    PyObject* self
    PyObject* py_callback_fn
    evt_topic* topic
    dma_protocol_pycallback_ctx* prev
    dma_protocol_pycallback_ctx* next


cdef class DMAProtocol:
    cdef dma_protocol_ctx* header
    cdef bint owner

    cdef EventEngineEx event_engine
    cdef TopicSetEx topic_set
    cdef dma_protocol_pycallback_ctx* callbacks

    cdef evt_message_payload ws_payload
    cdef const md_variant* ws_md
    cdef MarketData ws_instance

    @staticmethod
    cdef void c_callback_adaptor(const md_variant* market_data, const evt_topic* topic, void* user_data) noexcept with gil

    @staticmethod
    cdef void c_detach_md(ap_callback_event event, void* buf, void* user_data) noexcept with gil

    cdef MarketData c_construct_md(self, const md_variant* market_data)

    cpdef int send(self, MarketData market_data)

    cpdef int register_callback(self, object callback, Topic topic)

    cpdef int unregister_callback(self, object callback, Topic topic)

    cpdef void config(self, bint block, size_t max_spin=?, double timeout=?)
