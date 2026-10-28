from libcpp cimport bool as c_bool

from cbase.allocator_protocol.c_allocator_protocol cimport allocator_protocol
from event_engine.capi cimport evt_topic, Topic, EventEngineEx

from algo_engine.base cimport md_data_type, md_variant, MarketData


cdef extern from "algo_engine/engine/c_event_engine.h":
    ctypedef enum evt_topic_variant:
        EVT_TOPIC_VARIANT_C
        EVT_TOPIC_VARIANT_PY
        EVT_TOPIC_VARIANT_COUNT
        EVT_TOPIC_VARIANT_MASK

    ctypedef enum topic_set_ret_code:
        TOPIC_SET_OK
        TOPIC_SET_ERR_INVALID_INPUT
        TOPIC_SET_ERR_INVALID_TOPIC
        TOPIC_SET_ERR_UNKNOWN_DTYPE
        TOPIC_SET_ERR_OOM

    const char* TOPIC_SET_ON_ORDER
    const char* TOPIC_SET_ON_REPORT
    const char* TOPIC_SET_EOD
    const char* TOPIC_SET_EOD_DONE
    const char* TOPIC_SET_BOD
    const char* TOPIC_SET_BOD_DONE
    const char* TOPIC_SET_LAUNCH_ORDER_PATTERN
    const char* TOPIC_SET_CANCEL_ORDER_PATTERN
    const char* TOPIC_SET_REALTIME
    const char* TOPIC_SET_REALTIME_PATTERN

    ctypedef struct evt_topic_set:
        pass

    const char*    c_topic_set_variant_suffix(evt_topic_variant variant) noexcept nogil
    md_data_type   c_topic_set_dtype_from_name(const char* name, size_t name_len) noexcept nogil

    int            c_topic_set_init(evt_topic_set* topic_set, allocator_protocol* allocator) noexcept nogil
    void           c_topic_set_dealloc(evt_topic_set* topic_set) noexcept nogil
    evt_topic_set* c_topic_set_new(allocator_protocol* allocator) noexcept nogil
    void           c_topic_set_free(evt_topic_set* topic_set) noexcept nogil
    evt_topic_set* c_topic_set_union(const evt_topic_set* left, const evt_topic_set* right, allocator_protocol* allocator) noexcept nogil

    c_bool         c_topic_set_contains(const evt_topic_set* topic_set, const char* topic_literal, size_t topic_len) noexcept nogil
    c_bool         c_topic_set_contains_topic(const evt_topic_set* topic_set, const evt_topic* topic) noexcept nogil
    evt_topic*     c_topic_set_get(evt_topic_set* topic_set, const char* topic_literal, size_t topic_len) noexcept nogil
    evt_topic*     c_topic_set_register(evt_topic_set* topic_set, evt_topic* topic) noexcept nogil

    evt_topic*     c_md_topic_push(evt_topic_set* topic_set, const md_variant* market_data) noexcept nogil
    int            c_md_topic_parse(const evt_topic_set* topic_set, const evt_topic* topic, const char** out_ticker, md_data_type* out_dtype) noexcept nogil


cdef class TopicSet:
    cdef dict __dict__

    cdef readonly Topic on_order
    cdef readonly Topic on_report
    cdef readonly Topic eod
    cdef readonly Topic eod_done
    cdef readonly Topic bod
    cdef readonly Topic bod_done
    cdef readonly Topic launch_order
    cdef readonly Topic cancel_order
    cdef readonly Topic realtime
    cdef readonly dict push_topic_map

    cpdef Topic push(self, MarketData market_data)

    cpdef dict parse(self, Topic topic)


cdef class TopicSetEx:
    cdef dict __dict__

    cdef evt_topic_set* header
    cdef readonly bint owner

    cdef readonly Topic on_order
    cdef readonly Topic on_report
    cdef readonly Topic eod
    cdef readonly Topic eod_done
    cdef readonly Topic bod
    cdef readonly Topic bod_done
    cdef readonly Topic launch_order
    cdef readonly Topic cancel_order
    cdef readonly Topic realtime

    @staticmethod
    cdef TopicSetEx c_from_header(evt_topic_set* header)

    cdef Topic c_push(self, MarketData market_data)
    cdef tuple c_parse(self, Topic topic)
    cdef Topic c_get(self, const char* topic_literal)
    cdef bint c_contains(self, str name)
    cdef Topic c_lookup(self, const char* topic_literal, bint required)
    cdef void c_bind_builtins(self, bint required)

    cpdef Topic push(self, MarketData market_data)
    cpdef tuple parse(self, Topic topic)
    cpdef dict parse_map(self, Topic topic)
    cpdef Topic register(self, str name)
    cpdef Topic get(self, str name)


cdef evt_topic_set* C_TOPIC_SET
cdef TopicSetEx TOPIC_SET
cdef EventEngineEx EVENT_ENGINE
