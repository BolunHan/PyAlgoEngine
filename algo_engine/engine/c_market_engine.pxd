from cpython.object cimport PyObject
from libc.stdint cimport int16_t, uintptr_t
from libcpp cimport bool as c_bool

from cbase.allocator_protocol cimport CCPType, allocator_protocol
from cbase.bytemap cimport BoundByteMap, bytemap

from algo_engine.base.c_market_data.c_internal cimport InternalData
from algo_engine.base.c_market_data.c_market_data cimport MarketData, md_filter_flag, md_internal, md_variant
from algo_engine.exchange_profile.c_exchange_profile cimport ExchangeProfile, session_datetime_t


cdef extern from "Python.h":
    PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, ...)
    PyObject* PyErr_GetRaisedException()
    void PyErr_DisplayException(PyObject* exc)


cdef extern from "algo_engine/engine/c_market_engine.h":
    ctypedef enum mds_ret_code:
        MDS_OK
        MDS_OK_SUBSCRIBED
        MDS_ERR_INVALID_INPUT
        MDS_ERR_UNKNOWN_DTYPE
        MDS_ERR_INTERNAL_DATA
        MDS_ERR_OOM
        MDS_ERR_NOT_FOUND

    ctypedef unsigned char mds_monitor_id[16]

    ctypedef struct mds_monitor:
        const char* name
        mds_monitor_id monitor_id
        md_filter_flag md_filter
        int16_t priority
        c_bool enabled
        c_bool is_ready

    ctypedef struct mds_subscription:
        session_datetime_t last_update
        session_datetime_t last_trade_update
        double last_price
        size_t n_feeds
        c_bool is_subscribed_td
        c_bool is_subscribed_od
        c_bool is_subscribed_tk_lite
        c_bool is_subscribed_tk
        c_bool is_subscribed_cs

    ctypedef void (*mds_monitor_manager_callback_func)(mds_monitor_manager* manager, const md_variant* market_data, void* user_data) noexcept

    ctypedef struct mds_monitor_manager_callback_ctx:
        mds_monitor_manager_callback_func fn
        void* user_data
        uintptr_t id
        mds_monitor_manager_callback_ctx* next

    ctypedef struct mds_monitor_manager:
        bytemap* monitors
        mds_monitor_manager_callback_ctx* callbacks
        c_bool enabled

    ctypedef struct mds:
        mds_monitor_manager* monitor_manager
        bytemap* subscriptions
        bytemap* monitors
        session_datetime_t ts

    mds_monitor* c_mds_monitor_new(const char* name, int16_t priority, md_filter_flag md_filter, allocator_protocol* allocator) noexcept nogil
    int          c_mds_monitor_init(mds_monitor* monitor, const char* name, int16_t priority, md_filter_flag md_filter, allocator_protocol* allocator) noexcept nogil
    void         c_mds_monitor_dealloc(mds_monitor* monitor) noexcept nogil
    void         c_mds_monitor_free(mds_monitor* monitor) noexcept nogil
    void         c_mds_monitor_gen_uid(mds_monitor* monitor, const char* salt, size_t salt_len) noexcept nogil
    c_bool       c_mds_monitor_filter_md(const mds_monitor* monitor, const md_variant* market_data) noexcept nogil
    void         c_mds_monitor_set_enabled(mds_monitor* monitor, c_bool enabled) noexcept nogil
    c_bool       c_mds_monitor_is_enabled(const mds_monitor* monitor) noexcept nogil

    mds_monitor_manager* c_mds_monitor_manager_new(allocator_protocol* allocator) noexcept nogil
    int                  c_mds_monitor_manager_init(mds_monitor_manager* manager, allocator_protocol* allocator) noexcept nogil
    void                 c_mds_monitor_manager_dealloc(mds_monitor_manager* manager) noexcept nogil
    void                 c_mds_monitor_manager_free(mds_monitor_manager* manager) noexcept nogil
    int                  c_mds_monitor_manager_register_callback(mds_monitor_manager* manager, mds_monitor_manager_callback_func callback, void* user_data, uintptr_t* out_id) noexcept nogil
    int                  c_mds_monitor_manager_unregister_callback(mds_monitor_manager* manager, uintptr_t callback_id) noexcept nogil
    void                 c_mds_monitor_manager_feed(mds_monitor_manager* manager, const md_variant* market_data) noexcept nogil
    int                  c_mds_monitor_manager_register_monitor(mds_monitor_manager* manager, mds_monitor* monitor) noexcept nogil
    int                  c_mds_monitor_manager_unregister_monitor(mds_monitor_manager* manager, mds_monitor* monitor) noexcept nogil
    mds_monitor*         c_mds_monitor_manager_get_monitor(const mds_monitor_manager* manager, const mds_monitor_id monitor_id) noexcept nogil
    int                  c_mds_monitor_manager_clear_monitors(mds_monitor_manager* manager) noexcept nogil
    void                 c_mds_monitor_manager_set_enabled(mds_monitor_manager* manager, c_bool enabled) noexcept nogil
    c_bool               c_mds_monitor_manager_is_enabled(const mds_monitor_manager* manager) noexcept nogil

    mds*         c_mds_new(allocator_protocol* allocator) noexcept nogil
    int          c_mds_init(mds* MDS, allocator_protocol* allocator) noexcept nogil
    void         c_mds_dealloc(mds* MDS) noexcept nogil
    void         c_mds_free(mds* MDS) noexcept nogil
    int          c_mds_clear(mds* MDS) noexcept nogil
    int          c_mds_set_manager(mds* MDS, mds_monitor_manager* manager) noexcept nogil
    mds_monitor_manager* c_mds_get_manager(const mds* MDS) noexcept nogil
    int          c_mds_register_monitor(mds* MDS, mds_monitor* monitor) noexcept nogil
    int          c_mds_unregister_monitor(mds* MDS, mds_monitor* monitor) noexcept nogil
    mds_monitor* c_mds_get_monitor(const mds* MDS, const mds_monitor_id monitor_id) noexcept nogil
    size_t       c_mds_n_monitors(const mds* MDS) noexcept nogil
    int          c_mds_update_subscription(mds* MDS, const md_variant* market_data) noexcept nogil
    int          c_mds_on_market_data(mds* MDS, const md_variant* market_data) noexcept nogil
    int          c_mds_on_internal_data(mds* MDS, const md_internal* internal_data) noexcept nogil
    double       c_mds_get_market_price(const mds* MDS, const char* ticker) noexcept nogil


cdef class MarketDataMonitor(CCPType):
    cdef dict __dict__
    cdef mds_monitor* monitor_meta
    cdef bint owner

    cdef readonly str name

    cdef void c_set_monitor_id(self, object monitor_id)

    @staticmethod
    cdef MarketDataMonitor c_from_header(mds_monitor* header, bint owner)

    cpdef void clear(self)


cdef class MarketDataMonitorMap(BoundByteMap):
    cdef bytes ws_key
    cdef void* ws_value

    @staticmethod
    cdef MarketDataMonitorMap c_from_header(bytemap* header, bint owner=?)

    cdef const char* c_serialize_key(self, object obj, size_t* key_len)

    cdef object c_deserialize_key(self, const char* key, size_t key_len)

    cdef const char* c_serialize_value(self, object obj, size_t* value_len)

    cdef object c_deserialize_value(self, const char* value, size_t value_len)


cdef class MonitorManager(CCPType):
    cdef mds_monitor_manager* header
    cdef bint owner
    cdef uintptr_t callback_id

    cdef readonly MarketDataMonitorMap monitor

    cdef mds_monitor* c_get_monitor(self, object monitor_id)

    @staticmethod
    cdef void c_callback_adaptor(mds_monitor_manager* manager, const md_variant* market_data, void* user_data) noexcept with gil

    cpdef void feed_monitor(self, object monitor_id, MarketData market_data)

    cdef void c_on_market_data(self, const md_variant* market_data)

    cpdef void on_market_data(self, MarketData market_data)

    cpdef void add_monitor(self, object monitor)

    cpdef void pop_monitor(self, object monitor_id)

    cpdef void clear_monitors(self)

    cpdef void start(self)

    cpdef void stop(self)

    cpdef void clear(self)

    cpdef dict get_values(self)


cdef class MarketDataService(CCPType):
    cdef dict __dict__
    cdef mds* header
    cdef bint owner

    cdef readonly ExchangeProfile profile
    cdef readonly MonitorManager monitor_manager
    cdef readonly MarketDataMonitorMap monitor

    cdef void c_on_internal_data(self, const md_internal* md)

    cdef void c_on_market_data(self, const md_variant* md)

    cpdef void on_internal_data(self, InternalData internal_data)

    cpdef void on_market_data(self, MarketData market_data)

    cpdef double get_market_price(self, str ticker)

    cdef double c_get_market_price(self, const char* ticker)

    cpdef void set_manager(self, MonitorManager manager)

    cpdef void add_monitor(self, object monitor)

    cpdef void pop_monitor(self, object monitor=?, object monitor_id=?, str monitor_name=?)

    cpdef void clear(self)


cdef MarketDataService MDS
