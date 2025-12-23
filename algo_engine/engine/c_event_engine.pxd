from event_engine.capi cimport Topic, EventEngineEx

from ..base.c_market_data_ng.c_market_data cimport MarketData


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

    cpdef Topic push(self, object market_data)

    cpdef Topic push_ng(self, MarketData market_data)

    cpdef dict parse(self, Topic topic)

cdef EventEngineEx EVENT_ENGINE
cdef TopicSet TOPIC
