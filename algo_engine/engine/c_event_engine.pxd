from event_engine.capi cimport PyTopic


cdef class TopicSet:
    cdef dict __dict__

    cdef readonly PyTopic on_order
    cdef readonly PyTopic on_report
    cdef readonly PyTopic eod
    cdef readonly PyTopic eod_done
    cdef readonly PyTopic bod
    cdef readonly PyTopic bod_done
    cdef readonly PyTopic launch_order
    cdef readonly PyTopic cancel_order
    cdef readonly PyTopic realtime
    cdef readonly dict push_topic_map

    cpdef PyTopic push(self, object market_data)

    cpdef dict parse(self, PyTopic topic)
