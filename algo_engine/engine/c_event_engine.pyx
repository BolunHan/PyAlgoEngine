from cpython.unicode cimport PyUnicode_AsUTF8AndSize, PyUnicode_FromString, PyUnicode_FromStringAndSize
from libc.stdint cimport uintptr_t
from libc.string cimport strlen

from event_engine.base cimport EE_DEFAULT_ALLOCATOR
from event_engine.capi cimport Topic, EventEngineEx, evt_topic, evt_topic_match, evt_topic_part_variant, evt_topic_type, c_topic_match

from algo_engine.base.c_market_data.c_market_data cimport MarketData, c_md_dtype_name, md_data_type, md_variant


cdef class TopicSet:
    def __cinit__(self):
        if not C_TOPIC_SET:
            raise RuntimeError('The C topic set must exist before a legacy TopicSet.')

        self.on_order = Topic.c_from_header(c_topic_set_get(C_TOPIC_SET, TOPIC_SET_ON_ORDER, strlen(TOPIC_SET_ON_ORDER)), False)
        self.on_report = Topic.c_from_header(c_topic_set_get(C_TOPIC_SET, TOPIC_SET_ON_REPORT, strlen(TOPIC_SET_ON_REPORT)), False)
        self.eod = Topic.c_from_header(c_topic_set_get(C_TOPIC_SET, TOPIC_SET_EOD, strlen(TOPIC_SET_EOD)), False)
        self.eod_done = Topic.c_from_header(c_topic_set_get(C_TOPIC_SET, TOPIC_SET_EOD_DONE, strlen(TOPIC_SET_EOD_DONE)), False)
        self.bod = Topic.c_from_header(c_topic_set_get(C_TOPIC_SET, TOPIC_SET_BOD, strlen(TOPIC_SET_BOD)), False)
        self.bod_done = Topic.c_from_header(c_topic_set_get(C_TOPIC_SET, TOPIC_SET_BOD_DONE, strlen(TOPIC_SET_BOD_DONE)), False)
        self.launch_order = Topic.c_from_header(c_topic_set_get(C_TOPIC_SET, TOPIC_SET_LAUNCH_ORDER_PATTERN, strlen(TOPIC_SET_LAUNCH_ORDER_PATTERN)), False)
        self.cancel_order = Topic.c_from_header(c_topic_set_get(C_TOPIC_SET, TOPIC_SET_CANCEL_ORDER_PATTERN, strlen(TOPIC_SET_CANCEL_ORDER_PATTERN)), False)
        self.realtime = Topic.c_from_header(c_topic_set_get(C_TOPIC_SET, TOPIC_SET_REALTIME_PATTERN, strlen(TOPIC_SET_REALTIME_PATTERN)), False)

        self.push_topic_map = {}

    cpdef Topic push(self, MarketData market_data):
        cdef const md_variant* market_data_ptr = market_data.header
        cdef md_data_type dtype = market_data_ptr.meta_info.dtype
        cdef const char* dtype_name = c_md_dtype_name(dtype)
        cdef str ticker = PyUnicode_FromString(market_data_ptr.meta_info.ticker)
        cdef dict topic_map
        cdef Topic topic

        if not dtype_name:
            raise ValueError(f'Unknown dtype {dtype} for ticker {ticker}')

        if ticker in self.push_topic_map:
            topic_map = self.push_topic_map[ticker]
        else:
            topic_map = {}
            self.push_topic_map[ticker] = topic_map

        if dtype in topic_map:
            return topic_map[dtype]

        topic = self.realtime.format(ticker=ticker, dtype=PyUnicode_FromString(dtype_name))
        topic_map[dtype] = topic
        return topic

    cpdef dict parse(self, Topic topic):
        cdef evt_topic* topic_ptr = topic.header
        cdef evt_topic_match* match_res = c_topic_match(self.realtime.header, topic_ptr, NULL)

        if not topic.is_exact:
            raise ValueError(f'Topic {topic} is not an exact topic.')

        cdef dict out = {}
        cdef evt_topic_match* node = match_res
        cdef str literal
        cdef evt_topic_part_variant* part

        while node:
            if not node.matched:
                raise ValueError(f'Topic {topic} not match with {self.realtime}')
            part = node.part_a
            if part.header.ttype == evt_topic_type.TOPIC_PART_EXACT:
                node = node.next
                continue
            literal = PyUnicode_FromStringAndSize(part.exact.part, part.exact.part_len)
            part = node.part_b
            out[literal] = PyUnicode_FromStringAndSize(part.exact.part, part.exact.part_len)
            node = node.next
        return out


cdef class TopicSetEx:
    def __cinit__(self):
        self.push_topic_cache = {}

    def __init__(self):
        self.header = c_topic_set_new(EE_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError('Failed to allocate the C topic set.')

        self.owner = True
        self.c_bind_builtins(True)

    def __dealloc__(self):
        if self.owner and self.header:
            c_topic_set_free(self.header)
            self.header = NULL

    @staticmethod
    cdef TopicSetEx c_from_header(evt_topic_set* header):
        cdef TopicSetEx instance = TopicSetEx.__new__(TopicSetEx)
        instance.header = header
        instance.owner = True
        instance.c_bind_builtins(False)
        return instance

    cdef Topic c_lookup(self, const char* topic_literal, bint required):
        cdef size_t topic_len = strlen(topic_literal)

        if not c_topic_set_contains(self.header, topic_literal, topic_len):
            if required:
                raise KeyError(PyUnicode_FromString(topic_literal))
            return None

        cdef evt_topic* topic = c_topic_set_get(self.header, topic_literal, topic_len)
        if not topic:
            if required:
                raise MemoryError(f'Failed to resolve the topic {topic_literal!r}.')
            return None

        return Topic.c_from_header(topic, False)

    cdef void c_bind_builtins(self, bint required):
        self.on_order = self.c_lookup(TOPIC_SET_ON_ORDER, required)
        self.on_report = self.c_lookup(TOPIC_SET_ON_REPORT, required)
        self.eod = self.c_lookup(TOPIC_SET_EOD, required)
        self.eod_done = self.c_lookup(TOPIC_SET_EOD_DONE, required)
        self.bod = self.c_lookup(TOPIC_SET_BOD, required)
        self.bod_done = self.c_lookup(TOPIC_SET_BOD_DONE, required)
        self.launch_order = self.c_lookup(TOPIC_SET_LAUNCH_ORDER_PATTERN, required)
        self.cancel_order = self.c_lookup(TOPIC_SET_CANCEL_ORDER_PATTERN, required)
        self.realtime = self.c_lookup(TOPIC_SET_REALTIME_PATTERN, required)

    cdef Topic c_push(self, MarketData market_data):
        cdef evt_topic* topic = c_md_topic_push(self.header, market_data.header)

        if not topic:
            raise ValueError('Failed to resolve the market data topic.')

        return Topic.c_from_header(topic, False)

    cpdef Topic push(self, MarketData market_data):
        cdef evt_topic* topic = c_md_topic_push(self.header, market_data.header)
        if not topic:
            raise ValueError('Failed to resolve the market data topic.')

        cdef Topic view = self.push_topic_cache.get(<uintptr_t> topic)
        if view is None:
            view = Topic.c_from_header(topic, False)
            self.push_topic_cache[<uintptr_t> topic] = view
        return view

    cdef tuple c_parse(self, Topic topic):
        cdef const char* out_ticker = NULL
        cdef const char* dtype_name = NULL
        cdef md_data_type out_dtype = <md_data_type> 0
        cdef int ret_code = c_md_topic_parse(self.header, topic.header, &out_ticker, &out_dtype)

        if ret_code != topic_set_ret_code.TOPIC_SET_OK:
            if ret_code == topic_set_ret_code.TOPIC_SET_ERR_INVALID_TOPIC:
                raise ValueError(f'Topic {topic!r} is not a realtime topic.')
            elif ret_code == topic_set_ret_code.TOPIC_SET_ERR_UNKNOWN_DTYPE:
                raise ValueError(f'Topic {topic!r} carries an unknown dtype.')
            elif ret_code == topic_set_ret_code.TOPIC_SET_ERR_INVALID_INPUT:
                raise TypeError('Topic is not initialized.')
            raise RuntimeError(f'Failed to parse the topic, code: {ret_code}')

        if not out_ticker:
            raise RuntimeError('Topic parse returned no ticker.')

        dtype_name = c_md_dtype_name(out_dtype)
        if not dtype_name:
            raise RuntimeError('Unknown dtype code.')

        return PyUnicode_FromString(out_ticker), PyUnicode_FromString(dtype_name)

    cpdef tuple parse(self, Topic topic):
        return self.c_parse(topic)

    cpdef dict parse_map(self, Topic topic):
        cdef tuple parsed = self.c_parse(topic)
        return {'ticker': parsed[0], 'dtype': parsed[1]}

    cdef Topic c_get(self, const char* topic_literal):
        cdef size_t topic_len = strlen(topic_literal)
        cdef evt_topic* topic = c_topic_set_get(self.header, topic_literal, topic_len)

        if not topic:
            if not c_topic_set_contains(self.header, topic_literal, topic_len):
                raise KeyError(PyUnicode_FromString(topic_literal))
            raise MemoryError(f'Failed to resolve the topic {topic_literal!r}.')

        return Topic.c_from_header(topic, False)

    cpdef Topic get(self, str name):
        cdef Py_ssize_t name_len = 0
        cdef const char* name_ptr = PyUnicode_AsUTF8AndSize(name, &name_len)
        return self.c_get(name_ptr)

    cpdef Topic register(self, str name):
        cdef Py_ssize_t name_len = 0
        cdef const char* name_ptr = PyUnicode_AsUTF8AndSize(name, &name_len)
        return self.c_get(name_ptr)

    cdef bint c_contains(self, str name):
        cdef Py_ssize_t name_len = 0
        cdef const char* name_ptr = PyUnicode_AsUTF8AndSize(name, &name_len)
        return c_topic_set_contains(self.header, name_ptr, <size_t> name_len)

    def __getitem__(self, name):
        return self.get(name)

    def __contains__(self, name):
        if not isinstance(name, str):
            return False
        return self.c_contains(name)

    def __getattr__(self, name):
        cdef Py_ssize_t name_len = 0
        cdef const char* name_ptr
        cdef size_t topic_len
        cdef evt_topic* topic

        if not isinstance(name, str):
            raise AttributeError(name)

        name_ptr = PyUnicode_AsUTF8AndSize(name, &name_len)
        topic_len = <size_t> name_len

        if not self.header or not c_topic_set_contains(self.header, name_ptr, topic_len):
            raise AttributeError(name)

        topic = c_topic_set_get(self.header, name_ptr, topic_len)
        if not topic:
            raise AttributeError(name)

        return Topic.c_from_header(topic, False)

    def __or__(self, other):
        cdef evt_topic_set* header

        if not isinstance(other, TopicSetEx):
            return NotImplemented

        header = c_topic_set_union(self.header, (<TopicSetEx> other).header, EE_DEFAULT_ALLOCATOR)
        if not header:
            raise MemoryError('Failed to merge the topic sets.')

        return TopicSetEx.c_from_header(header)

    property address:
        def __get__(self):
            if not self.header:
                raise RuntimeError('Topic set is not initialized.')
            return <uintptr_t> self.header


cdef EventEngineEx EVENT_ENGINE = EventEngineEx()
cdef TopicSetEx TOPIC_SET = TopicSetEx()
cdef evt_topic_set* C_TOPIC_SET = TOPIC_SET.header

globals()['EVENT_ENGINE'] = EVENT_ENGINE
globals()['TOPIC_SET'] = TOPIC_SET
