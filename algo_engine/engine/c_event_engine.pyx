from cpython.unicode cimport PyUnicode_FromString, PyUnicode_FromStringAndSize
from libc.stdint cimport uint8_t, uintptr_t

from event_engine.capi cimport evt_topic, evt_topic_part_variant, evt_topic_type, evt_topic_match, c_topic_match

from ..base.c_market_data cimport _MarketDataVirtualBase, _MarketDataBuffer
from ..base.c_market_data_ng.c_market_data cimport md_variant


cdef class TopicSet:
    def __cinit__(self):
        self.on_order = Topic('on_order')
        self.on_report = Topic('on_report')
        self.eod = Topic('eod')
        self.eod_done = Topic('eod_done')
        self.bod = Topic('bod')
        self.bod_done = Topic('bod_done')

        self.launch_order = Topic('launch_order.{ticker}')
        self.cancel_order = Topic('cancel_order.{ticker}')
        self.realtime = Topic('realtime.{ticker}.{dtype}')

        self.push_topic_map = {}

    cpdef Topic push(self, object market_data):
        cdef uintptr_t data_addr = market_data._data_addr
        cdef _MarketDataBuffer* market_data_ptr = <_MarketDataBuffer*> data_addr
        cdef uint8_t dtype = market_data_ptr.MetaInfo.dtype
        cdef str ticker = PyUnicode_FromString(&market_data_ptr.MetaInfo.ticker[0])
        cdef dict topic_map

        if ticker in self.push_topic_map:
            topic_map = self.push_topic_map[ticker]
        else:
            topic_map = {}
            self.push_topic_map[ticker] = topic_map

        if dtype in topic_map:
            return topic_map[dtype]
        cdef Topic topic = self.realtime.format(ticker=ticker, dtype=_MarketDataVirtualBase.c_dtype_name(dtype))
        topic_map[dtype] = topic
        return topic

    cpdef Topic push_ng(self, MarketData market_data):
        cdef md_variant* market_data_ptr = market_data.header
        cdef uint8_t dtype = <uint8_t> market_data_ptr.meta_info.dtype
        cdef str ticker = PyUnicode_FromString(market_data_ptr.meta_info.ticker)
        cdef dict topic_map

        if ticker in self.push_topic_map:
            topic_map = self.push_topic_map[ticker]
        else:
            topic_map = {}
            self.push_topic_map[ticker] = topic_map

        if dtype in topic_map:
            return topic_map[dtype]
        cdef Topic topic = self.realtime.format(ticker=ticker, dtype=_MarketDataVirtualBase.c_dtype_name(dtype))
        topic_map[dtype] = topic
        return topic

    cpdef dict parse(self, Topic topic):
        cdef evt_topic* topic_ptr = topic.header
        cdef evt_topic_match* match_res = c_topic_match(self.realtime.header, topic_ptr, NULL, 1)

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


cdef EventEngineEx EVENT_ENGINE = EventEngineEx()
cdef TopicSet TOPIC = TopicSet()

globals()['EVENT_ENGINE'] = EVENT_ENGINE
globals()['TOPIC'] = TOPIC
