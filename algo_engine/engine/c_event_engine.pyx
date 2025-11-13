from cpython.unicode cimport PyUnicode_FromString, PyUnicode_FromStringAndSize
from libc.stdint cimport uint8_t, uintptr_t

from event_engine.capi cimport Topic, TopicPart, TopicType, EventEngineEx, TopicPartMatchResult, c_topic_match

from ..base.c_market_data cimport _MarketDataVirtualBase, _MarketDataBuffer


cdef class TopicSet:
    def __cinit__(self):
        self.on_order = PyTopic('on_order')
        self.on_report = PyTopic('on_report')
        self.eod = PyTopic('eod')
        self.eod_done = PyTopic('eod_done')
        self.bod = PyTopic('bod')
        self.bod_done = PyTopic('bod_done')

        self.launch_order = PyTopic('launch_order.{ticker}')
        self.cancel_order = PyTopic('cancel_order.{ticker}')
        self.realtime = PyTopic('realtime.{ticker}.{dtype}')

        self.push_topic_map = {}

    cpdef PyTopic push(self, object market_data):
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
        cdef PyTopic topic = self.realtime.format(ticker=ticker, dtype=_MarketDataVirtualBase.c_dtype_name(dtype))
        topic_map[dtype] = topic
        return topic

    cpdef dict parse(self, PyTopic topic):
        cdef Topic* topic_ptr = topic.header
        cdef TopicPartMatchResult* match_res = c_topic_match(self.realtime.header, topic_ptr, NULL)

        if not topic.is_exact:
            raise ValueError(f'Topic {topic} is not an exact topic.')

        cdef dict out = {}
        cdef TopicPartMatchResult* node = match_res
        cdef str literal
        cdef TopicPart* part

        while node:
            if not node.matched:
                raise ValueError(f'Topic {topic} not match with {self.realtime}')
            part = node.part_a
            if part.header.ttype == TopicType.TOPIC_PART_EXACT:
                node = node.next
                continue
            literal = PyUnicode_FromStringAndSize(part.exact.part, part.exact.part_len)
            part = node.part_b
            out[literal] = PyUnicode_FromStringAndSize(part.exact.part, part.exact.part_len)
            node = node.next
        return out


EVENT_ENGINE = EventEngineEx()
TOPIC = TopicSet()