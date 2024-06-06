from types import SimpleNamespace

import event_engine
from event_engine import Topic, PatternTopic, EventEngine

from . import LOGGER

__all__ = ['EVENT_ENGINE', 'TOPIC']

event_engine.set_logger(LOGGER.getChild('EventEngine'))


class TopicSet(object):
    on_order = Topic('on_order')
    on_report = Topic('on_report')
    eod = Topic('eod')
    eod_done = Topic('eod_done')
    bod = Topic('bod')
    bod_done = Topic('bod_done')

    launch_order = PatternTopic('launch_order.{ticker}')
    cancel_order = PatternTopic('cancel_order.{ticker}')
    realtime = PatternTopic('realtime.{ticker}.{dtype}')

    @classmethod
    def push(cls, market_data):
        return cls.realtime(ticker=market_data.ticker, dtype=market_data.__class__.__name__)

    @classmethod
    def parse(cls, topic: Topic) -> SimpleNamespace:
        try:
            _ = topic.value.split('.')

            action = _.pop(0)
            if action in ['open', 'close']:
                dtype = None
            else:
                dtype = _.pop(-1)
            ticker = '.'.join(_)

            p = SimpleNamespace(
                action=action,
                dtype=dtype,
                ticker=ticker
            )
            return p
        except Exception as _:
            raise ValueError(f'Invalid topic {topic}')


EVENT_ENGINE = EventEngine()
TOPIC = TopicSet
# EVENT_ENGINE.start()
