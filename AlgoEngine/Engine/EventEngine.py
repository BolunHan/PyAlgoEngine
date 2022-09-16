from types import SimpleNamespace

import EventEngine

from . import LOGGER

__all__ = ['EVENT_ENGINE', 'TOPIC']
LOGGER = LOGGER.getChild('EventEngine')
EventEngine.set_logger(LOGGER)


class TopicSet(object):
    on_order = EventEngine.Topic('on_order')
    on_report = EventEngine.Topic('on_report')
    eod = EventEngine.Topic('eod')
    eod_done = EventEngine.Topic('eod_done')
    bod = EventEngine.Topic('bod')
    bod_done = EventEngine.Topic('bod_done')

    launch_order = EventEngine.PatternTopic('launch_order.{ticker}')
    cancel_order = EventEngine.PatternTopic('cancel_order.{ticker}')
    realtime = EventEngine.PatternTopic('realtime.{ticker}.{dtype}')

    @classmethod
    def push(cls, market_data):
        return cls.realtime(ticker=market_data.ticker, dtype=market_data.__class__.__name__)

    @classmethod
    def parse(cls, topic: EventEngine.Topic) -> SimpleNamespace:
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


EVENT_ENGINE = EventEngine.EventEngine()
TOPIC = TopicSet
# EVENT_ENGINE.start()
