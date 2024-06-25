import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('Backtester')

from .doc_server import CandleStick, StickTheme
from .web_app import WebApp, start_app
from .tester import Tester


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    doc_server.LOGGER = LOGGER
    web_app.LOGGER = LOGGER


__all__ = ['CandleStick', 'StickTheme', 'WebApp', 'start_app', 'Tester']
