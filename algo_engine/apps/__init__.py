import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('Apps')


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    backtest.set_logger(LOGGER.getChild('Backtester'))


from .bokeh_server import DocServer, DocTheme
from . import backtest
