import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('Apps')


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    backtester.set_logger(LOGGER.getChild('Backtester'))


from . import backtester
