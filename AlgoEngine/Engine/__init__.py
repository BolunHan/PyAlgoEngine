from __future__ import annotations

import logging
import os
import sys
import time

LOGGER: logging.Logger | None = None
LOG_LEVEL = logging.INFO

if 'ALGO_DIR' in os.environ:
    WORKING_DIRECTORY = os.path.realpath(os.environ['ALGO_DIR'])
else:
    WORKING_DIRECTORY = str(os.getcwd())


class ColoredFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    def __init__(self, fmt=None, datefmt=None, style='{', validate=True):
        self.format_str = '[{asctime} {name} - {threadName} - {module}:{lineno} - {levelname}] {message}' if fmt is None else fmt
        self.date_fmt = '%Y-%m-%d %H:%M:%S' if datefmt is None else datefmt
        self.style = style

        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)

    def _get_format(self, level: int, select=False):
        bold_red = f"\33[31;1;3;4{';7' if select else ''}m"
        red = f"\33[31;1{';7' if select else ''}m"
        green = f"\33[32;1{';7' if select else ''}m"
        yellow = f"\33[33;1{';7' if select else ''}m"
        blue = f"\33[34;1{';7' if select else ''}m"
        reset = "\33[0m"

        if level <= logging.NOTSET:
            fmt = self.format_str
        elif level <= logging.DEBUG:
            fmt = blue + self.format_str + reset
        elif level <= logging.INFO:
            fmt = green + self.format_str + reset
        elif level <= logging.WARNING:
            fmt = yellow + self.format_str + reset
        elif level <= logging.ERROR:
            fmt = red + self.format_str + reset
        else:
            fmt = bold_red + self.format_str + reset

        return fmt

    def format(self, record):
        log_fmt = self._get_format(level=record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.date_fmt, style=self.style)
        return formatter.format(record)


def get_logger(**kwargs) -> logging.Logger:
    level = kwargs.get('level', LOG_LEVEL)
    stream_io = kwargs.get('stream_io', sys.stdout)
    formatter = kwargs.get('formatter', ColoredFormatter())
    global LOGGER

    if LOGGER is not None:
        return LOGGER

    LOGGER = logging.getLogger('PyAlgoEngine')
    LOGGER.setLevel(level)
    logging.Formatter.converter = time.gmtime

    if stream_io:
        have_handler = False
        for handler in LOGGER.handlers:
            # noinspection PyUnresolvedReferences
            if type(handler) == logging.StreamHandler and handler.stream == stream_io:
                have_handler = True
                break

        if not have_handler:
            logger_ch = logging.StreamHandler(stream=stream_io)
            logger_ch.setLevel(level=level)
            logger_ch.setFormatter(fmt=formatter)
            LOGGER.addHandler(logger_ch)

    return LOGGER


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger


_ = get_logger()

from .EventEngine import EVENT_ENGINE, TOPIC
from .AlgoEngine import AlgoTemplate, ALGO_ENGINE, ALGO_REGISTRY
from .MarketEngine import MDS, MarketDataService, MarketDataMonitor, SyntheticOrderBookMonitor, MinuteBarMonitor, ProgressiveReplay, SimpleReplay, Replay
from .TradeEngine import DirectMarketAccess, Balance, PositionManagementService, Inventory, RiskProfile, SimMatch

__all__ = ['set_logger', 'LOGGER', 'EVENT_ENGINE', 'TOPIC',
           'AlgoTemplate', 'ALGO_ENGINE', 'ALGO_REGISTRY',
           'MDS', 'MarketDataService', 'MarketDataMonitor', 'SyntheticOrderBookMonitor', 'MinuteBarMonitor', 'ProgressiveReplay', 'SimpleReplay', 'Replay',
           'DirectMarketAccess', 'Balance', 'PositionManagementService', 'Inventory', 'RiskProfile', 'SimMatch']
