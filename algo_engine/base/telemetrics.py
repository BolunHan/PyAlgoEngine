import logging

from cbase.backports.telemetrics import ColoredFormatter, DuplicateWarningFilter, LOG_LEVEL, get_logger as _get_logger


def get_logger(**kwargs) -> logging.Logger:
    return _get_logger(name='PyAlgoEngine', **kwargs)


LOGGER = get_logger()

__all__ = [
    'LOGGER',
    'LOG_LEVEL',
    'ColoredFormatter',
    'DuplicateWarningFilter',
    'get_logger',
]
