import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('Backtester')

from .doc_server import DocServer, ClassicTheme
from .web_app import DOC_MANAGER, DocManager, start_app


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    doc_server.LOGGER = LOGGER
    web_app.LOGGER = LOGGER


__all__ = [DocServer, ClassicTheme, DOC_MANAGER, DocManager, start_app]
