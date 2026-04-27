import logging
import sys
import time

LOGGER: logging.Logger | None = None
LOG_LEVEL = logging.INFO


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


class DuplicateWarningFilter(logging.Filter):
    """Filter that lets each WARNING message pass only once per handler."""

    def __init__(self, name: str = ''):
        super().__init__(name=name)
        self._seen_warnings: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.WARNING:
            return True

        message = record.getMessage()
        if message in self._seen_warnings:
            return False

        self._seen_warnings.add(message)
        return True


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
        stream_handler: logging.StreamHandler | None = None
        for handler in LOGGER.handlers:
            # noinspection PyUnresolvedReferences
            if isinstance(handler, logging.StreamHandler) and handler.stream == stream_io:
                have_handler = True
                stream_handler = handler
                break

        if not have_handler:
            logger_ch = logging.StreamHandler(stream=stream_io)
            logger_ch.setLevel(level=level)
            logger_ch.setFormatter(fmt=formatter)
            LOGGER.addHandler(logger_ch)
            stream_handler = logger_ch

        if stream_handler is not None and not any(isinstance(flt, DuplicateWarningFilter) for flt in stream_handler.filters):
            stream_handler.addFilter(DuplicateWarningFilter())

    assert LOGGER is not None
    return LOGGER


_ = get_logger()
