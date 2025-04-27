import abc
import datetime
import enum
import inspect
import logging
import operator
import warnings
from collections.abc import Sequence, Mapping, Iterable, Callable
from typing import Literal, Protocol, runtime_checkable, get_type_hints, Self

from . import LOGGER
from ..base import MarketData, DataType, MarketDataBuffer

LOGGER = LOGGER.getChild('Replay')
__all__ = ['PyDataScope', 'MarketDateCallable', 'MarketDataLoader', 'MarketDataBulkLoader', 'Replay', 'SimpleReplay', 'ProgressReplay', 'ProgressiveReplay']


class PyDataScope(enum.Flag):
    SCOPE_TRANSACTION = enum.auto()
    SCOPE_ORDER = enum.auto()
    SCOPE_TICK = enum.auto()
    SCOPE_TICK_LITE = enum.auto()

    SCOPE_ALL = SCOPE_TRANSACTION | SCOPE_ORDER | SCOPE_TICK

    @classmethod
    def _missing_(cls, value: Literal['TickData', 'TickDataLite', 'OrderData', 'TransactionData']):
        if isinstance(value, int):
            return super()._missing_(value)

        if isinstance(value, str):
            dtypes = value.split(',')
        elif isinstance(value, Iterable):
            dtypes = value
        else:
            raise TypeError(value)

        _ = PyDataScope(0)
        for dtype in dtypes:
            _ = _.from_str(dtype)
        return _

    @classmethod
    def get_dtype(cls, dtype: DataType | str) -> str | Literal['TickData', 'TickDataLite', 'OrderData', 'TransactionData']:
        match dtype:
            case 'TickData' | 'TickDataLite' | 'OrderData' | 'TransactionData':
                return str(dtype)
            case 'TradeData':  # handle the alias
                return 'TransactionData'
            case DataType.DTYPE_TICK | DataType.DTYPE_ORDER | DataType.DTYPE_TRANSACTION:
                return DataType(dtype).name.removeprefix('DTYPE_').capitalize() + 'Data'
            case DataType.DTYPE_TICK_LITE:
                return 'Data'.join(_.capitalize() for _ in DataType(dtype).name.removeprefix('DTYPE_').split('_'))
            case _:
                raise ValueError(f'Invalid dtype {dtype}, expect str or int.')

    def __iter__(self):
        return iter(self.to_dtype())

    def to_dtype(self) -> list[DataType]:
        scope = list(super().__iter__())
        scope_dtype = set()

        for dtype in scope:

            if dtype is PyDataScope.SCOPE_TRANSACTION:
                scope_dtype.add(DataType.DTYPE_TRANSACTION)
            elif dtype is PyDataScope.SCOPE_ORDER:
                scope_dtype.add(DataType.DTYPE_ORDER)
            elif dtype is PyDataScope.SCOPE_TICK_LITE:
                scope_dtype.add(DataType.DTYPE_TICK_LITE)
            elif dtype is PyDataScope.SCOPE_TICK:
                scope_dtype.add(DataType.DTYPE_TICK)

        return list(scope_dtype)

    def to_int(self) -> list[int]:
        return [int(_) for _ in self.to_dtype()]

    def to_str(self) -> list[str]:
        return [self.get_dtype(_) for _ in self.to_dtype()]

    def from_str(self, dtype: Literal['TickData', 'TickDataLite', 'OrderData', 'TransactionData']) -> Self:
        match dtype:
            case 'TickData':
                return self | self.SCOPE_TICK
            case 'TickDataLite':
                return self | self.SCOPE_TICK_LITE
            case 'OrderData':
                return self | self.SCOPE_ORDER
            case 'TransactionData' | 'TradeData':
                return self | self.SCOPE_TRANSACTION
            case _:
                raise ValueError(f'Invalid str {dtype}.')


@runtime_checkable
class MarketDateCallable(Protocol):
    def __call__(self, market_date: datetime.date) -> None:
        ...


@runtime_checkable
class MarketDataLoader(Protocol):
    def __call__(self, market_date: datetime.date, ticker: str, dtype: str | DataType) -> Sequence[MarketData] | Mapping[float, MarketData]:
        pass


@runtime_checkable
class MarketDataBulkLoader(Protocol):
    def __call__(self, market_date: datetime.date, tickers: Sequence[str], dtypes: Sequence[str | DataType] | PyDataScope) -> Sequence[MarketData] | Mapping[float, MarketData] | MarketDataBuffer:
        pass


def check_protocol_signature(func: Callable, protocol: type) -> bool:
    if not callable(func):
        raise TypeError(f"{func} is not callable")

    proto_sig = inspect.signature(protocol.__call__)
    func_sig = inspect.signature(func)

    proto_params = list(proto_sig.parameters.values())[1:]  # Skip 'self'
    func_params = list(func_sig.parameters.values())
    enable_keywords = False

    # Check for *args (VAR_POSITIONAL) — not allowed
    for p in func_params:
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(f"{func.__name__} uses *args, which is not allowed")
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            enable_keywords = True

    # Extract positional args (POSITIONAL_ONLY or POSITIONAL_OR_KEYWORD)
    proto_arg_names = [p.name for p in proto_params if p.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD
    )]

    func_arg_names = [p.name for p in func_params if p.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD
    )]

    # Check if required positional args match (ignore **kwargs)
    if not enable_keywords and sorted(proto_arg_names) != sorted(func_arg_names):
        warnings.warn(
            f"{func} argument names {func_arg_names} do not match protocol {proto_arg_names}",
            stacklevel=2
        )
        return False

    # Type hint comparison (warn if mismatched, but allow)
    proto_hints = get_type_hints(protocol.__call__)
    func_hints = get_type_hints(func)

    for pname in proto_arg_names:
        expected = proto_hints.get(pname)
        actual = func_hints.get(pname)
        if expected and actual and expected != actual:
            warnings.warn(
                f"Type hint mismatch for parameter '{pname}': expected {expected}, got {actual}",
                stacklevel=2
            )

    # Optional: check return type
    expected_ret = proto_hints.get("return")
    actual_ret = func_hints.get("return")
    if expected_ret and actual_ret and expected_ret != actual_ret:
        warnings.warn(
            f"Return type mismatch: expected {expected_ret}, got {actual_ret}",
            stacklevel=2
        )

    return True


class Replay(object, metaclass=abc.ABCMeta):
    # __slots__ = ('start_date', 'end_date', 'market_date', 'calendar', 'bod', 'eod', 'subscription', '_calendar', '_market_date', '_status', '_progress')

    def __init__(self, start_date: datetime.date = None, end_date: datetime.date = None, market_date: datetime.date = None, calendar: Sequence[datetime.date] = None, bod: MarketDateCallable = None, eod: MarketDateCallable = None) -> None:
        self.start_date = start_date or market_date or calendar[0]
        self.end_date = end_date or calendar[-1]
        self.market_date = market_date or start_date
        self.calendar = calendar or []

        self.bod = []
        self.eod = []
        self.subscription = {}

        if bod is not None:
            self.add_bod(bod)

        if eod is not None:
            self.add_eod(eod)

    def add_bod(self, func: MarketDateCallable, priority: int = None) -> None:
        if priority is None:
            self.bod.append(func)
        else:
            self.bod.insert(priority, func)

    def add_eod(self, func: MarketDateCallable, priority: int = None):
        if priority is None:
            self.eod.append(func)
        else:
            self.eod.insert(priority, func)

    def add_subscription(self, ticker: str, dtype: DataType | str):
        dtype = PyDataScope.get_dtype(dtype)
        topic = f'{ticker}.{dtype}'

        self.subscription[topic] = (ticker, dtype)

    def remove_subscription(self, ticker: str, dtype: DataType | str):
        dtype = PyDataScope.get_dtype(dtype)
        topic = f'{ticker}.{dtype}'

        try:
            self.subscription.pop(topic)
        except KeyError as _:
            LOGGER.info(f'{topic} not in {self.subscription}')

    @abc.abstractmethod
    def __next__(self):
        ...

    @abc.abstractmethod
    def __iter__(self):
        ...


class SimpleReplay(Replay):
    def __init__(
            self,
            loader: MarketDataBulkLoader | MarketDataLoader = None,
            market_date: datetime.date = None,
            start_date: datetime.date = None,
            end_date: datetime.date = None,
            calendar: Sequence[datetime.date] = None,
            bod: MarketDateCallable = None,
            eod: MarketDateCallable = None
    ):
        super().__init__(market_date=market_date, start_date=start_date, end_date=end_date, calendar=calendar, bod=bod, eod=eod)
        self.loader = loader

    def __iter__(self):
        self._calendar = self.calendar or [self.start_date + datetime.timedelta(days=i) for i in range((self.end_date - self.start_date).days + 1)]
        self._market_date = sorted(_ for _ in self._calendar if _ >= self.market_date)[0]
        self._status = {market_date: 'skipped' if market_date < self.market_date else 'idle' for market_date in self._calendar}
        self._idx_buffer = 0
        self._idx_date = sum([1 for _ in self._calendar if _ < self.market_date])

        for func in self.bod:
            func(self._market_date)

        self._safe_load()

        return self

    def __next__(self) -> MarketData:
        if self._idx_buffer < self._buffer_size:
            self._idx_buffer += 1
            return next(self._buffer)

        for func in self.eod:
            func(self._market_date)

        self._idx_buffer = 0
        self._idx_date += 1

        if self._idx_date >= len(self._calendar):
            self._calendar.clear()
            del self._calendar
            del self._market_date
            del self._status
            del self._idx_buffer
            del self._idx_date
            del self._buffer
            del self._buffer_size
            raise StopIteration()

        self._market_date = self._calendar[self._idx_date]

        for func in self.bod:
            func(self._market_date)

        self._safe_load()
        return self.__next__()

    def __repr__(self):
        return f'{self.__class__.__name__}{{id={id(self)}, from={self.start_date}, to={self.end_date}}}'

    def _bulk_load_protocol(self):
        LOGGER.info(f'{self} loading {self._market_date} {(', '.join(self.dtypes)) if self.dtypes else 'data'} for {len(self.tickers)} tickers...')
        buffer = self.loader(market_date=self._market_date, tickers=self.tickers, dtypes=self.dtypes)
        LOGGER.info(f'{self} sorting {self._market_date} data...')
        buffer.sort()

        if isinstance(buffer, MarketDataBuffer):
            self._buffer = buffer
            self._buffer_size = len(self._buffer)
        elif isinstance(buffer, Sequence):
            self._buffer = iter(buffer)
            self._buffer_size = len(buffer)
        elif isinstance(buffer, Mapping):
            self._buffer = iter(buffer.values())
            self._buffer_size = len(buffer)
        LOGGER.info(f'{self} {self._market_date} total {self._buffer_size:,} items loaded.')

    def _individual_load_protocol(self):
        buffer = []
        for topic, (_ticker, _dtype) in self.subscription.items():
            LOGGER.info(f'{self} loading {self._market_date} {_ticker} {_dtype}...')
            data = self.loader(market_date=self._market_date, ticker=_ticker, dtype=_dtype)
            if isinstance(data, Mapping):
                buffer.extend(list(data.values()))
            elif isinstance(data, Sequence):
                buffer.extend(data)
            else:
                raise TypeError(f'The loader {self.loader} returned {type(data)}. Expect a sequence or mapping of MarketData')
        LOGGER.info(f'{self} sorting {self._market_date} data...')
        buffer.sort(key=operator.attrgetter('timestamp', 'ticker', '_dtype'))
        self._buffer = iter(buffer)
        self._buffer_size = len(buffer)
        LOGGER.info(f'{self} {self._market_date} total {self._buffer_size:,} items loaded.')

    def _safe_load(self):
        if self.loader is None:
            assert hasattr(self, '_buffer') and isinstance(self._buffer, Iterable), f'Without assigning a data loader, the _buffer of {self.__class__.__name__} should be set in bod process.'
            return None

        is_bulk_loader = check_protocol_signature(self.loader, MarketDataBulkLoader)
        is_individual_loader = check_protocol_signature(self.loader, MarketDataLoader)

        if (is_bulk_loader and is_individual_loader) or (not is_bulk_loader and not is_individual_loader):
            try:
                return self._bulk_load_protocol()
            except Exception as e:
                LOGGER.info('Failed to load data using MarketDataBulkLoader protocol!')

            try:
                return self._individual_load_protocol()
            except Exception as e:
                LOGGER.info('Failed to load data using MarketDataLoader protocol!')
                raise

        if is_bulk_loader:
            return self._bulk_load_protocol()

        return self._individual_load_protocol()

    @property
    def progress(self) -> float:
        if not hasattr(self, '_buffer'):
            raise RuntimeError(f'{self.__class__.__name__} not started yet.')

        return (self._idx_date + self._idx_buffer / self._buffer_size) / len(self._calendar)

    @property
    def tickers(self) -> list[str]:
        tickers = set()
        for _, (ticker, dtype) in self.subscription.items():
            tickers.add(ticker)
        return list(tickers)

    @property
    def dtypes(self) -> list[str]:
        dtypes = set()
        for _, (ticker, dtype) in self.subscription.items():
            dtypes.add(dtype)
        return list(dtypes)

    @property
    def status(self) -> dict[datetime.date, str]:
        if not hasattr(self, '_status'):
            raise RuntimeError(f'{self.__class__.__name__} not started yet.')

        return self._status


class ProgressReplay(SimpleReplay):
    def __init__(
            self,
            loader: MarketDataBulkLoader | MarketDataLoader = None,
            market_date: datetime.date = None,
            start_date: datetime.date = None,
            end_date: datetime.date = None,
            calendar: Sequence[datetime.date] = None,
            bod: MarketDateCallable = None,
            eod: MarketDateCallable = None,
            **pbar_config
    ):
        super().__init__(
            loader=loader,
            market_date=market_date,
            start_date=start_date,
            end_date=end_date,
            calendar=calendar,
            bod=bod,
            eod=eod
        )

        self.pbar_config = {
            'backend': pbar_config.pop('backend', 'tqdm'),  # tqdm or native
            'config': pbar_config,
        }
        self._pbar = None

    def _init_pbar_tqdm(self):
        from tqdm.auto import tqdm
        from tqdm.std import tqdm as tqdm_std
        from tqdm.contrib.logging import _TqdmLoggingHandler, _get_first_found_console_logging_handler, _is_console_logging_handler

        tqdm_config = {
            'total': 1,
            'unit_scale': True,
            'unit': 'percent',
            'mininterval': 0.1,
            'miniters': 0.001,
            **self.pbar_config['config'],
        }
        self._pbar = tqdm(**tqdm_config)

        self.pbar_config['loggers'] = loggers = [LOGGER.root] + [_ for _ in LOGGER.root.manager.loggerDict.values() if isinstance(_, logging.Logger) and _.handlers]
        self.pbar_config['original_handlers_list'] = [logger.handlers for logger in loggers]
        for logger in loggers:
            tqdm_handler = _TqdmLoggingHandler(tqdm_std)
            orig_handler = _get_first_found_console_logging_handler(logger.handlers)
            if orig_handler is not None:
                tqdm_handler.setFormatter(orig_handler.formatter)
                tqdm_handler.stream = orig_handler.stream
            logger.handlers = [handler for handler in logger.handlers if not _is_console_logging_handler(handler)] + [tqdm_handler]

        self.add_bod(self._init_pbar_tqdm_secondary, priority=0)
        self.add_eod(self._close_pbar_tqdm_secondary, priority=0)
        self.add_bod(self._update_tqdm_prefix, priority=0)
        self._update_pbar_progress = self._update_tqdm_progress

    def _init_pbar_tqdm_secondary(self, market_date):
        from tqdm.auto import tqdm

        tqdm_secondary_config = {
            'total': 1,
            'unit_scale': True,
            'unit': 'percent',
            'mininterval': 0.1,
            'miniters': 0.001,
            **self.pbar_config['config'],
        }
        self._pbar_secondary = tqdm(**tqdm_secondary_config)
        prompt = f'Progress Total ({self._idx_date + 1} / {len(self._calendar)})'
        prompt_secondary = f'Progress [{market_date:%Y-%m-%d}]'
        prompt_length = max(len(prompt), len(prompt_secondary))
        self._pbar_secondary.n = 0
        self._pbar_secondary.set_description(prompt_secondary.ljust(prompt_length))
        self._pbar_secondary.refresh()

    def _close_pbar_tqdm_secondary(self, market_date: datetime.date):
        self._pbar_secondary.n = 1
        # self._pbar_secondary.refresh()
        self._pbar_secondary.close()
        self._pbar_secondary = None

    def _init_pbar_native(self):
        from ..base import Progress

        progress_config = dict(
            tasks=1,
            tick_size=0.001,
            **self.pbar_config['config'],
        )

        self.add_bod(self._update_native_prefix, priority=0)
        self._pbar = Progress(**progress_config)
        self._update_pbar_progress = self._update_native_progress

    def _update_tqdm_prefix(self, market_date: datetime.date):
        prompt = f'Progress Total ({self._idx_date + 1} / {len(self._calendar)})'
        self._pbar.set_description(prompt)
        self._pbar.refresh()

    def _update_native_prefix(self, market_date: datetime.date):
        self._pbar.prompt = f'Replay {market_date:%Y-%m-%d} ({self._idx_date + 1} / {len(self._calendar)}):'
        self._pbar.output()

    def _close_pbar_tqdm(self):
        for logger, original_handlers in zip(self.pbar_config['loggers'], self.pbar_config['original_handlers_list']):
            logger.handlers = original_handlers

        self._pbar.n = 1
        # self._pbar.refresh()
        self._pbar.close()
        self._pbar = None

    def _close_pbar_native(self):
        self._pbar.done_tasks = 1
        self._pbar.output()

    def _update_tqdm_progress(self):
        self._pbar.n = self.progress
        self._pbar.update(0)

        self._pbar_secondary.n = self._idx_buffer / self._buffer_size
        self._pbar_secondary.update(0)

    def _update_native_progress(self):
        self._pbar.done_tasks = self.progress

        if (not self._pbar.tick_size) \
                or self._pbar.progress >= self._pbar.tick_size + self._pbar.last_output \
                or self._pbar.is_done:
            self._pbar.output()

    def __iter__(self):
        pbar_backend = self.pbar_config['backend']

        match pbar_backend:
            case 'tqdm':
                self._init_pbar_tqdm()
            case 'native':
                self._init_pbar_native()
            case _:
                raise NotImplementedError(f'Invalid pbar backend {pbar_backend}')

        return super().__iter__()

    def __next__(self) -> MarketData:
        try:
            result = super().__next__()
            if self._pbar is not None:
                self._update_pbar_progress()
            return result
        except StopIteration:
            if self._pbar is not None:
                pbar_backend = self.pbar_config['backend']
                match pbar_backend:
                    case 'tqdm':
                        self._close_pbar_tqdm()
                    case 'native':
                        self._close_pbar_native()
                    case _:
                        raise NotImplementedError(f'Invalid pbar backend {pbar_backend}')
            raise


class ProgressiveReplay(SimpleReplay):
    """
    progressively loading and replaying market data

    requires arguments
    loader: a data loading function. Expect loader = Callable(market_date: datetime.date, ticker: str, dtype: str| type) -> dict[any, MarketData]
    start_date & end_date: the given replay period
    or calendar: the given replay calendar.

    accepts kwargs:
    ticker / tickers: the given symbols to replay, expect a str| list[str]
    dtype / dtypes: the given dtype(s) of symbol to replay, expect a str | type, list[str | type]. default = all, which is (TradeData, TickData, OrderBook)
    subscription / subscribe: the given ticker-dtype pair to replay, expect a list[dict[str, str | type]]
    """

    def __init__(
            self,
            loader: MarketDataLoader,
            tickers: str | Sequence[str] = None,
            dtypes: str | DataType | Sequence[str] | Sequence[DataType] = None,
            market_date: datetime.date = None,
            start_date: datetime.date = None,
            end_date: datetime.date = None,
            calendar: Sequence[datetime.date] = None,
            bod: MarketDateCallable = None,
            eod: MarketDateCallable = None,
            **progress_config
    ) -> None:
        warnings.warn('User ProgressReplay instead!', DeprecationWarning, stacklevel=2)
        self.loader = loader
        super().__init__(loader=loader, market_date=market_date, start_date=start_date, end_date=end_date, calendar=calendar, bod=bod, eod=eod)

        tickers = tickers or []
        dtypes = dtypes or ['TransactionData', 'TickData', 'OrderData']

        if not isinstance(loader, MarketDataLoader):
            raise TypeError('loader function has 3 requires args, market_date, ticker and dtype.')

        if isinstance(tickers, str):
            tickers = [tickers]
        elif isinstance(tickers, Iterable):
            tickers = list(tickers)
        else:
            raise TypeError(f'Invalid ticker {tickers}, expect str or list[str]')

        if isinstance(dtypes, (str, int, DataType)):
            dtypes = [dtypes]
        elif isinstance(dtypes, Iterable):
            dtypes = list(dtypes)
        else:
            raise TypeError(f'Invalid dtype {dtypes}, expect str or list[str]')

        for ticker in tickers:
            for dtype in dtypes:
                self.add_subscription(ticker=ticker, dtype=dtype)

        self.progress_config = dict(
            tasks=1,
            **progress_config
        )
        self._pbar = None
        self.add_bod(self._update_progress_bar, priority=0)

    def __iter__(self):
        from ..base import Progress
        self._pbar = Progress(**self.progress_config)
        return super().__iter__()

    def __next__(self) -> MarketData:
        try:
            result = super().__next__()
            if self._pbar:
                self._pbar.done_tasks = self.progress

                if (not self._pbar.tick_size) \
                        or self._pbar.progress >= self._pbar.tick_size + self._pbar.last_output \
                        or self._pbar.is_done:
                    self._pbar.output()

            return result
        except StopIteration:
            if self._pbar is not None and not self._pbar.is_done:
                self.progress.done_tasks = 1
                self._pbar.output()
            raise

    def _update_progress_bar(self, market_date: datetime.date):
        if self._pbar:
            self.progress.prompt = f'Replay {market_date:%Y-%m-%d} ({self._idx_date + 1} / {len(self._calendar)}):'
            self._pbar.output()
