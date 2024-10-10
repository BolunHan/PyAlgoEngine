import ctypes
import os
import platform

from . import LOGGER
from . import market_utils_posix
from .market_utils_posix import Contexts, BufferConstructor as _BufferConstructor
from .market_utils_posix import TransactionSide, OrderType, MarketData, OrderBook, BarData, DailyBar, CandleStick, TickData, TransactionData, TradeData, OrderData, MarketDataBuffer, MarketDataRingBuffer

LOGGER = LOGGER.getChild('MarketUtils')
__all__ = ['TransactionSide', 'OrderType',
           'MarketData', 'OrderBook', 'BarData', 'DailyBar', 'CandleStick', 'TickData', 'TransactionData', 'TradeData', 'OrderData',
           'MarketDataBuffer', 'MarketDataRingBuffer']
__cache__ = {}

if os.name == 'nt':
    LOGGER.warning(f'MarketUtils support for {platform.system()}-{platform.release()}-{platform.machine()} is limited! Setting contents will have no effect!')


class _OrderBookBuffer(ctypes.Structure):
    ticker_size = Contexts.TICKER_SIZE
    book_size = Contexts.BOOK_SIZE

    _fields_ = [
        ("dtype", ctypes.c_uint8),
        ("ticker", ctypes.c_char * ticker_size),
        ("timestamp", ctypes.c_double),
        ('bid_price', ctypes.c_double * book_size),
        ('ask_price', ctypes.c_double * book_size),
        ('bid_volume', ctypes.c_double * book_size),
        ('ask_volume', ctypes.c_double * book_size),
        ('bid_n_orders', ctypes.c_uint * book_size),
        ('ask_n_orders', ctypes.c_uint * book_size)
    ]


class _CandlestickBuffer(ctypes.Structure):
    ticker_size = Contexts.TICKER_SIZE

    _fields_ = [
        ("dtype", ctypes.c_uint8),
        ("ticker", ctypes.c_char * ticker_size),
        ("timestamp", ctypes.c_double),
        ('start_timestamp', ctypes.c_double),
        ('bar_span', ctypes.c_double),
        ('high_price', ctypes.c_double),
        ('low_price', ctypes.c_double),
        ('open_price', ctypes.c_double),
        ('close_price', ctypes.c_double),
        ('volume', ctypes.c_double),
        ('notional', ctypes.c_double),
        ('trade_count', ctypes.c_uint),
    ]


class _TickDataBuffer(ctypes.Structure):
    ticker_size = Contexts.TICKER_SIZE

    _fields_ = [
        ("dtype", ctypes.c_uint8),
        ("ticker", ctypes.c_char * ticker_size),
        ("timestamp", ctypes.c_double),
        ('order_book', _OrderBookBuffer),
        ('bid_price', ctypes.c_double),
        ('bid_volume', ctypes.c_double),
        ('ask_price', ctypes.c_double),
        ('ask_volume', ctypes.c_double),
        ('last_price', ctypes.c_double),
        ('total_traded_volume', ctypes.c_double),
        ('total_traded_notional', ctypes.c_double),
        ('total_trade_count', ctypes.c_uint),
    ]


class IntID(ctypes.Structure):
    id_size = Contexts.ID_SIZE

    _fields_ = [
        ('id_type', ctypes.c_int),
        ('data', ctypes.c_byte * id_size),
    ]


class StrID(ctypes.Structure):
    id_size = Contexts.ID_SIZE

    _fields_ = [
        ('id_type', ctypes.c_int),
        ('data', ctypes.c_char * id_size),
    ]


class UnionID(ctypes.Union):
    id_size = Contexts.ID_SIZE

    _fields_ = [
        ('id_type', ctypes.c_int),
        ('id_int', IntID),
        ('id_str', StrID),
    ]


class _TransactionDataBuffer(ctypes.Structure):
    ticker_size = Contexts.TICKER_SIZE
    id_size = Contexts.ID_SIZE

    _fields_ = [
        ("dtype", ctypes.c_uint8),
        ("ticker", ctypes.c_char * ticker_size),  # Dynamic size based on TICKER_LEN
        ("timestamp", ctypes.c_double),
        ("price", ctypes.c_double),
        ("volume", ctypes.c_double),
        ("side", ctypes.c_int),
        ("multiplier", ctypes.c_double),
        ("notional", ctypes.c_double),
        ("transaction_id", UnionID),
        ("buy_id", UnionID),
        ("sell_id", UnionID)
    ]


class _OrderDataBuffer(ctypes.Structure):
    ticker_size = Contexts.TICKER_SIZE
    id_size = Contexts.ID_SIZE

    _fields_ = [
        ("dtype", ctypes.c_uint8),
        ("ticker", ctypes.c_char * ticker_size),  # Dynamic size based on TICKER_LEN
        ("timestamp", ctypes.c_double),
        ("price", ctypes.c_double),
        ("volume", ctypes.c_double),
        ("side", ctypes.c_int),
        ("order_id", UnionID),
        ("order_type", ctypes.c_int),
    ]


class _MarketDataBuffer(ctypes.Union):
    _fields_ = [
        ("dtype", ctypes.c_uint8),
        ("OrderBook", _OrderBookBuffer),
        ("BarData", _CandlestickBuffer),
        ("TickData", _TickDataBuffer),
        ("TransactionData", _TransactionDataBuffer),
        ('OrderData', _OrderDataBuffer)
    ]


class BufferConstructor(_BufferConstructor):
    def __init__(self, **kwargs):
        pass

    def __call__(self, dtype: 'str') -> type[ctypes.Structure]:
        match dtype:
            case 'MarketData':
                return self.new_market_data_buffer()
            case 'OrderBook':
                return self.new_orderbook_buffer()
            case 'BarData':
                return self.new_candlestick_buffer()
            case 'TickData':
                return self.new_tick_buffer()
            case 'TradeData' | 'TransactionData':
                return self.new_transaction_buffer()
            case _:
                raise ValueError(f'Invalid dtype {dtype}')

    def new_orderbook_buffer(self) -> type[ctypes.Structure]:
        return _OrderBookBuffer

    def new_candlestick_buffer(self) -> type[ctypes.Structure]:
        return _CandlestickBuffer

    def new_tick_buffer(self) -> type[ctypes.Structure]:
        return _TickDataBuffer

    def new_transaction_buffer(self) -> type[ctypes.Structure]:
        return _TransactionDataBuffer

    def new_order_buffer(self) -> type[ctypes.Structure]:
        return _OrderDataBuffer

    def new_market_data_buffer(self) -> type[ctypes.Union]:
        return _MarketDataBuffer


market_utils_posix._BUFFER_CONSTRUCTOR = BufferConstructor()
MarketDataBuffer.ctype_buffer = _MarketDataBuffer
