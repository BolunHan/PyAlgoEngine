import logging
import os
import pathlib

from .telemetrics import LOGGER

USE_CYTHON = True


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger
    from .c_market_data import c_trade_utils
    c_trade_utils.LOGGER = logger.getChild('TradeUtils')
    console_utils.LOGGER = logger.getChild('Console')


def check_cython_module(cython_module) -> bool:
    if not USE_CYTHON:
        return False

    for name in cython_module:
        cython_ext = '.pyd' if os.name == 'nt' else '.so'
        for file in pathlib.Path(__file__).parent.glob(f'*{cython_ext}'):
            if file.name.startswith(name):
                break
        else:
            LOGGER.warning(f'Cython module {name} not found!')
            return False

    return True


from .finance_decimal import FinancialDecimal

from .c_market_data.c_allocator_protocol import EnvConfigContext, AllocatorProtocol, MD_SHARED, MD_LOCKED, MD_FREELIST
from .c_market_data.c_market_data import BookConfigContext, MD_BOOK5, MD_BOOK10, MD_BOOK20, DataType, MarketData, FilterMode, CONFIG
from .c_market_data.c_internal import InternalData
from .c_market_data.c_transaction import OrderType, TransactionDirection, TransactionOffset, TransactionSide, TransactionData, OrderData, TradeData
from .c_market_data.c_tick import TickDataLite, OrderBook, TickData
from .c_market_data.c_candlestick import BarData, DailyBar
from .c_market_data.c_trade_utils import OrderState, TradeReport, TradeInstruction
from .c_market_data.c_market_data_buffer import InvalidBufferError, NotInSharedMemoryError, BufferFull, BufferEmpty, PipeTimeoutError, BufferCorruptedError, MarketDataBuffer, MarketDataBufferCache, MarketDataRingBuffer, MarketDataConcurrentBuffer

from .console_utils import Progress, GetInput, GetArgs, count_ordinal, TerminalStyle, InteractiveShell, ShellTransfer

__all__ = [
    'FinancialDecimal',
    "EnvConfigContext",
    "MD_SHARED", "MD_LOCKED", "MD_FREELIST", "MD_BOOK5", "MD_BOOK10", "MD_BOOK20",
    "DataType", "MarketData", "FilterMode",
    "CONFIG",
    "InternalData",
    "OrderType", "TransactionDirection", "TransactionOffset", "TransactionSide", "TransactionData", "OrderData", "TradeData",
    "TickDataLite", "OrderBook", "TickData",
    "BarData", "DailyBar",
    "OrderState", "TradeReport", "TradeInstruction",
    "InvalidBufferError", "NotInSharedMemoryError", "BufferFull", "BufferEmpty", "PipeTimeoutError", "BufferCorruptedError", "MarketDataBuffer", "MarketDataBufferCache", "MarketDataRingBuffer", "MarketDataConcurrentBuffer",
    'Progress', 'GetInput', 'GetArgs', 'count_ordinal', 'TerminalStyle', 'InteractiveShell', 'ShellTransfer'
]
