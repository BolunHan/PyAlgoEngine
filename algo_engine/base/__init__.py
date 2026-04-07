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
    technical_analysis.LOGGER = logger.getChild('TA')
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

from .c_market_data_ng import MarketData, DataType, OrderType, InternalData, FilterMode, CONFIG
from .c_market_data_ng import TransactionDirection, TransactionOffset, TransactionSide, TransactionData, TradeData, OrderData
from .c_market_data_ng import TickDataLite, OrderBook, TickData
from .c_market_data_ng import BarData, DailyBar
from .c_market_data_ng import MarketDataBuffer, MarketDataRingBuffer, MarketDataConcurrentBuffer
from .c_market_data_ng import OrderState, TradeInstruction, TradeReport

from .technical_analysis import TechnicalAnalysis
from .console_utils import Progress, GetInput, GetArgs, count_ordinal, TerminalStyle, InteractiveShell, ShellTransfer

__all__ = [
    # 'PROFILE',
    'FinancialDecimal',
    'TransactionDirection', 'TransactionOffset', 'TransactionSide', 'OrderType', 'InternalData', 'MarketData', 'DataType', 'BarData', 'DailyBar', 'TickDataLite', 'OrderBook', 'TickData', 'TransactionData', 'TradeData', 'OrderData', 'MarketDataBuffer', 'MarketDataRingBuffer', 'MarketDataConcurrentBuffer',
    # 'MarketDataMemoryBuffer', 'OrderBookBuffer', 'BarDataBuffer', 'TickDataBuffer', 'TransactionDataBuffer',
    # 'MarketDataPointer', 'OrderBookPointer', 'BarDataPointer', 'TickDataPointer', 'TransactionDataPointer',
    'TechnicalAnalysis',
    'OrderState', 'TradeInstruction', 'TradeReport',
    'Progress', 'GetInput', 'GetArgs', 'count_ordinal', 'TerminalStyle', 'InteractiveShell', 'ShellTransfer'
]
