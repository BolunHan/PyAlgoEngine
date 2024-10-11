import logging
import os

from .telemetrics import LOGGER
from ..profile import PROFILE


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    if os.name == 'nt':
        market_utils_nt.LOGGER = logger.getChild('MarketUtils')
    else:
        market_utils_posix.LOGGER = logger.getChild('MarketUtils')

    trade_utils.LOGGER = logger.getChild('TradeUtils')
    technical_analysis.LOGGER = logger.getChild('TA')
    console_utils.LOGGER = logger.getChild('Console')


from .finance_decimal import FinancialDecimal

if os.name == 'nt':
    from .market_utils_nt import TransactionSide, OrderType, MarketData, OrderBook, BarData, DailyBar, CandleStick, TickData, TransactionData, TradeData, OrderData, MarketDataBuffer, MarketDataRingBuffer
else:
    from .market_utils_posix import TransactionSide, OrderType, MarketData, OrderBook, BarData, DailyBar, CandleStick, TickData, TransactionData, TradeData, OrderData, MarketDataBuffer, MarketDataRingBuffer

    # from .market_utils_posix import OrderType
    # from .market_utils import TransactionSide, MarketData, OrderBook, BarData, DailyBar, CandleStick, TickData, TransactionData, TradeData
    # from .market_buffer import MarketDataPointer, MarketDataMemoryBuffer, OrderBookPointer, OrderBookBuffer, BarDataPointer, BarDataBuffer, TickDataPointer, TickDataBuffer, TransactionDataPointer, TransactionDataBuffer

from .technical_analysis import TechnicalAnalysis
from .trade_utils import OrderState, TradeInstruction, TradeReport
from .console_utils import Progress, GetInput, GetArgs, count_ordinal, TerminalStyle, InteractiveShell, ShellTransfer

__all__ = ['PROFILE',
           'FinancialDecimal',
           'TransactionSide', 'OrderType', 'MarketData', 'OrderBook', 'BarData', 'DailyBar', 'CandleStick', 'TickData', 'TransactionData', 'TradeData', 'OrderData', 'MarketDataBuffer', 'MarketDataRingBuffer',
           # 'MarketDataMemoryBuffer', 'OrderBookBuffer', 'BarDataBuffer', 'TickDataBuffer', 'TransactionDataBuffer',
           # 'MarketDataPointer', 'OrderBookPointer', 'BarDataPointer', 'TickDataPointer', 'TransactionDataPointer',
           'TechnicalAnalysis',
           'OrderState', 'TradeInstruction', 'TradeReport',
           'Progress', 'GetInput', 'GetArgs', 'count_ordinal', 'TerminalStyle', 'InteractiveShell', 'ShellTransfer']
