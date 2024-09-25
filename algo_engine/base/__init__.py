import logging
import os

from .telemetrics import LOGGER
from ..profile import PROFILE


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    market_utils.LOGGER = logger.getChild('MarketUtils')
    trade_utils.LOGGER = logger.getChild('TradeUtils')
    technical_analysis.LOGGER = logger.getChild('TA')
    console_utils.LOGGER = logger.getChild('Console')


from .finance_decimal import FinancialDecimal

if os.name == 'posix':
    from .market_utils_posix import TransactionSide, MarketData, OrderBook, BarData, DailyBar, CandleStick, TickData, TransactionData, TradeData, MarketDataBuffer, MarketDataRingBuffer
elif os.name == 'nt':
    from .market_utils_nt import TransactionSide, MarketData, OrderBook, BarData, DailyBar, CandleStick, TickData, TransactionData, TradeData, MarketDataBuffer, MarketDataRingBuffer
else:
    from .market_utils import TransactionSide, MarketData, OrderBook, BarData, DailyBar, CandleStick, TickData, TransactionData, TradeData
    from .market_buffer import MarketDataPointer, MarketDataMemoryBuffer, OrderBookPointer, OrderBookBuffer, BarDataPointer, BarDataBuffer, TickDataPointer, TickDataBuffer, TransactionDataPointer, TransactionDataBuffer

from .technical_analysis import TechnicalAnalysis
from .trade_utils import OrderState, OrderType, TradeInstruction, TradeReport
from .console_utils import Progress, GetInput, GetArgs, count_ordinal, TerminalStyle, InteractiveShell, ShellTransfer

__all__ = ['PROFILE',
           'FinancialDecimal',
           'TransactionSide', 'MarketData', 'OrderBook', 'BarData', 'DailyBar', 'CandleStick', 'TickData', 'TransactionData', 'TradeData', 'MarketDataBuffer', 'MarketDataRingBuffer',
           # 'MarketDataMemoryBuffer', 'OrderBookBuffer', 'BarDataBuffer', 'TickDataBuffer', 'TransactionDataBuffer',
           # 'MarketDataPointer', 'OrderBookPointer', 'BarDataPointer', 'TickDataPointer', 'TransactionDataPointer',
           'TechnicalAnalysis',
           'OrderState', 'OrderType', 'TradeInstruction', 'TradeReport',
           'Progress', 'GetInput', 'GetArgs', 'count_ordinal', 'TerminalStyle', 'InteractiveShell', 'ShellTransfer']
