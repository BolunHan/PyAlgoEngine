import logging

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
from .market_utils import TransactionSide, MarketData, OrderBook, BarData, TickData, TransactionData, TradeData, OrderBookBuffer, BarDataBuffer, TickDataBuffer, TransactionDataBuffer
from .technical_analysis import TechnicalAnalysis
from .trade_utils import OrderState, OrderType, TradeInstruction, TradeReport
from .console_utils import Progress, GetInput, GetArgs, count_ordinal, TerminalStyle, InteractiveShell, ShellTransfer

__all__ = ['PROFILE',
           'FinancialDecimal',
           'TransactionSide', 'MarketData', 'OrderBook', 'BarData', 'TickData', 'TransactionData', 'TradeData', 'OrderBookBuffer', 'BarDataBuffer', 'TickDataBuffer', 'TransactionDataBuffer',
           'TechnicalAnalysis',
           'OrderState', 'OrderType', 'TradeInstruction', 'TradeReport',
           'Progress', 'GetInput', 'GetArgs', 'count_ordinal', 'TerminalStyle', 'InteractiveShell', 'ShellTransfer']
