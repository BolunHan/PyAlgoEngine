import logging
import os
import pathlib

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


def check_cython_module() -> bool:
    for name in ['market_data', 'transaction', 'tick', 'candlestick', 'market_data_buffer']:
        cython_ext = '.pyd' if os.name == 'nt' else '.so'
        for file in pathlib.Path(__file__).parent.glob(f'*{cython_ext}'):
            if file.name.startswith(name):
                break
        else:
            LOGGER.warning(f'Cython module {name} not found!')
            return False

    return True


from .finance_decimal import FinancialDecimal

if check_cython_module():
    from .market_data import MarketData
    from .transaction import TransactionDirection, TransactionOffset, TransactionSide, OrderType, TransactionData, TradeData, OrderData
    from .tick import TickDataLite, TickData
    from .candlestick import BarData, DailyBar
    from .market_data_buffer import MarketDataBuffer
else:
    import_cmd = f'from .market_utils_{os.name} import TransactionSide, OrderType, MarketData, OrderBook, BarData, DailyBar, CandleStick, TickData, TransactionData, TradeData, OrderData, MarketDataBuffer, MarketDataRingBuffer'
    eval(import_cmd)

from .technical_analysis import TechnicalAnalysis
from .trade_utils import OrderState, TradeInstruction, TradeReport
from .console_utils import Progress, GetInput, GetArgs, count_ordinal, TerminalStyle, InteractiveShell, ShellTransfer

__all__ = [
    'PROFILE',
    'FinancialDecimal',
    'TransactionDirection', 'TransactionOffset', 'TransactionSide', 'OrderType', 'MarketData', 'BarData', 'DailyBar', 'TickDataLite', 'TickData', 'TransactionData', 'TradeData', 'OrderData', 'MarketDataBuffer',
    # 'MarketDataMemoryBuffer', 'OrderBookBuffer', 'BarDataBuffer', 'TickDataBuffer', 'TransactionDataBuffer',
    # 'MarketDataPointer', 'OrderBookPointer', 'BarDataPointer', 'TickDataPointer', 'TransactionDataPointer',
    'TechnicalAnalysis',
    'OrderState', 'TradeInstruction', 'TradeReport',
    'Progress', 'GetInput', 'GetArgs', 'count_ordinal', 'TerminalStyle', 'InteractiveShell', 'ShellTransfer'
]
