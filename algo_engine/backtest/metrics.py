import uuid

import numpy as np
import pandas as pd


class TradeMetrics(object):
    def __init__(self):
        self.trades = {}
        self.trade_batch = []

        self.exposure = 0.
        self.total_pnl = 0.
        self.total_cash_flow = 0.

        self.current_pnl = 0.
        self.current_cash_flow = 0.
        self.current_trade_batch = {'cash_flow': 0., 'pnl': 0., 'turnover': 0., 'trades': []}
        self.market_price = None

    def update(self, market_price: float):
        self.market_price = market_price
        self.total_pnl = self.exposure * market_price + self.total_cash_flow
        self.current_pnl = self.exposure * market_price + self.current_cash_flow
        self.current_trade_batch['pnl'] = self.exposure * market_price + self.current_trade_batch['cash_flow']

    def add_trades(self, side: int, price: float, timestamp: float, volume: float = None, trade_id: int | str = None):
        assert side in {1, -1}, f"trade side must in {1, -1}, got {side}."
        assert volume is None or volume >= 0, "volume must be positive."

        if volume is None:
            if self.exposure * side < 0:
                volume = abs(self.exposure)
            elif self.exposure * side > 0:
                volume = 0.
            else:
                volume = 1.

        if trade_id is None:
            trade_id = uuid.uuid4().int
        elif trade_id in self.trades:
            return

        # split the trades
        if (target_exposure := self.exposure + volume * side) * self.exposure < 0:
            self.add_trades(side=side, volume=abs(self.exposure), price=price, timestamp=timestamp, trade_id=f'{trade_id}.0')
            volume = volume - abs(self.exposure)
            trade_id = f'{trade_id}.1'

        self.exposure += volume * side
        self.total_cash_flow -= volume * side * price
        self.total_pnl = self.exposure * price + self.total_cash_flow
        self.current_cash_flow -= volume * side * price
        self.current_pnl = self.exposure * price + self.current_cash_flow
        self.market_price = price

        self.trades[trade_id] = trade_log = dict(
            side=side,
            volume=volume,
            timestamp=timestamp,
            price=price,
            exposure=self.exposure,
            cash_flow=self.current_cash_flow,
            pnl=self.current_pnl
        )

        if 'init_side' not in self.current_trade_batch:
            self.current_trade_batch['init_side'] = side

        self.current_trade_batch['cash_flow'] -= volume * side * price
        self.current_trade_batch['pnl'] = self.exposure * price + self.current_trade_batch['cash_flow']
        self.current_trade_batch['turnover'] += abs(volume) * price
        self.current_trade_batch['trades'].append(trade_log)

        if not self.exposure:
            self.trade_batch.append(self.current_trade_batch)
            self.current_trade_batch = {'cash_flow': 0., 'pnl': 0., 'turnover': 0., 'trades': []}
            self.current_pnl = self.current_cash_flow = 0.

    def add_trades_batch(self, trade_logs: pd.DataFrame):
        for timestamp, row in trade_logs.iterrows():  # type: float, dict
            side = row['side']
            price = row['current_price']
            volume = row['signal']
            self.add_trades(side=side, volume=volume, price=price, timestamp=timestamp)

    def clear(self):
        self.trades.clear()
        self.trade_batch.clear()

        self.exposure = 0.
        self.total_pnl = 0.
        self.total_cash_flow = 0.

        self.current_pnl = 0.
        self.current_cash_flow = 0.
        self.current_trade_batch = {'cash_flow': 0., 'pnl': 0., 'turnover': 0., 'trades': []}
        self.market_price = None

    @property
    def summary(self):
        info_dict = dict(
            total_gain=0.,
            total_loss=0.,
            trade_count=0,
            win_count=0,
            lose_count=0,
            turnover=0.,
        )

        for trade_batch in self.trade_batch:
            if trade_batch['pnl'] > 0:
                info_dict['total_gain'] += trade_batch['pnl']
                info_dict['trade_count'] += 1
                info_dict['win_count'] += 1
                info_dict['turnover'] += trade_batch['turnover']
            else:
                info_dict['total_loss'] += trade_batch['pnl']
                info_dict['trade_count'] += 1
                info_dict['lose_count'] += 1
                info_dict['turnover'] += trade_batch['turnover']

        info_dict['win_rate'] = info_dict['win_count'] / info_dict['trade_count'] if info_dict['trade_count'] else 0.
        info_dict['average_gain'] = info_dict['total_gain'] / info_dict['win_count'] / self.market_price if info_dict['win_count'] else 0.
        info_dict['average_loss'] = info_dict['total_loss'] / info_dict['lose_count'] / self.market_price if info_dict['lose_count'] else 0.
        info_dict['gain_loss_ratio'] = -info_dict['average_gain'] / info_dict['average_loss'] if info_dict['average_loss'] else 1.
        info_dict['long_avg_pnl'] = np.average([_['pnl'] for _ in long_trades]) / self.market_price if (long_trades := [_ for _ in self.trade_batch if _['init_side'] == 1]) else np.nan
        info_dict['short_avg_pnl'] = np.average([_['pnl'] for _ in short_trades]) / self.market_price if (short_trades := [_ for _ in self.trade_batch if _['init_side'] == -1]) else np.nan
        info_dict['ttl_pnl.no_leverage'] = np.sum([trade_batch['pnl'] for trade_batch in self.trade_batch])
        info_dict['net_pnl.optimistic'] = info_dict['ttl_pnl.no_leverage'] - (0.00034 + 0.000023) / 2 * info_dict['turnover']

        return info_dict

    @property
    def info(self):
        trade_info = []
        trade_index = []
        for batch_id, trade_batch in enumerate(self.trade_batch):
            for trade_id, trade_dict in enumerate(trade_batch['trades']):
                trade_info.append(
                    dict(
                        timestamp=trade_dict['timestamp'],
                        side=trade_dict['side'],
                        volume=trade_dict['volume'],
                        price=trade_dict['price'],
                        exposure=trade_dict['exposure'],
                        pnl=trade_dict['pnl']
                    )
                )
                trade_index.append((f'batch.{batch_id}', f'trade.{trade_id}'))

        df = pd.DataFrame(trade_info, index=trade_index)
        return df

    def to_string(self) -> str:
        metric_info = self.summary

        fmt_dict = {
            'total_gain': f'{metric_info["total_gain"]:,.3f}',
            'total_loss': f'{metric_info["total_loss"]:,.3f}',
            'trade_count': f'{metric_info["trade_count"]:,}',
            'win_count': f'{metric_info["win_count"]:,}',
            'lose_count': f'{metric_info["lose_count"]:,}',
            'turnover': f'{metric_info["turnover"]:,.3f}',
            'win_rate': f'{metric_info["win_rate"]:.2%}',
            'average_gain': f'{metric_info["average_gain"]:,.4%}',
            'average_loss': f'{metric_info["average_loss"]:,.4%}',
            'long_avg_pnl': f'{metric_info["long_avg_pnl"]:,.4%}',
            'short_avg_pnl': f'{metric_info["short_avg_pnl"]:,.4%}',
            'gain_loss_ratio': f'{metric_info["gain_loss_ratio"]:,.3%}'
        }

        info_str = (f'Trade Metrics Report:'
                    f'\n'
                    f'{pd.Series(fmt_dict).to_string()}'
                    f'\n'
                    f'{self.info.to_string()}')

        return info_str
