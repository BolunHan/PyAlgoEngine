import datetime
import pathlib
from functools import partial
from typing import TypedDict, NotRequired

import pandas as pd

from .. import DocServer, DocTheme
from ...base import MarketData, TradeData, TransactionData
from ...profile import Profile, PROFILE
from ...utils import ts_indices


class StickTheme(DocTheme):
    stick_padding = 0.1
    range_padding = 0.01

    ColorStyle = TypedDict('ColorStyle', fields={'up': str, 'down': str})
    ws_style = ColorStyle(up="green", down="red")
    cn_style = ColorStyle(up="red", down="green")

    def __init__(self, profile: Profile = PROFILE, style: ColorStyle = None):
        self.profile = profile

        if style is None:
            if profile.profile_id == 'cn':
                self.style = self.cn_style
            else:
                self.style = self.ws_style
        else:
            self.style = style

    def stick_style(self, pct_change: float | int) -> dict:
        style_dict = dict()

        if pct_change > 0:
            style_dict['stick_color'] = self.style['up']
        else:
            style_dict['stick_color'] = self.style['down']

        return style_dict


class CandleStick(DocServer):
    class ActiveBarData(TypedDict):
        idx: int
        ts_start: float
        ts_end: float
        open_price: float
        close_price: float
        high_price: float
        low_price: float
        volume: NotRequired[float]

    def __init__(self, ticker: str, start_date: datetime.date, end_date: datetime.date, profile: Profile = PROFILE, interval: float = 60., x_axis: list[float] = None, theme: DocTheme = None, **kwargs):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.profile = profile
        self.interval = interval
        self.indices = self.ts_indices() if x_axis is None else x_axis

        assert self.indices, 'Must assign x_axis to render candlesticks!'

        super().__init__(
            theme=theme,
            max_size=kwargs.get('max_size'),
            update_interval=kwargs.get('update_interval', 0),
        )

        self.theme = StickTheme(profile=self.profile) if self.theme is None else self.theme
        self.timestamp: float = 0.
        self.active_bar_data: CandleStick.ActiveBarData | None = None
        self._data = {
            'index': [],
            'market_time': [],
            'open_price': [],
            'cs.high_price': [],
            'cs.low_price': [],
            'close_price': [],
            'volume': [],
            '_max_price': [],
            '_min_price': [],
            'stick_color': []
        }

    def ts_indices(self) -> list[float]:
        """generate integer indices
        from start date to end date, with given interval, in seconds
        """

        calendar = self.profile.trade_calendar(start_date=self.start_date, end_date=self.end_date)
        timestamps = []
        for market_date in calendar:
            _ts_indices = ts_indices(
                market_date=market_date,
                interval=self.interval,
                session_start=self.profile.session_start,
                session_end=self.profile.session_end,
                session_break=self.profile.session_break,
                time_zone=self.profile.time_zone,
                ts_mode='both'
            )

            timestamps.extend(_ts_indices)

        return timestamps

    def loc_indices(self, timestamp: float, start_idx: int = 0) -> tuple[int, float]:
        last_idx = idx = start_idx

        while idx < len(self.indices):
            ts = self.indices[idx]

            if ts > timestamp:
                break

            last_idx = idx
            idx += 1

        return last_idx, self.indices[last_idx]

    def update(self, **kwargs):
        self.lock.acquire()

        if 'market_data' in kwargs:
            market_data: MarketData = kwargs['market_data']

            if market_data.ticker != self.ticker:
                return

            if isinstance(market_data, (TradeData, TransactionData)):
                self._on_obs(timestamp=market_data.timestamp, price=market_data.price, volume=market_data.volume)
            else:
                self._on_obs(timestamp=market_data.timestamp, price=market_data.market_price)
            self.timestamp = market_data.timestamp
        else:
            kwargs = kwargs.copy()
            timestamp = kwargs.pop('timestamp', self.timestamp)
            ticker = kwargs.pop('ticker')
            price = kwargs.pop('market_price', kwargs.pop('close_price'))
            volume = kwargs.pop('volume', 0)

            assert ticker is not None, 'Must assign a ticker for update function!'
            assert price is not None, f'Must assign a market_price or close_price for {self.__class__} update function!'

            if ticker != self.ticker:
                return

            self._on_obs(timestamp=timestamp, price=price, volume=volume, **kwargs)
            self.timestamp = timestamp

        self.lock.release()

    def _on_obs(self, timestamp: float, price: float, volume: float = 0., **kwargs):
        open_price = kwargs.get('open_price', price)
        high_price = kwargs.get('high_price', price)
        low_price = kwargs.get('low_price', price)

        if self.active_bar_data is None:
            int_idx, ts_idx = self.loc_indices(timestamp=timestamp, start_idx=0)
            if timestamp < ts_idx:
                return

            self.active_bar_data = self.ActiveBarData(
                idx=int_idx,
                ts_start=ts_idx,
                ts_end=ts_idx + self.interval,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=price,
                volume=volume
            )
        elif timestamp <= self.active_bar_data['ts_end']:
            if 'open_price' in kwargs:
                self.active_bar_data['open_price'] = open_price

            self.active_bar_data['high_price'] = max(high_price, self.active_bar_data['high_price'])
            self.active_bar_data['low_price'] = min(low_price, self.active_bar_data['low_price'])
            self.active_bar_data['close_price'] = price

            self.active_bar_data['volume'] += volume

        if timestamp >= self.active_bar_data['ts_end']:
            self.pipe(sequence=self.data)

            for doc_id in list(self.bokeh_documents):
                doc = self.bokeh_documents[doc_id]
                new_data = self.bokeh_data_pipe[doc_id]

                self.pipe(sequence=new_data)

                if not self.update_interval:
                    doc.add_next_tick_callback(partial(self.stream, doc_id=doc_id))

            int_idx, ts_idx = self.loc_indices(timestamp=timestamp, start_idx=self.active_bar_data['idx'])
            self.active_bar_data['idx'] = int_idx
            self.active_bar_data['ts_start'] = ts_idx
            self.active_bar_data['ts_end'] = ts_idx + self.interval
            self.active_bar_data['open_price'] = price
            self.active_bar_data['close_price'] = price
            self.active_bar_data['high_price'] = price
            self.active_bar_data['low_price'] = price
            self.active_bar_data['volume'] = volume

    def pipe(self, sequence: dict[str, list]):
        sequence['index'].append(self.active_bar_data['idx'] + 0.5)  # to ensure bar rendered in the center of the interval
        sequence['market_time'].append(datetime.datetime.fromtimestamp(self.active_bar_data['ts_start'], tz=self.profile.time_zone))
        sequence['open_price'].append(self.active_bar_data['open_price'])
        sequence['close_price'].append(self.active_bar_data['close_price'])
        sequence['cs.high_price'].append(self.active_bar_data['high_price'])
        sequence['cs.low_price'].append(self.active_bar_data['low_price'])
        sequence['volume'].append(self.active_bar_data['volume'])
        sequence['_max_price'].append(max(self.active_bar_data['open_price'], self.active_bar_data['close_price']))
        sequence['_min_price'].append(min(self.active_bar_data['open_price'], self.active_bar_data['close_price']))
        sequence['stick_color'].append(self.theme.stick_style(self.active_bar_data['close_price'] - self.active_bar_data['open_price'])['stick_color'])

    def layout(self, doc_id: int):
        self._register_candlestick(doc_id=doc_id)

    def _register_candlestick(self, doc_id: int):
        from bokeh.models import PanTool, WheelPanTool, WheelZoomTool, BoxZoomTool, ResetTool, ExamineTool, SaveTool, CrosshairTool, HoverTool, RangeTool, Range1d
        from bokeh.plotting import figure, gridplot

        doc = self.bokeh_documents[doc_id]
        source = self.bokeh_data_source[doc_id]

        tools = [
            PanTool(dimensions="width", syncable=False),
            WheelPanTool(dimension="width", syncable=False),
            BoxZoomTool(dimensions="auto", syncable=False),
            WheelZoomTool(dimensions="width", syncable=False),
            CrosshairTool(dimensions="both", syncable=False),
            HoverTool(mode='vline', syncable=False, formatters={'@market_time': 'datetime'}),
            ExamineTool(syncable=False),
            ResetTool(syncable=False),
            SaveTool(syncable=False)
        ]

        tooltips = [
            ("market_time", "@market_time{%H:%M:%S}"),
            ("close_price", "@close_price"),
            ("open_price", "@open_price"),
            ("high_price", "@{cs.high_price}"),
            ("low_price", "@{cs.low_price}"),
        ]

        plot = figure(
            title=f"{self.ticker} Candlestick",
            x_range=Range1d(start=0, end=len(self.indices), bounds='auto'),
            x_axis_type="linear",
            # sizing_mode="stretch_both",
            min_height=80,
            tools=tools,
            tooltips=tooltips,
            y_axis_location="right",
        )

        _shadows = plot.segment(
            name='candlestick.shade',
            x0='index',
            x1='index',
            y0='cs.low_price',
            y1='cs.high_price',
            line_width=1,
            color="black",
            alpha=0.8,
            source=source
        )

        _candlestick = plot.vbar(
            name='candlestick',
            x='index',
            top='_max_price',
            bottom='_min_price',
            width=1 - self.theme.stick_padding,
            color='stick_color',
            alpha=0.5,
            source=source
        )

        plot.xaxis.major_label_overrides = {i: datetime.datetime.fromtimestamp(ts, tz=self.profile.time_zone).strftime('%Y-%m-%d %H:%M:%S') for i, ts in enumerate(self.indices)}
        plot.xaxis.ticker.min_interval = 1.
        tools[5].renderers = [_candlestick]

        range_selector = figure(
            y_range=plot.y_range,
            min_height=20,
            tools=[],
            toolbar_location=None,
            # sizing_mode="stretch_both"
        )

        range_tool = RangeTool(x_range=plot.x_range)
        range_tool.overlay.fill_alpha = 0.5

        range_selector.line('index', 'close_price', source=source)
        range_selector.add_tools(range_tool)
        range_selector.x_range.range_padding = self.theme.range_padding
        range_selector.xaxis.visible = False
        range_selector.xgrid.visible = False
        range_selector.ygrid.visible = False

        root = gridplot(
            children=[
                [plot],
                [range_selector]
            ],
            sizing_mode="stretch_both",
            merge_tools=True,
            toolbar_options={
                'autohide': True,
                'active_drag': tools[0],
                'active_scroll': tools[3]
            },
        )
        root.rows = ['80%', '20%']
        root.width_policy = 'max'
        root.height_policy = 'max'

        doc.add_root(root)

    def to_csv(self, filename: str | pathlib.Path):
        df = pd.DataFrame(self.data).set_index(keys='market_time')
        df = df[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]
        df.to_csv(filename)

    @property
    def data(self) -> dict[str, list]:
        return self._data
