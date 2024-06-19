import datetime
import pathlib
import uuid
from copy import deepcopy
from functools import partial
from typing import overload, TypedDict, NotRequired

import pandas as pd
from bokeh.document import Document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure

from . import LOGGER
from ...base import MarketData, TradeData, TransactionData


class ActiveBarData(TypedDict):
    ts_start: float
    ts_end: float
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: NotRequired[float]


class ClassicTheme(object):
    stick_padding = 0.1

    @classmethod
    def stick_style(cls, pct_change: float) -> dict:
        style_dict = dict()

        if pct_change > 0:
            style_dict['stick_color'] = "#49a3a3"
        else:
            style_dict['stick_color'] = "#eb3c40"

        return style_dict


class DocServer:
    def __init__(self, ticker: str, url: str = None, interval: float = 60., themes: ClassicTheme = None, max_size: int = None, **kwargs):
        self.ticker = ticker
        self.url = f'/{ticker}' if url is None else url
        self.interval = interval
        self.themes = ClassicTheme() if themes is None else themes
        self.max_size = max_size

        self.timestamp: float = 0.
        self.active_bar_data: ActiveBarData | None = None
        self.bokeh_documents: dict[int, Document] = {}
        self.bokeh_source: dict[int, ColumnDataSource] = {}
        self.bokeh_data_queue: dict[int, dict[str, list]] = {}  # this is a dict of deepcopy of self.data, to update each documents
        self.bokeh_update_interval = kwargs.get('bokeh_update_interval', 0)
        self.data: dict[str, list] = dict(
            market_time=[],
            open_price=[],
            high_price=[],
            low_price=[],
            close_price=[],
            volume=[],
            _max_price=[],
            _min_price=[],
            stick_color=[]
        )

    def __str__(self):
        return f'<{self.__class__.__name__}>(id={id(self.__class__)}, url={self.url})'

    def __call__(self, doc: Document):
        self.register_document(doc=doc)

    @overload
    def update(self, timestamp: float, market_price: float, **kwargs):
        ...

    @overload
    def update(self, timestamp: float, open_price: float, close_price: float, high_price: float, low_price: float, **kwargs):
        ...

    @overload
    def update(self, market_data: MarketData, **kwargs):
        ...

    def update(self, **kwargs):
        if 'market_data' in kwargs:
            market_data: MarketData = kwargs['market_data']

            if isinstance(market_data, (TradeData, TransactionData)):
                self._on_obs(timestamp=market_data.timestamp, price=market_data.price, volume=market_data.volume)
            else:
                self._on_obs(timestamp=market_data.timestamp, price=market_data.market_price)
        else:
            kwargs = kwargs.copy()
            timestamp = kwargs.pop('timestamp', self.timestamp)
            price = kwargs.pop('market_price', kwargs.pop('close_price'))
            volume = kwargs.pop('volume', 0)

            assert price is not None, f'Must assign a market_price or close_price for {self.__class__} update function!'

            self._on_obs(timestamp=timestamp, price=price, volume=volume, **kwargs)

    def _on_obs(self, timestamp: float, price: float, volume: float = 0., **kwargs):
        open_price = kwargs.get('open_price', price)
        high_price = kwargs.get('high_price', price)
        low_price = kwargs.get('low_price', price)

        if self.active_bar_data is None:
            ts_index = timestamp // self.interval * self.interval

            self.active_bar_data = ActiveBarData(
                ts_start=ts_index,
                ts_end=ts_index + self.interval,
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
            self.pipe()

            ts_index = timestamp // self.interval * self.interval
            self.active_bar_data['ts_start'] = ts_index
            self.active_bar_data['ts_end'] = ts_index + self.interval
            self.active_bar_data['open_price'] = price
            self.active_bar_data['close_price'] = price
            self.active_bar_data['high_price'] = price
            self.active_bar_data['low_price'] = price
            self.active_bar_data['volume'] = volume

    def pipe(self):
        for doc_id in list(self.bokeh_documents):
            doc = self.bokeh_documents[doc_id]
            new_data = self.bokeh_data_queue[doc_id]
            source = self.bokeh_source[doc_id]

            new_data['market_time'].append(datetime.datetime.fromtimestamp(self.active_bar_data['ts_start']))
            new_data['open_price'].append(self.active_bar_data['open_price'])
            new_data['close_price'].append(self.active_bar_data['close_price'])
            new_data['high_price'].append(self.active_bar_data['high_price'])
            new_data['low_price'].append(self.active_bar_data['low_price'])
            new_data['volume'].append(self.active_bar_data['volume'])
            new_data['_max_price'].append(max(self.active_bar_data['open_price'], self.active_bar_data['close_price']))
            new_data['_min_price'].append(min(self.active_bar_data['open_price'], self.active_bar_data['close_price']))
            new_data['stick_color'].append(self.themes.stick_style(self.active_bar_data['close_price'] - self.active_bar_data['open_price'])['stick_color'])

            if not self.bokeh_update_interval:
                doc.add_next_tick_callback(partial(self.stream, doc_id=doc_id))

    def stream(self, doc_id: int = None):
        if doc_id is None:
            for doc_id in list(self.bokeh_documents):
                self.stream(doc_id=doc_id)
            return

        doc = self.bokeh_documents[doc_id]
        new_data = self.bokeh_data_queue[doc_id]
        source = self.bokeh_source[doc_id]

        source.stream(new_data=deepcopy(new_data), rollover=self.max_size)
        for key, seq in new_data.items():
            seq.clear()

        LOGGER.debug(f'{self.__class__} {self.url} stream updated!')

    def register_document(self, doc: Document):
        doc_id = uuid.uuid4().int

        self.bokeh_documents[doc_id] = doc
        self.bokeh_data_queue[doc_id] = {key: [] for key in self.data}
        self.bokeh_source[doc_id] = ColumnDataSource(data=self.data)

        self._register_candlestick(doc_id=doc_id)

        if self.bokeh_update_interval:
            doc.add_periodic_callback(callback=partial(self.stream, doc_id=doc_id), period_milliseconds=int(self.bokeh_update_interval * 1000))

        LOGGER.info(f'{self} registered Bokeh document id = {doc_id}!')

    def _register_candlestick(self, doc_id: int):
        doc = self.bokeh_documents[doc_id]
        source = self.bokeh_source[doc_id]

        plot = figure(
            title=f"{self.ticker} Candlestick",
            x_axis_type="datetime",
            # sizing_mode="stretch_width",
            sizing_mode="stretch_both",
            min_height=80,
            tools="xpan,xwheel_zoom,xbox_zoom,reset",
            y_axis_location="right",
        )

        plot.segment(
            x0='market_time',
            x1='market_time',
            y0='low_price',
            y1='high_price',
            line_width=1,
            color="black",
            alpha=0.8,
            source=source
        )

        # plot.segment(
        #     x0='market_time',
        #     x1='market_time',
        #     y0='open_price',
        #     y1='close_price',
        #     line_width=8,
        #     color='stick_color',
        #     source=source
        # )

        plot.vbar(
            x='market_time',
            top='_max_price',
            bottom='_min_price',
            width=datetime.timedelta(seconds=(1 - self.themes.stick_padding) * self.interval),
            color='stick_color',
            alpha=0.5,
            source=source
        )

        range_selector = figure(
            y_range=plot.y_range,
            x_axis_type="datetime",
            y_axis_type=None,
            min_height=20,
            tools="",
            toolbar_location=None,
            # sizing_mode="stretch_width",
            sizing_mode="stretch_both"
        )

        range_tool = RangeTool(x_range=plot.x_range)
        range_tool.overlay.fill_alpha = 0.2

        range_selector.line('market_time', 'close_price', source=source)
        range_selector.add_tools(range_tool)
        range_selector.x_range.range_padding = 0.01

        layout = column([plot, range_selector], sizing_mode="stretch_both")
        doc.add_root(layout)

    def to_csv(self, filename: str | pathlib.Path):
        df = pd.DataFrame(self.data).set_index(keys='market_time')
        df = df[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]
        df.to_csv(filename)
