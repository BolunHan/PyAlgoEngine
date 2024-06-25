import argparse
import datetime
import pathlib
from threading import Thread

from .doc_server import CandleStick
from .. import LOGGER
from ..bokeh_server import DocManager, DocServer
from ...profile import Profile, PROFILE


class WebApp(object):
    def __init__(self, start_date: datetime.date, end_date: datetime.date, name: str = 'WebApp.Backtest', address: str = '0.0.0.0', port: int = 8080, profile: Profile = None, **kwargs):
        from flask import Flask
        self.start_date = start_date
        self.end_date = end_date
        self.name = name
        self.root_dir = pathlib.Path(__file__).parent
        self.profile = PROFILE if profile is None else profile
        self.host = address
        self.port = port

        self.flask = Flask(
            import_name=self.name,
            template_folder=self.root_dir.joinpath('templates'),
            static_folder=self.root_dir.joinpath('static')
        )
        self.doc_manager = DocManager(host='localhost', port=port)
        self.dashboard: dict[str, dict[str, DocServer]] = {}

    def update(self, **kwargs):
        for doc_server in self.doc_manager.doc_server.values():
            doc_server.update(**kwargs)

    def register(self, ticker: str, **kwargs):
        if ticker in self.dashboard:
            raise ValueError(f'Ticker {ticker} already registered.')

        dashboard = self.dashboard[ticker] = {}
        candlestick = dashboard[f'candlesticks'] = CandleStick(ticker=ticker, start_date=self.start_date, end_date=self.end_date, **kwargs)

        self.doc_manager.register(url=f'/candlesticks/{ticker}', doc_server=candlestick)

    def render_index(self):
        from flask import render_template

        dashboard_url = {ticker: f'{self.url}/{ticker}' for ticker in self.dashboard}

        html = render_template(
            'index.html',
            title=f'PyAlgoEngine.Backtest.App',
            data=dashboard_url
        )

        return html

    def render_dashboard(self, ticker: str):
        from flask import render_template
        from bokeh.embed import server_document

        dashboard = self.dashboard[ticker]
        bokeh_scripts = {}

        for name, doc_server in dashboard.items():
            url = self.doc_manager.doc_url[doc_server]
            doc_script = server_document(url=f'http://{self.doc_manager.bokeh_host}:{self.doc_manager.bokeh_port}{url}')
            bokeh_scripts[name] = doc_script

        html = render_template(
            'dash.html',
            ticker=ticker,
            framework="flask",
            **bokeh_scripts
        )
        return html

    def serve(self, blocking: bool = True):
        from waitress import serve

        LOGGER.info(f'starting {self} service...')

        self.doc_manager.start()
        self.flask.route(rule='/', methods=["GET"])(self.render_index)

        for ticker in self.dashboard:
            def renderer():
                return self.render_dashboard(ticker=ticker)

            self.flask.route(rule=f'/{ticker}', methods=["GET"])(renderer)

        if blocking:
            return serve(app=self.flask, host=self.host, port=self.port)

        t = Thread(target=serve, kwargs=dict(app=self.flask, host=self.host, port=self.port))
        t.start()

        # a monkey patch to resolve flask double logging issues
        for hdl in (logger := self.flask.logger).handlers:
            logger.removeHandler(hdl)

        for hdl in (logger := LOGGER.root).handlers:
            logger.removeHandler(hdl)

    @property
    def url(self) -> str:
        if self.host == '0.0.0.0':
            return f'http://localhost:{self.port}/'
        else:
            return f'http://{self.host}:{self.port}/'


def start_app(start_date: datetime.date, end_date: datetime.date, blocking: bool = True, **kwargs):
    web_app = WebApp(start_date=start_date, end_date=end_date, **kwargs)
    web_app.serve(blocking=blocking)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start Backtest.App')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')

    args = parser.parse_args()

    start_app(
        start_date=datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date(),
        end_date=datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date(),
    )
