from threading import Thread

from bokeh.embed import server_document
from flask import Flask, render_template
from waitress import serve

from .doc_server import CandleStick
from ..bokeh_server import DocManager
from .. import LOGGER


class WebApp(object):
    def __init__(self, name: str = 'WebApp.Backtest', address: str = '0.0.0.0', port: int = 8080, **kwargs):
        self.name = name

        self.host = address
        self.port = port

        self.flask = Flask(import_name=name)
        self.doc_manager = DocManager(host='localhost', port=port)
        self.candle_sticks = {}

    def update(self, **kwargs):
        for doc_server in self.doc_manager.doc_server.values():
            doc_server.update(**kwargs)

    def register(self, ticker: str, **kwargs):
        self.candle_sticks[ticker] = candlestick = CandleStick(ticker=ticker, url=f'/candlesticks/{ticker}', **kwargs)

        self.doc_manager.register(doc_server=candlestick)

    def index(self):
        embedded_docs = []

        for doc_server in self.doc_manager.doc_server.values():
            doc_script = server_document(url=f'http://{self.doc_manager.bokeh_host}:{self.doc_manager.bokeh_port}{doc_server.url}')
            embedded_docs.append(doc_script)

        html = render_template(
            'index.html',
            script=embedded_docs[0],
            template="Flask"
        )

        return html

    def serve(self, blocking: bool = True):
        LOGGER.info(f'starting {self} service...')

        self.doc_manager.start()
        self.flask.route(rule='/', methods=["GET"])(self.index)

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


def start_app(blocking: bool = True):
    web_app = WebApp()
    web_app.serve(blocking=blocking)


if __name__ == '__main__':
    start_app()
