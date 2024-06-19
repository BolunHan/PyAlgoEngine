import pathlib
from threading import Thread

from flask import Flask, render_template
from tornado.ioloop import IOLoop

from .doc_server import DocServer
from . import LOGGER


class DocManager(object):
    def __init__(self, port: int = 21543, io_loop: IOLoop = None, **kwargs):
        self.port = port
        self.io_loop = IOLoop() if io_loop is None else io_loop

        self.doc_server: dict[str, DocServer] = {}
        self.bokeh_address = kwargs.get('bokeh_address', 'localhost')
        self.bokeh_port = kwargs.get('bokeh_port', 5006)
        self.bokeh_thread = Thread(target=self.serve_bokeh, daemon=True)

    def register(self, ticker: str, doc_server: DocServer = None, url: str = None, **kwargs):
        if url is None:
            url = f'/{ticker}'

        if doc_server is None:
            doc_server = DocServer(ticker=ticker, url=url, **kwargs)

        self.doc_server[ticker] = doc_server
        return doc_server

    def serve_bokeh(self):
        from bokeh.server.server import Server

        applications = {doc_server.url: doc_server.__call__ for doc_server in self.doc_server.values()}
        server = Server(
            applications=applications,
            # io_loop=self.io_loop,
            address='localhost',
            port=self.bokeh_port,
            allow_websocket_origin=[f"localhost:{self.bokeh_port}", f"127.0.0.1:{self.bokeh_port}", f"localhost:{self.port}", f"127.0.0.1:{self.port}"],
            num_procs=1
        )

        LOGGER.info(f'bokeh service started at {self.bokeh_address}:{self.bokeh_port}!\n{applications}')

        server.start()

        # for url in applications:
        #     server.io_loop.add_callback(server.show, url)

        server.io_loop.start()

    def start(self):
        self.bokeh_thread.start()


app = Flask('Backtest.WebApp')
DOC_MANAGER = DocManager()


def index():
    from bokeh.embed import server_document

    embedded_docs = []

    for doc_server in DOC_MANAGER.doc_server.values():
        doc_script = server_document(f'http://localhost:{DOC_MANAGER.bokeh_port}{doc_server.url}')
        embedded_docs.append(doc_script)

    return render_template(
        'index.html',
        script=embedded_docs[0],
        template="Flask"
    )


def start_app():
    DOC_MANAGER.start()

    app.route('/')(index)
    app.run(port=DOC_MANAGER.port)


if __name__ == '__main__':
    start_app()
