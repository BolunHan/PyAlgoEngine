import abc
import uuid
from copy import deepcopy
from functools import partial
from threading import Thread, Lock
from typing import overload

from bokeh.document import Document
from bokeh.models import ColumnDataSource
from bokeh.server.server import Server

from . import LOGGER
from ..base import MarketData


class DocTheme(object, metaclass=abc.ABCMeta):
    pass


class DocServer(object, metaclass=abc.ABCMeta):
    def __init__(self, theme: DocTheme = None, max_size: int = None, update_interval: float = 0., lock: Lock = None, **kwargs):
        self.theme: DocTheme = theme
        self.max_size: int = max_size
        self.update_interval: float = update_interval
        self.lock = Lock() if lock is None else lock

        self.bokeh_documents: dict[int, Document] = {}
        self.bokeh_source: dict[int, ColumnDataSource] = {}
        self.bokeh_data_queue: dict[int, dict[str, list]] = {}  # this is a dict of deepcopy of self.data, to update each documents
        self.data: dict[str, list] = dict()

    def __str__(self):
        return f'<{self.__class__.__name__}>(id={id(self.__class__)})'

    def __call__(self, doc: Document):
        self.register_document(doc=doc)

    def __hash__(self):
        return id(self)

    @overload
    def update(self, timestamp: float, market_price: float, **kwargs):
        ...

    @overload
    def update(self, timestamp: float, open_price: float, close_price: float, high_price: float, low_price: float, **kwargs):
        ...

    @overload
    def update(self, market_data: MarketData, **kwargs):
        ...

    @abc.abstractmethod
    def update(self, **kwargs):
        ...

    @abc.abstractmethod
    def layout(self, doc_id: int):
        ...

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

        LOGGER.debug(f'{self.__class__} stream updated!')

    def register_document(self, doc: Document):
        self.lock.acquire()

        doc_id = uuid.uuid4().int

        self.bokeh_documents[doc_id] = doc
        self.bokeh_data_queue[doc_id] = {key: [] for key in self.data}
        self.bokeh_source[doc_id] = ColumnDataSource(data=deepcopy(self.data))

        self.layout(doc_id=doc_id)

        if self.update_interval:
            doc.add_periodic_callback(callback=partial(self.stream, doc_id=doc_id), period_milliseconds=int(self.update_interval * 1000))

        doc.on_session_destroyed(partial(self._unregister_document, doc_id=doc_id))

        LOGGER.info(f'{self} registered Bokeh document id = {doc_id}!')
        self.lock.release()

    def _unregister_document(self, session_context, doc_id: int):
        self.lock.acquire()
        LOGGER.info(f'Session {doc_id} disconnected!')

        self.bokeh_documents.pop(doc_id)
        self.bokeh_source.pop(doc_id)
        self.bokeh_data_queue.pop(doc_id)
        self.lock.release()


class DocManager(object):
    def __init__(self, host: str = 'localhost', port: int = 21543, **kwargs):
        self.host = host
        self.port = port

        self.bokeh_host = kwargs.get('bokeh_host', 'localhost')
        self.bokeh_port = kwargs.get('bokeh_port', 5006)
        self.bokeh_check_unused_sessions = kwargs.get('bokeh_check_unused_sessions', 1)

        self.doc_server: dict[str, DocServer] = {}
        self.doc_url: dict[DocServer, str] = {}
        self.bokeh_thread = Thread(target=self.serve_bokeh, daemon=True)

    def __getitem__(self, url: str):
        return self.doc_server.__getitem__(url)

    def __setitem__(self, url: str, doc_server: DocServer):
        return self.register(url=url, doc_server=doc_server)

    def __contains__(self, url: str):
        return self.doc_server.__contains__(url)

    def register(self, url: str, doc_server: DocServer):
        if url in self.doc_server:
            LOGGER.warning(f'{url} already registered! Existed doc_server {self.doc_server[url]} overridden!')

        self.doc_server[url] = doc_server
        self.doc_url[doc_server] = url
        return doc_server

    def serve_bokeh(self):
        server = Server(
            applications=self.doc_server,
            address=self.bokeh_host,
            port=self.bokeh_port,
            check_unused_sessions_milliseconds=(self.bokeh_check_unused_sessions * 1000),
            allow_websocket_origin=[f"{self.bokeh_host}:{self.bokeh_port}", f"{self.host}:{self.port}"],
            # num_procs=1
        )

        LOGGER.info(
            f'bokeh service started at {self.bokeh_host}:{self.bokeh_port}!\n' +
            '\n'.join([f'http://{self.bokeh_host}:{self.bokeh_port}{url} => {app}' for url, app in self.doc_server.items()])
        )

        server.start()

        # for url in applications:
        #     server.io_loop.add_callback(server.show, url)

        server.io_loop.start()

    def start(self):
        self.bokeh_thread.start()
