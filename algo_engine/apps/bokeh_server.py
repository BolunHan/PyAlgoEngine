import abc
import uuid
from copy import deepcopy
from functools import partial
from threading import Thread, Lock, Event
from typing import overload

from . import LOGGER
from ..base import MarketData


class DocTheme(object, metaclass=abc.ABCMeta):
    pass


class DocServer(object, metaclass=abc.ABCMeta):
    def __init__(self, theme: DocTheme = None, max_size: int = None, update_interval: float = 0., lock: Lock = None, **kwargs):
        from bokeh.document import Document
        from bokeh.models import ColumnDataSource

        self.theme: DocTheme = theme
        self.max_size: int = max_size
        self.update_interval: float = update_interval
        self.lock = Lock() if lock is None else lock

        self.bokeh_documents: dict[int, Document] = {}
        # self.bokeh_source: dict[int, ColumnDataSource] = {}
        self.bokeh_data_pipe: dict[int, dict[str, list[...]]] = {}
        self.bokeh_data_patch: dict[int, dict[str, list[tuple[int, ...]]]] = {}
        self.bokeh_data_source: dict[int, ColumnDataSource] = {}

    def __str__(self):
        return f'<{self.__class__.__name__}>(id={id(self.__class__)})'

    def __call__(self, doc):
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

        # doc = self.bokeh_documents[doc_id]
        data_pipe = self.bokeh_data_pipe[doc_id]
        source = self.bokeh_data_source[doc_id]

        source.stream(new_data=deepcopy(data_pipe), rollover=self.max_size)
        for key, seq in data_pipe.items():
            seq.clear()

        LOGGER.debug(f'{self.__class__} <stream> updated!')

    def patch(self, doc_id: int = None):
        if doc_id is None:
            for doc_id in list(self.bokeh_documents):
                self.patch(doc_id=doc_id)
            return

        # doc = self.bokeh_documents[doc_id]
        data_patch = self.bokeh_data_patch[doc_id]
        source = self.bokeh_data_source[doc_id]

        source.patch(patches=deepcopy(data_patch))
        for key, seq in data_patch.items():
            seq.clear()

        LOGGER.debug(f'{self.__class__} <patch> updated!')

    def register_document(self, doc):
        from bokeh.models import ColumnDataSource

        self.lock.acquire()

        doc_id = uuid.uuid4().int

        data = deepcopy(self.data)
        self.bokeh_documents[doc_id] = doc
        self.bokeh_data_pipe[doc_id] = {key: [] for key in data}
        self.bokeh_data_patch[doc_id] = {key: [] for key in data}
        self.bokeh_data_source[doc_id] = ColumnDataSource(data=data)

        self.layout(doc_id=doc_id)

        if self.update_interval:
            doc.add_periodic_callback(callback=partial(self.stream, doc_id=doc_id), period_milliseconds=int(self.update_interval * 1000))
            doc.add_periodic_callback(callback=partial(self.patch, doc_id=doc_id), period_milliseconds=int(self.update_interval * 1000))

        doc.on_session_destroyed(partial(self._unregister_document, doc_id=doc_id))

        LOGGER.info(f'{self} registered Bokeh document id = {doc_id}!')
        self.lock.release()

    def _unregister_document(self, session_context, doc_id: int):
        self.lock.acquire()
        LOGGER.info(f'Session {doc_id} disconnected!')

        self.bokeh_documents.pop(doc_id)
        self.bokeh_data_pipe.pop(doc_id)
        self.bokeh_data_patch.pop(doc_id)
        self.bokeh_data_source.pop(doc_id)
        self.lock.release()

    @property
    @abc.abstractmethod
    def data(self) -> dict[str, list]:
        """
        the data used to provide initial values for new bokeh.ColumnDataSource.
        """
        ...


class DocManager(object):
    def __init__(self, host: str = 'localhost', port: int = 21543, **kwargs):
        self.host = host
        self.port = port

        self.bokeh_host = kwargs.get('bokeh_host', 'localhost')
        self.bokeh_port = kwargs.get('bokeh_port', 5006)
        self.bokeh_check_unused_sessions = kwargs.get('bokeh_check_unused_sessions', 1)

        self.doc_server: dict[str, DocServer] = {}
        self.doc_url: dict[DocServer, str] = {}
        self.bokeh_thread = Thread(target=self.serve_bokeh)
        self.stop_event = Event()
        self.bokeh_server = None

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
        from bokeh.server.server import Server
        import psutil
        import socket

        # Get all network interfaces and their IP addresses
        addrs = psutil.net_if_addrs()
        websocket_origin = [
            f"{self.host}:{self.port}"
        ]

        for interface, addr_info in addrs.items():
            for addr in addr_info:
                if addr.family == socket.AF_INET:  # Filter only IPv4 addresses
                    LOGGER.info(f"Binding network interface: {interface}, IP Address: {addr.address}")
                    websocket_origin.append(f"{addr.address}:{self.port}")
                    websocket_origin.append(f"{addr.address}:{self.bokeh_port}")

        if (_ := f'localhost:{self.port}') not in websocket_origin:
            websocket_origin.append(_)
        if (_ := f'127.0.0.1:{self.port}') not in websocket_origin:
            websocket_origin.append(_)
        if (_ := f'localhost:{self.bokeh_port}') not in websocket_origin:
            websocket_origin.append(_)
        if (_ := f'127.0.0.1:{self.bokeh_port}') not in websocket_origin:
            websocket_origin.append(_)
        if (_ := f'{self.bokeh_host}:{self.port}') not in websocket_origin:
            websocket_origin.append(_)
        if (_ := f'{self.bokeh_host}:{self.bokeh_port}') not in websocket_origin:
            websocket_origin.append(_)

        self.bokeh_server = Server(
            applications=self.doc_server,
            address='0.0.0.0',
            port=self.bokeh_port,
            check_unused_sessions_milliseconds=(self.bokeh_check_unused_sessions * 1000),
            allow_websocket_origin=websocket_origin,
            use_xheaders=True,
            # num_procs=1
        )

        LOGGER.info(
            f'bokeh service started at {self.bokeh_host}:{self.bokeh_port}!\n' +
            '\n'.join([f'http://{self.bokeh_host}:{self.bokeh_port}{url} => {app}' for url, app in self.doc_server.items()])
        )

        self.bokeh_server.start()

        # Start the Bokeh server IOLoop unless stop_event is triggered
        while not self.stop_event.is_set():
            try:
                self.bokeh_server.io_loop.start()
            except Exception as e:
                LOGGER.error(f"Error in Bokeh server: {e}")
            finally:
                LOGGER.info("Bokeh server has been stopped.")
                break

    def start(self):
        LOGGER.info(f'Starting Bokeh service...')
        self.bokeh_thread.start()

    def stop(self):
        LOGGER.info(f'Stopping Bokeh service...')

        # Signal the event to stop
        self.stop_event.set()

        # Stop the Bokeh server gracefully
        if self.bokeh_server is not None:
            self.bokeh_server.io_loop.stop()  # Stop the IOLoop
            self.bokeh_server.stop()  # Stop the Bokeh server itself

        # Wait for the Bokeh thread to finish if it's still running
        if self.bokeh_thread.is_alive():
            self.bokeh_thread.join()

        LOGGER.info('Bokeh service has been stopped.')
