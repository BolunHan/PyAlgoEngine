from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.ref cimport Py_XINCREF, Py_XDECREF
from libc.stdlib cimport calloc, realloc, free
from libc.string cimport memcpy, memset

from .c_market_data cimport MarketData, c_md_serialized_size, c_md_deserialize, MD_CFG_LOCKED, MD_CFG_SHARED, MD_CFG_FREELIST


class InvalidBufferError(Exception):
    pass


class NotInSharedMemoryError(Exception):
    pass


class BufferFull(Exception):
    pass


class BufferEmpty(Exception):
    pass


class PipeTimeoutError(Exception):
    pass


class BufferCorruptedError(Exception):
    pass


cdef class MarketDataBufferCache:
    def __cinit__(self, size_t capacity, object parent):
        self.parent = parent
        self.py_array = <PyObject**> calloc(capacity, sizeof(PyObject*))
        self.c_array = <md_variant**> calloc(capacity, sizeof(md_variant*))
        self.capacity = capacity

    def __dealloc__(self):
        cdef size_t i
        if self.py_array:
            for i in range(self.size):
                Py_XDECREF(self.py_array[i])
            free(self.py_array)

        if self.c_array:
            free(self.c_array)

    cdef int c_write_block_buffer(self, MarketDataBuffer buffer):
        # Step 1: Calculate required capacities
        cdef md_variant* md
        cdef size_t i = 0
        cdef size_t data_cap = 0
        cdef size_t ptr_cap = self.size

        if not ptr_cap:
            return 0

        for i in range(ptr_cap):
            md = self.c_array[i]
            data_cap += c_md_serialized_size(md)

        cdef md_block_buffer* header = buffer.header
        ptr_cap += header.ptr_tail
        data_cap += header.data_tail

        # Step 2: Extend buffer if needed
        if data_cap > header.data_capacity or ptr_cap > header.ptr_capacity:
            header = c_md_block_buffer_extend(
                buffer.header,
                ptr_cap,
                data_cap,
                SHM_ALLOCATOR if MD_CFG_SHARED else NULL,
                HEAP_ALLOCATOR if MD_CFG_FREELIST else NULL,
                <int> MD_CFG_LOCKED
            )

            if not header:
                raise MemoryError("Failed to allocate new buffer for MarketDataBuffer")

            if buffer.owner:
                c_md_block_buffer_free(buffer.header, 1)

            buffer.header = header
            buffer.owner = True

        # Step 3: Copy cached data into buffer
        for i in range(self.size):
            md = self.c_array[i]
            c_md_block_buffer_put(header, md)
        return 0

    def __iter__(self):
        cdef size_t i
        for i in range(self.size):
            yield MarketData.c_from_header(self.c_array[i], False)

    def __len__(self):
        return self.size

    def __getitem__(self, ssize_t idx):
        return self.get(idx)

    def __enter__(self):
        self.clear()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if isinstance(self.parent, MarketDataBuffer):
            self.c_write_block_buffer(self.parent)
        self.clear()

    def put(self, MarketData market_data):
        cdef PyObject** py_array = self.py_array
        cdef md_variant** c_array = self.c_array
        cdef size_t capacity = self.capacity

        if self.size >= capacity:
            capacity *= 2

            py_array = <PyObject**> realloc(py_array, capacity * sizeof(PyObject*))
            if not py_array:
                raise MemoryError("Failed to reallocate MarketDataBufferCache arrays")
            self.py_array = py_array

            c_array = <md_variant**> realloc(c_array, capacity * sizeof(md_variant*))
            if not c_array:
                raise MemoryError("Failed to reallocate MarketDataBufferCache arrays")
            self.c_array = c_array
            self.capacity = capacity

        self.c_array[self.size] = market_data.header
        self.py_array[self.size] = <PyObject*> market_data
        Py_XINCREF(<PyObject*> market_data)
        self.size += 1

    def get(self, ssize_t idx):
        cdef ssize_t total = self.size
        if idx >= total or idx < -total:
            raise IndexError(f'{self.__class__.__name__} index {idx} out of range {total}')
        if idx < 0:
            idx += total
        cdef md_variant* md = self.c_array[idx]
        return MarketData.c_from_header(md, False)

    def clear(self):
        cdef size_t i
        for i in range(self.size):
            Py_XDECREF(self.py_array[i])
        memset(self.py_array, 0, self.capacity * sizeof(PyObject*))
        memset(self.c_array, 0, self.capacity * sizeof(md_variant*))
        self.size = 0


cdef class MarketDataBuffer:
    def __cinit__(self, size_t ptr_cap, size_t data_cap):
        self.header = c_md_block_buffer_new(
            ptr_cap,
            data_cap,
            SHM_ALLOCATOR if MD_CFG_SHARED else NULL,
            HEAP_ALLOCATOR if MD_CFG_FREELIST else NULL,
            <int> MD_CFG_LOCKED
        )
        if not self.header:
            raise MemoryError("Failed to allocate MarketDataBuffer")

        self.owner = True
        self.iter_idx = 0

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_md_block_buffer_free(self.header, 1)

    cdef void c_sort(self):
        cdef int ret_code = c_md_block_buffer_sort(self.header)

        if ret_code == MD_BUF_OK:
            return
        elif ret_code == MD_BUF_ERR_INVALID:
            raise InvalidBufferError('Invalid buffer')
        else:
            raise RuntimeError(f'Failed to sort MarketDataBuffer, error code: {ret_code}')

    cdef void c_put(self, md_variant* market_data):
        cdef md_block_buffer* header = self.header
        cdef size_t ptr_tail = header.ptr_tail
        cdef size_t ptr_capacity = header.ptr_capacity
        cdef size_t data_tail = header.data_tail
        cdef size_t data_capacity = header.data_capacity
        cdef size_t md_size = c_md_serialized_size(market_data)

        if ptr_tail >= ptr_capacity or data_tail + md_size > data_capacity:
            header = c_md_block_buffer_extend(
                self.header,
                max(ptr_capacity * 2, ptr_tail + 1),
                max(data_capacity * 2, data_tail + md_size),
                SHM_ALLOCATOR if MD_CFG_SHARED else NULL,
                HEAP_ALLOCATOR if MD_CFG_FREELIST else NULL,
                <int> MD_CFG_LOCKED if (MD_CFG_SHARED or MD_CFG_FREELIST) else 0
            )
            if not header:
                raise MemoryError("Failed to allocate new buffer for MarketDataBuffer")
            c_md_block_buffer_free(self.header, 1)
            self.header = header

        cdef int ret_code = c_md_block_buffer_put(self.header, market_data)

        if ret_code == MD_BUF_OK:
            return
        elif ret_code == MD_BUF_ERR_INVALID:
            raise ValueError('Invalid args')
        elif ret_code == MD_BUF_ERR_FULL:
            raise BufferFull('Buffer is full')
        else:
            raise RuntimeError(f'Failed to put a MarketData into block buffer, error code: {ret_code}')

    cdef md_variant* c_get(self, ssize_t idx):
        cdef ssize_t total = self.header.ptr_tail
        if idx >= total or idx < -total:
            raise IndexError(f'{self.__class__.__name__} index {idx} out of range {total}')
        if idx < 0:
            idx += total
        cdef const char* blob = c_md_block_buffer_get(self.header, idx)
        cdef md_variant* md = c_md_deserialize(
            blob,
            SHM_ALLOCATOR if MD_CFG_SHARED else NULL,
            HEAP_ALLOCATOR if MD_CFG_FREELIST else NULL,
            <int> MD_CFG_LOCKED if (MD_CFG_SHARED or MD_CFG_FREELIST) else 0
        )
        return md

    cdef void c_clear(self):
        cdef int ret_code = c_md_block_buffer_clear(self.header)

        if ret_code == MD_BUF_OK:
            return
        elif ret_code == MD_BUF_ERR_INVALID:
            raise ValueError('Invalid args')
        else:
            raise RuntimeError(f'Failed to clear MarketDataBuffer, error code: {ret_code}')

    # --- python interface ---

    def __iter__(self):
        self.c_sort()
        self.iter_idx = 0
        return self

    def __getitem__(self, idx: int):
        cdef md_variant* md = self.c_get(idx)
        return MarketData.c_from_header(md, False)

    def __len__(self):
        return self.header.ptr_tail

    def __next__(self):
        if self.iter_idx >= self.header.ptr_tail:
            raise StopIteration

        cdef md_variant* md = self.c_get(self.iter_idx)
        self.iter_idx += 1
        return MarketData.c_from_header(md, False)

    def cache(self):
        return MarketDataBufferCache.__new__(MarketDataBufferCache, 4096, self)

    def put(self, MarketData market_data):
        self.c_put(market_data.header)

    def get(self, idx: int):
        cdef md_variant* md = self.c_get(idx)
        return MarketData.c_from_header(md, False)

    def sort(self):
        c_md_block_buffer_sort(self.header)

    def to_bytes(self):
        cdef size_t serialized_size = c_md_block_buffer_serialized_size(self.header)
        cdef bytes result = PyBytes_FromStringAndSize(NULL, serialized_size)
        c_md_block_buffer_serialize(self.header, <char*> result)
        return result

    @classmethod
    def from_bytes(cls, data: bytes):
        cdef md_block_buffer* src = <md_block_buffer*> <char*> data
        cdef size_t size = len(data)
        cdef size_t ptr_cap = src.ptr_capacity
        cdef size_t data_cap = src.data_capacity
        cdef MarketDataBuffer instance = MarketDataBuffer.__new__(MarketDataBuffer, ptr_cap, data_cap)
        memcpy(instance.header, src, size)
        return instance

    property ptr_capacity:
        def __get__(self):
            return self.header.ptr_capacity

    property ptr_tail:
        def __get__(self):
            return self.header.ptr_tail

    property data_capacity:
        def __get__(self):
            return self.header.data_capacity

    property data_tail:
        def __get__(self):
            return self.header.data_tail

    property is_sorted:
        def __get__(self):
            return <bint> self.header.sorted


cdef class MarketDataRingBuffer:
    def __cinit__(self, size_t ptr_cap, size_t data_cap):
        self.header = c_md_ring_buffer_new(
            ptr_cap,
            data_cap,
            SHM_ALLOCATOR if MD_CFG_SHARED else NULL,
            HEAP_ALLOCATOR if MD_CFG_FREELIST else NULL,
            <int> MD_CFG_LOCKED
        )
        if not self.header:
            raise MemoryError("Failed to allocate MarketDataBuffer")

        self.owner = True
        self.iter_idx = 0

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_md_ring_buffer_free(self.header, 1)

    cdef bint c_is_empty(self):
        if not self.header:
            raise RuntimeError(f'{self.__class__.__name__} uninitialized')
        return c_md_ring_buffer_is_empty(self.header)

    cdef bint c_is_full(self, md_variant* market_data):
        if not self.header:
            raise RuntimeError(f'{self.__class__.__name__} uninitialized')
        return c_md_ring_buffer_is_full(self.header, market_data)

    cdef void c_put(self, md_variant* market_data, bint block=True, double timeout=0.0):
        cdef int ret_code = c_md_ring_buffer_put(self.header, market_data, block, timeout)

        if ret_code == MD_BUF_OK:
            return
        elif ret_code == MD_BUF_ERR_INVALID:
            raise ValueError("Invalid args")
        elif ret_code == MD_BUF_ERR_FULL:
            raise BufferFull('Ring buffer is full')
        elif ret_code == MD_BUF_ERR_TIMEOUT:
            raise PipeTimeoutError('Timeout while putting to ring buffer')
        else:
            raise RuntimeError(f'Failed to put MarketData, error code: {ret_code}')

    cdef md_variant* c_get(self, ssize_t idx):
        cdef ssize_t total = self.header.ptr_tail
        if idx >= total or idx < -total:
            raise IndexError(f'{self.__class__.__name__} index {idx} out of range {total}')
        if idx < 0:
            idx += total
        cdef const char* blob = c_md_ring_buffer_get(self.header, idx)
        cdef md_variant* md = c_md_deserialize(
            blob,
            SHM_ALLOCATOR if MD_CFG_SHARED else NULL,
            HEAP_ALLOCATOR if MD_CFG_FREELIST else NULL,
            <int> MD_CFG_LOCKED if (MD_CFG_SHARED or MD_CFG_FREELIST) else 0
        )
        return md

    cdef md_variant* c_listen(self, bint block=True, double timeout=0):
        cdef const char* blob = NULL
        cdef int ret_code = c_md_ring_buffer_listen(self.header, block, timeout, &blob)
        cdef md_variant* md

        if not ret_code:
            md = c_md_deserialize(
                blob,
                SHM_ALLOCATOR if MD_CFG_SHARED else NULL,
                HEAP_ALLOCATOR if MD_CFG_FREELIST else NULL,
                <int> MD_CFG_LOCKED if (MD_CFG_SHARED or MD_CFG_FREELIST) else 0
            )
            return md

        if ret_code == MD_BUF_ERR_INVALID:
            raise ValueError('Invalid args')
        elif ret_code == MD_BUF_ERR_EMPTY:
            raise BufferEmpty('Empty buffer')
        elif ret_code == MD_BUF_ERR_CORRUPT:
            raise BufferCorruptedError('Corrupted buffer')
        elif ret_code == MD_BUF_ERR_TIMEOUT:
            raise PipeTimeoutError('Timeout')
        else:
            raise RuntimeError(f'Failed to fetch a MarketData from buffer, error code: {ret_code}')

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __getitem__(self, idx: int):
        cdef md_variant* md = self.c_get(idx)
        return MarketData.c_from_header(md, False)

    def __len__(self):
        return c_md_ring_buffer_size(self.header)

    def __next__(self):
        cdef size_t total = c_md_ring_buffer_size(self.header)
        if self.iter_idx >= total:
            raise StopIteration
        cdef md_variant* md = self.c_get(self.iter_idx)
        self.iter_idx += 1
        return MarketData.c_from_header(md, False)

    def put(self, MarketData market_data, bint block=True, double timeout=0.0):
        self.c_put(market_data.header, block, timeout)

    def get(self, idx: int):
        cdef md_variant* md = self.c_get(idx)
        return MarketData.c_from_header(md, False)

    def listen(self, bint block=True, double timeout=0.0):
        cdef md_variant* md = self.c_listen(block, timeout)
        return MarketData.c_from_header(md, False)

    property ptr_capacity:
        def __get__(self):
            return self.header.ptr_capacity

    property ptr_head:
        def __get__(self):
            return self.header.ptr_head

    property ptr_tail:
        def __get__(self):
            return self.header.ptr_tail

    property data_capacity:
        def __get__(self):
            return self.header.data_capacity

    property data_tail:
        def __get__(self):
            return self.header.data_tail

    property is_empty:
        def __get__(self):
            return self.c_is_empty()


cdef class MarketDataConcurrentBuffer:
    def __cinit__(self, size_t n_workers, size_t capacity):
        self.header = c_md_concurrent_buffer_new(
            n_workers,
            capacity,
            SHM_ALLOCATOR,
            <int> MD_CFG_LOCKED
        )
        if not self.header:
            raise MemoryError("Failed to allocate MarketDataBuffer")

        self.owner = True
        self.iter_idx = 0

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_md_concurrent_buffer_free(self.header, 1)

    cdef bint c_is_worker_empty(self, size_t worker_id):
        if not self.header:
            raise RuntimeError(f'{self.__class__.__name__} uninitialized')
        cdef int ret_code = c_md_concurrent_buffer_is_empty(self.header, worker_id)

        if ret_code >= 0:
            return <bint> ret_code
        elif ret_code == MD_BUF_OOR:
            raise IndexError(f'worker_id {worker_id} out of range {self.header.n_workers}')
        elif ret_code == MD_BUF_DISABLED:
            raise ValueError(f'worker {worker_id} disabled')
        else:
            raise RuntimeError(f'c_is_worker_empty failed, err code: {ret_code}')

    cdef bint c_is_empty(self):
        if not self.header:
            raise RuntimeError(f'{self.__class__.__name__} uninitialized')
        cdef size_t worker_id
        cdef md_concurrent_buffer_worker_t* worker
        for worker_id in range(self.header.n_workers):
            worker = self.header.workers + worker_id
            if worker.enabled and not c_md_concurrent_buffer_is_empty(self.header, worker_id):
                return False
        return True

    cdef bint c_is_full(self):
        if not self.header:
            raise RuntimeError(f'{self.__class__.__name__} uninitialized')
        return <bint> c_md_concurrent_buffer_is_full(self.header)

    cdef void c_put(self, md_variant* market_data, bint block=True, double timeout=0):
        if not market_data.meta_info.shm_allocator:
            market_data = c_md_send_to_shm(
                market_data,
                SHM_ALLOCATOR,
                SHM_POOL,
                <int> MD_CFG_LOCKED
            )
        cdef int ret_code = c_md_concurrent_buffer_put(self.header, market_data, block, timeout)

        if ret_code == MD_BUF_OK:
            return
        elif ret_code == MD_BUF_ERR_INVALID:
            raise ValueError("Invalid args")
        elif ret_code == MD_BUF_ERR_NOT_SHM:
            raise NotInSharedMemoryError('Must put a shm backed market data.')
        elif ret_code == MD_BUF_ERR_FULL:
            raise BufferFull(f'{self.__class__.__name__} is full')
        elif ret_code == MD_BUF_ERR_TIMEOUT:
            raise PipeTimeoutError('Timeout while putting to ring buffer')
        else:
            raise RuntimeError(f'Failed to put MarketData, error code: {ret_code}')

    cdef md_variant* c_listen(self, size_t worker_id, bint block=True, double timeout=0):
        cdef md_variant* market_data = NULL
        cdef int ret_code = c_md_concurrent_buffer_listen(
            self.header,
            worker_id,
            block,
            timeout,
            &market_data
        )
        if not ret_code:
            return market_data

        if ret_code == MD_BUF_ERR_INVALID:
            raise ValueError('Invalid args')
        elif ret_code == MD_BUF_OOR:
            raise IndexError(f'Worker_id {worker_id} out of range')
        elif ret_code == MD_BUF_DISABLED:
            raise ValueError(f'Worker {worker_id} disabled')
        elif ret_code == MD_BUF_ERR_EMPTY:
            raise BufferEmpty('Empty buffer')
        elif ret_code == MD_BUF_ERR_TIMEOUT:
            raise PipeTimeoutError('Timeout')
        else:
            raise RuntimeError(f'Failed to fetch a MarketData from buffer, error code: {ret_code}')

    cdef void c_disable_worker(self, size_t worker_id):
        if not self.header:
            raise RuntimeError(f'{self.__class__.__name__} uninitialized')
        cdef int ret_code = c_md_concurrent_buffer_disable_worker(self.header, worker_id)

        if ret_code == MD_BUF_OK:
            return
        elif ret_code == MD_BUF_OOR:
            raise IndexError(f'Worker_id {worker_id} out of range')
        else:
            raise RuntimeError(f'Failed to disable worker, error code: {ret_code}')

    cdef void c_enable_worker(self, size_t worker_id):
        if not self.header:
            raise RuntimeError(f'{self.__class__.__name__} uninitialized')
        cdef int ret_code = c_md_concurrent_buffer_enable_worker(self.header, worker_id)

        if ret_code == MD_BUF_OK:
            return
        elif ret_code == MD_BUF_OOR:
            raise IndexError(f'Worker_id {worker_id} out of range')
        else:
            raise RuntimeError(f'Failed to enable worker, error code: {ret_code}')

    def put(self, MarketData market_data, bint block=True, double timeout=0):
        self.c_put(market_data.header, block, timeout)

    def listen(self, size_t worker_id, bint block=True, double timeout=0.0):
        cdef md_variant* md = self.c_listen(worker_id, block, timeout)
        return MarketData.c_from_header(md, False)

    def is_worker_empty(self, size_t worker_id):
        return self.c_is_worker_empty(worker_id)

    def is_empty(self):
        return self.c_is_empty()

    def is_full(self):
        return self.c_is_full()

    def disable_worker(self, size_t worker_id):
        self.c_disable_worker(worker_id)

    def enable_worker(self, size_t worker_id):
        self.c_enable_worker(worker_id)
