from cpython.unicode cimport PyUnicode_AsUTF8, PyUnicode_FromString
from libc.math cimport NAN, isnan

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR
from algo_engine.engine.c_dma cimport DMA, c_dma_new_algo
from algo_engine.engine.c_dma_algo cimport ALGO_REGISTRY, c_dma_algo_registry_reg, dma_algo_ctx, dma_algo_new
from algo_engine.engine.c_dma_algo import AlgoLimitMode


cdef dma_algo_ctx* REG_CTX = c_dma_algo_registry_reg(ALGO_REGISTRY.header, DMA_ALGO_PASSIVE_NAME, <dma_algo_new> c_dma_algo_passive_new, DMA_ALGO_PASSIVE_REG_ID)
if not REG_CTX:
    raise RuntimeError('Failed to register <passive>.')
ALGO_REGISTRY.add_algo(PyUnicode_FromString(DMA_ALGO_PASSIVE_NAME), Passive)


cdef class Passive(AlgoTemplate):
    def __init__(
            self,
            str ticker,
            md_side side,
            double target_volume,
            double limit_price=NAN,
            double limit_adjust_factor=NAN,
            dma_algo_limit_mode limit_mode=dma_algo_limit_mode.DMA_ALGO_LIMIT_STRICT,
            **kwargs
    ):
        cdef dma_algo_passive_params params

        params.limit_price = limit_price
        params.limit_adjust_factor = limit_adjust_factor
        params.limit_mode = limit_mode
        params.timeout = 0.
        params.relaunch = False

        cdef dma_algo* algo = c_dma_new_algo(
            DMA.header,
            DMA_ALGO_PASSIVE_REG_ID,
            PyUnicode_AsUTF8(ticker),
            side,
            target_volume,
            &params,
            MD_DEFAULT_ALLOCATOR
        )

        if not algo:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.algo_meta = algo
        self.ccp_bind(&self.algo_meta)

    # === Python Properties ===

    property limit_price:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            cdef dma_algo_passive* record = <dma_algo_passive*> self.algo_meta
            return None if isnan(record.limit_price) else record.limit_price

        def __set__(self, double value):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            cdef dma_algo_passive* record = <dma_algo_passive*> self.algo_meta
            record.limit_price = value

    property limit_adjust_factor:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            cdef dma_algo_passive* record = <dma_algo_passive*> self.algo_meta
            return None if isnan(record.limit_adjust_factor) else record.limit_adjust_factor

        def __set__(self, double value):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            cdef dma_algo_passive* record = <dma_algo_passive*> self.algo_meta
            record.limit_adjust_factor = value

    property limit_mode:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            cdef dma_algo_passive* record = <dma_algo_passive*> self.algo_meta
            return AlgoLimitMode(<int> record.limit_mode)

        def __set__(self, dma_algo_limit_mode mode):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            cdef dma_algo_passive* record = <dma_algo_passive*> self.algo_meta
            record.limit_mode = mode

    property timeout:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            cdef dma_algo_passive* record = <dma_algo_passive*> self.algo_meta
            return record.timeout

        def __set__(self, double value):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            cdef dma_algo_passive* record = <dma_algo_passive*> self.algo_meta
            record.timeout = value

    property relaunch:
        def __get__(self):
            if not self.algo_meta:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            cdef dma_algo_passive* record = <dma_algo_passive*> self.algo_meta
            return record.relaunch
