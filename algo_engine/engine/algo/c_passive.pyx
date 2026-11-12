from cpython.unicode cimport PyUnicode_AsUTF8
from libc.math cimport NAN, isnan

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR
from algo_engine.engine.c_dma cimport DMA, c_dma_new_algo
from algo_engine.engine.c_dma_algo cimport ALGO_REGISTRY, c_dma_algo_registry_reg, dma_algo_ctx, dma_algo_new
from algo_engine.engine.c_dma_algo import AlgoLimitMode


cdef dma_algo_ctx* REG_CTX = c_dma_algo_registry_reg(ALGO_REGISTRY.header, DMA_ALGO_PASSIVE_NAME, <dma_algo_new> c_dma_algo_passive_new, DMA_ALGO_PASSIVE_REG_ID)
if not REG_CTX:
    raise RuntimeError('Failed to register <passive>.')


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
        self.header = <dma_algo_passive*> algo
        self.ccp_bind(&self.algo_meta)

    cdef void __ccp_dealloc__(self):
        self.header = NULL
        self.orders_map = None
        self.working_orders_map = None

    # === Python Properties ===

    property limit_price:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return None if isnan(self.header.limit_price) else self.header.limit_price

        def __set__(self, object value):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.limit_price = NAN if value is None else value

    property limit_adjust_factor:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return None if isnan(self.header.limit_adjust_factor) else self.header.limit_adjust_factor

        def __set__(self, object value):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.limit_adjust_factor = NAN if value is None else value

    property limit_mode:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return AlgoLimitMode(<int> self.header.limit_mode)

        def __set__(self, dma_algo_limit_mode mode):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.limit_mode = mode

    property timeout:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.timeout

        def __set__(self, double value):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.timeout = value

    property relaunch:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.relaunch
