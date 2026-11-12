from cpython.unicode cimport PyUnicode_AsUTF8
from libc.math cimport NAN

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR
from algo_engine.engine.c_dma cimport DMA, c_dma_new_algo
from algo_engine.engine.c_dma_algo cimport ALGO_REGISTRY, c_dma_algo_registry_reg, dma_algo_ctx, dma_algo_limit_mode, dma_algo_new

from .c_passive cimport dma_algo_passive


cdef dma_algo_ctx* REG_CTX = c_dma_algo_registry_reg(ALGO_REGISTRY.header, DMA_ALGO_PASSIVE_TIMEOUT_NAME, <dma_algo_new> c_dma_algo_passive_timeout_new, DMA_ALGO_PASSIVE_TIMEOUT_REG_ID)
if not REG_CTX:
    raise RuntimeError('Failed to register <passive_timeout>.')


cdef class PassiveTimeout(Passive):
    def __init__(
            self,
            str ticker,
            md_side side,
            double target_volume,
            double timeout=0.,
            double limit_price=NAN,
            double limit_adjust_factor=NAN,
            dma_algo_limit_mode limit_mode=dma_algo_limit_mode.DMA_ALGO_LIMIT_STRICT,
            **kwargs
    ):
        cdef dma_algo_passive_params params

        params.limit_price = limit_price
        params.limit_adjust_factor = limit_adjust_factor
        params.limit_mode = limit_mode
        params.timeout = timeout
        params.relaunch = False

        cdef dma_algo* algo = c_dma_new_algo(
            DMA.header,
            DMA_ALGO_PASSIVE_TIMEOUT_REG_ID,
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
