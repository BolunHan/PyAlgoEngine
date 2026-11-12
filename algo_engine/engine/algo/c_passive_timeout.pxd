from libc.stdint cimport uint32_t

from cbase.allocator_protocol cimport allocator_protocol

from algo_engine.base.c_market_data.c_market_data cimport md_side
from algo_engine.engine.c_dma cimport dma

from .c_algo_template cimport dma_algo
from .c_passive cimport Passive, dma_algo_passive_params


cdef extern from "algo_engine/engine/algo/c_passive_timeout.h":
    const char* DMA_ALGO_PASSIVE_TIMEOUT_NAME
    const uint32_t DMA_ALGO_PASSIVE_TIMEOUT_REG_ID

    dma_algo* c_dma_algo_passive_timeout_new(dma* DMA, const char* ticker, md_side side, double target_volume, const dma_algo_passive_params* params, allocator_protocol* allocator) noexcept nogil


cdef class PassiveTimeout(Passive):
    pass
