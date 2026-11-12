from libc.stdint cimport uint32_t
from libcpp cimport bool as c_bool

from cbase.allocator_protocol cimport allocator_protocol

from algo_engine.base.c_market_data.c_market_data cimport md_side, md_trade_instruction, md_trade_report, md_variant
from algo_engine.engine.c_dma cimport dma
from algo_engine.engine.c_dma_algo cimport dma_algo_limit_mode

from .c_algo_template cimport AlgoTemplate, dma_algo


cdef extern from "algo_engine/engine/algo/c_passive.h":
    const char* DMA_ALGO_PASSIVE_NAME
    const uint32_t DMA_ALGO_PASSIVE_REG_ID

    ctypedef struct dma_algo_passive_params:
        double              limit_price
        double              limit_adjust_factor
        double              timeout
        dma_algo_limit_mode limit_mode
        c_bool              relaunch

    ctypedef struct dma_algo_passive:
        dma_algo            base
        double              limit_price
        double              limit_adjust_factor
        double              timeout
        dma_algo_limit_mode limit_mode
        c_bool              relaunch

    int  c_dma_algo_passive_launch(dma* DMA, dma_algo* base, void* user_data) noexcept nogil
    int  c_dma_algo_passive_stop(dma* DMA, dma_algo* base, void* user_data) noexcept nogil
    void c_dma_algo_passive_on_canceled(dma* DMA, dma_algo* base, const md_trade_instruction* order, void* user_data) noexcept nogil
    void c_dma_algo_passive_on_rejected(dma* DMA, dma_algo* base, const md_trade_instruction* order, void* user_data) noexcept nogil
    void c_dma_algo_passive_on_filled(dma* DMA, dma_algo* base, const md_trade_report* trade, void* user_data) noexcept nogil
    void c_dma_algo_passive_on_md(dma* DMA, dma_algo* base, const md_variant* market_data, void* user_data) noexcept nogil

    dma_algo* c_dma_algo_passive_new(dma* DMA, const char* ticker, md_side side, double target_volume, const dma_algo_passive_params* params, allocator_protocol* allocator) noexcept nogil


cdef class Passive(AlgoTemplate):
    cdef dma_algo_passive* header
