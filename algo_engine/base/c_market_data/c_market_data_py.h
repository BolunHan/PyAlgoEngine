#include <Python.h>

#include <algo_engine/base/c_market_data/c_market_data.h>
#include <cbase/allocator_protocol/c_dual_interface.h>

typedef struct md_variant_pyclass {
    // === PyObject_HEAD ====
    PyObject_HEAD  // macro already ends with ';' since Python 3.11 - a trailing ';' makes an empty declaration, which MSVC's C frontend rejects (C2059)
    // === Pyx Virt Table ===
    void*         __pyx_vtab;
    // === PyObject __dict__ ===
    PyObject*         py_dict;
    // === Readonly header address ===
    uintptr_t         data_addr;
    // === bint ownership marker ===
    int               owner;
    md_variant const* header;
} md_variant_pyclass;

typedef struct md_trade_report {
    md_variant_pyclass base;
    ccp_ctx            ccp_ctx;
} md_trade_report;

typedef struct md_trade_instruction {
    md_variant_pyclass base;
    ccp_ctx            ccp_ctx;
} md_trade_instruction;
