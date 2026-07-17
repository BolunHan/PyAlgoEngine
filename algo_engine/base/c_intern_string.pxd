from cbase.intern_string cimport (
    InternString,
    InternStringPool,
    istr_map,
    c_istr_map_new,
    c_istr_map_free,
    c_istr,
    c_istr_synced,
)

cdef istr_map* C_POOL
cdef InternStringPool POOL
cdef istr_map* C_INTRA_POOL
cdef InternStringPool INTRA_POOL
