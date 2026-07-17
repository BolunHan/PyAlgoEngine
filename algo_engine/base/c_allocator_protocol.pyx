from libc.stdlib cimport calloc

from cbase.allocator_protocol.c_heap_allocator cimport c_heap_allocator_new
from cbase.allocator_protocol.c_shm_comp cimport c_shm_allocator_new, AP_SHM_ALLOCATOR_DEFAULT_REGION_SIZE


cdef c_bool MD_CFG_LOCKED = False
cdef c_bool MD_CFG_SHARED = True
cdef c_bool MD_CFG_FREELIST = True


cdef class MDConfigContext(AllocatorConfigContext):
    cdef void c_bind(self, allocator_protocol* schematic=NULL):
        self.allocator_schematic = schematic if schematic else MD_DEFAULT_ALLOCATOR

    cdef void c_activate(self):
        if 'locked' in self.overrides:
            global MD_CFG_LOCKED
            MD_CFG_LOCKED = self.overrides['locked']

        if 'shared' in self.overrides:
            global MD_CFG_SHARED
            MD_CFG_SHARED = self.overrides['shared']

        if 'freelist' in self.overrides:
            global MD_CFG_FREELIST
            MD_CFG_FREELIST = self.overrides['freelist']

        AllocatorConfigContext.c_activate(self)

    cdef void c_deactivate(self):
        if 'locked' in self.originals:
            global MD_CFG_LOCKED
            MD_CFG_LOCKED = self.originals.get('locked')

        if 'shared' in self.originals:
            global MD_CFG_SHARED
            MD_CFG_SHARED = self.originals.get('shared')

        if 'freelist' in self.originals:
            global MD_CFG_FREELIST
            MD_CFG_FREELIST = self.originals.get('freelist')

        AllocatorConfigContext.c_deactivate(self)


cdef heap_allocator* HEAP_ALLOCATOR = c_heap_allocator_new()
if not HEAP_ALLOCATOR:
    raise OSError("Initialize MD heap allocator failed")

cdef shm_allocator_ctx* SHM_ALLOCATOR = c_shm_allocator_new(AP_SHM_ALLOCATOR_DEFAULT_REGION_SIZE, <char*> b"c_md_shm")
if not SHM_ALLOCATOR:
    raise OSError("Initialize MD SHM allocator failed (prefix='c_md_shm')")

cdef allocator_protocol* MD_DEFAULT_ALLOCATOR   = <allocator_protocol*> calloc(1, sizeof(allocator_protocol))
cdef allocator_protocol* MD_SHM_ALLOCATOR       = <allocator_protocol*> calloc(1, sizeof(allocator_protocol))
cdef allocator_protocol* MD_HEAP_ALLOCATOR      = <allocator_protocol*> calloc(1, sizeof(allocator_protocol))

MD_DEFAULT_ALLOCATOR.with_lock          = MD_CFG_LOCKED
MD_DEFAULT_ALLOCATOR.with_shm           = MD_CFG_SHARED
MD_DEFAULT_ALLOCATOR.with_freelist      = MD_CFG_FREELIST
MD_DEFAULT_ALLOCATOR.shm_allocator_ctx  = SHM_ALLOCATOR
MD_DEFAULT_ALLOCATOR.shm_allocator      = SHM_ALLOCATOR.shm_allocator
MD_DEFAULT_ALLOCATOR.heap_allocator     = HEAP_ALLOCATOR

MD_SHM_ALLOCATOR.with_lock              = True
MD_SHM_ALLOCATOR.with_shm               = True
MD_SHM_ALLOCATOR.with_freelist          = True
MD_SHM_ALLOCATOR.shm_allocator_ctx      = SHM_ALLOCATOR
MD_SHM_ALLOCATOR.shm_allocator          = SHM_ALLOCATOR.shm_allocator
MD_SHM_ALLOCATOR.heap_allocator         = NULL

MD_HEAP_ALLOCATOR.with_lock             = True
MD_HEAP_ALLOCATOR.with_shm              = False
MD_HEAP_ALLOCATOR.with_freelist         = True
MD_HEAP_ALLOCATOR.shm_allocator_ctx     = NULL
MD_HEAP_ALLOCATOR.shm_allocator         = NULL
MD_HEAP_ALLOCATOR.heap_allocator        = HEAP_ALLOCATOR

cdef MDConfigContext MD_SHARED          = MDConfigContext(shared=True)
cdef MDConfigContext MD_LOCKED          = MDConfigContext(locked=True)
cdef MDConfigContext MD_LOCKFREE        = MDConfigContext(locked=False)
cdef MDConfigContext MD_FREELIST        = MDConfigContext(freelist=True)

globals()['MD_SHARED'] = MD_SHARED
globals()['MD_LOCKED'] = MD_LOCKED
globals()['MD_LOCKFREE'] = MD_LOCKFREE
globals()['MD_FREELIST'] = MD_FREELIST
