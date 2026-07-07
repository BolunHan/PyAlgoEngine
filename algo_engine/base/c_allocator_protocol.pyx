from libc.stdlib cimport calloc

from cbase.allocator_protocol.c_heap_allocator cimport HeapAllocator
from cbase.allocator_protocol.c_shm_allocator cimport SharedMemoryAllocator


cdef bint MD_CFG_LOCKED = False
cdef bint MD_CFG_SHARED = True
cdef bint MD_CFG_FREELIST = True


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


cdef HeapAllocator _py_heap_allocator = HeapAllocator()
_py_heap_allocator.owner = False
cdef heap_allocator* HEAP_ALLOCATOR = _py_heap_allocator.allocator

cdef SharedMemoryAllocator _py_shm_allocator = SharedMemoryAllocator(shm_prefix='c_md_shm')
_py_shm_allocator.owner = False
cdef shm_allocator_ctx* SHM_ALLOCATOR = _py_shm_allocator.ctx

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
