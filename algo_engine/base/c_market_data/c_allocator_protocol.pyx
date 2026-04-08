from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc


cdef bint MD_CFG_LOCKED = False
cdef bint MD_CFG_SHARED = True
cdef bint MD_CFG_FREELIST = True


cdef class EnvConfigContext:
    def __cinit__(self, **kwargs):
        self.overrides = kwargs
        self.originals = {}

    cdef void c_activate(self):
        if 'locked' in self.overrides:
            global MD_CFG_LOCKED
            self.originals['locked'] = MD_CFG_LOCKED
            MD_CFG_LOCKED = self.overrides['locked']
            MD_DEFAULT_ALLOCATOR.with_lock = MD_CFG_LOCKED

        if 'shared' in self.overrides:
            global MD_CFG_SHARED
            self.originals['shared'] = MD_CFG_SHARED
            MD_CFG_SHARED = self.overrides['shared']
            MD_DEFAULT_ALLOCATOR.with_shm = MD_CFG_SHARED

        if 'freelist' in self.overrides:
            global MD_CFG_FREELIST
            self.originals['freelist'] = MD_CFG_FREELIST
            MD_CFG_FREELIST = self.overrides['freelist']
            MD_DEFAULT_ALLOCATOR.with_freelist = MD_CFG_FREELIST

    cdef void c_deactivate(self):
        if 'locked' in self.originals:
            global MD_CFG_LOCKED
            MD_CFG_LOCKED = self.originals.pop('locked')
            MD_DEFAULT_ALLOCATOR.with_lock = MD_CFG_LOCKED

        if 'shared' in self.originals:
            global MD_CFG_SHARED
            MD_CFG_SHARED = self.originals.pop('shared')
            MD_DEFAULT_ALLOCATOR.with_shm = MD_CFG_SHARED

        if 'freelist' in self.originals:
            global MD_CFG_FREELIST
            MD_CFG_FREELIST = self.originals.pop('freelist')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.overrides!r})'

    def __enter__(self):
        self.c_activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.c_deactivate()

    def __or__(self, EnvConfigContext other):
        if not isinstance(other, EnvConfigContext):
            return NotImplemented
        merged_overrides = self.overrides | other.overrides
        return EnvConfigContext(**merged_overrides)

    def __invert__(self):
        return EnvConfigContext.__new__(
            EnvConfigContext,
            **{k: not v if isinstance(v, bool) else v for k, v in self.overrides.items()}
        )

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.c_activate()
            ret = func(*args, **kwargs)
            self.c_deactivate()
            return ret
        return wrapper


cdef EnvConfigContext MD_SHARED     = EnvConfigContext(shared=True)
cdef EnvConfigContext MD_LOCKED     = EnvConfigContext(locked=True)
cdef EnvConfigContext MD_FREELIST   = EnvConfigContext(freelist=True)


globals()['MD_SHARED'] = MD_SHARED
globals()['MD_LOCKED'] = MD_LOCKED
globals()['MD_FREELIST'] = MD_FREELIST


cdef class AllocatorProtocol:
    def __cinit__(self, size_t size):
        if not size:
            return
        if MD_CFG_SHARED:
            self.protocol = c_md_allocator_protocol_new(size, SHM_ALLOCATOR, NULL, <int> MD_CFG_LOCKED)
        elif MD_CFG_FREELIST:
            self.protocol = c_md_allocator_protocol_new(size, NULL, HEAP_ALLOCATOR, <int> MD_CFG_LOCKED)
        else:
            self.protocol = c_md_allocator_protocol_new(size, NULL, NULL, 0)
        self.owner = True

    def __dealloc__(self):
        if not self.owner:
            return

        if self.protocol:
            c_md_allocator_protocol_free(self.protocol)

    @staticmethod
    cdef AllocatorProtocol c_from_protocol(allocator_protocol* protocol, bint owner):
        cdef AllocatorProtocol instance = AllocatorProtocol.__new__(AllocatorProtocol, 0)
        instance.protocol = protocol
        instance.owner = owner
        return instance

    def __repr__(self):
        if not self.protocol:
            return f'<{self.__class__.__name__}>(Uninitialized)'
        return f'<{self.__class__.__name__} {<uintptr_t> self.protocol:#0x}>(with_shm={self.protocol.with_shm}, with_lock={self.protocol.with_lock}, size={self.protocol.size})'

    property with_lock:
        def __get__(self):
            if not self.protocol:
                raise RuntimeError('allocator_protocol not initialized')
            return self.protocol.with_lock

    property with_shm:
        def __get__(self):
            if not self.protocol:
                raise RuntimeError('allocator_protocol not initialized')
            return self.protocol.with_shm

    property with_freelist:
        def __get__(self):
            if not self.protocol:
                raise RuntimeError('allocator_protocol not initialized')
            return self.protocol.with_freelist

    property size:
        def __get__(self):
            if not self.protocol:
                raise RuntimeError('allocator_protocol not initialized')
            return self.protocol.size

    property buf:
        def __get__(self):
            if not self.protocol:
                raise RuntimeError('allocator_protocol not initialized')
            if self.protocol:
                return <char[:self.protocol.size]> self.protocol.buf
            return None

    property addr:
        def __get__(self):
            if not self.protocol:
                raise RuntimeError('allocator_protocol not initialized')
            return <uintptr_t> self.protocol


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
