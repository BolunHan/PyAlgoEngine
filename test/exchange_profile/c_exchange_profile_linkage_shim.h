/* c_exchange_profile_linkage_shim.h — Windows-only shim for the
 * exchange_profile linkage test. PE DLLs have no global symbol scope (no
 * RTLD_GLOBAL equivalent), so a consumer extension cannot leave EX_PROFILE
 * undefined and resolve it at load time. Instead the symbol is redirected to
 * a slot that is resolved at runtime from the loaded c_exchange_profile.pyd.
 *
 * Resolution notes:
 * - The pyd exports EX_PROFILE (dllexport in c_ex_profile_base.h), but MSVC
 *   may bind the export-table entry to the EX_PROFILE_DEFAULT struct rather
 *   than the EX_PROFILE pointer variable. So we locate the pointer variable
 *   by scanning the module image for the unique 8-byte slot that holds the
 *   struct's address (the variable's initializer). The scan finds exactly one
 *   such slot, and it tracks profile activation.
 * - The macro below dereferences the slot on every use, so the inlined header
 *   functions always observe the live profile.
 * - This header MUST be included before c_ex_profile_base.h: the macro also
 *   rewrites the header's dllexport extern declaration (harmlessly — same
 *   type, and the slot definition below carries the matching dllexport).
 */
#ifndef C_EXCHANGE_PROFILE_LINKAGE_SHIM_H
#define C_EXCHANGE_PROFILE_LINKAGE_SHIM_H

#ifdef _WIN32

#include <stdint.h>
#include <string.h>
#include <windows.h>

struct exchange_profile;  /* forward decl; matches the typedef in c_ex_profile_base.h */

/* dllexport + external linkage: must match the storage class of the
 * macro-rewritten declaration from c_ex_profile_base.h. */
__declspec(dllexport) const struct exchange_profile** _pyx_test_ex_profile_slot = NULL;

#define EX_PROFILE (*_pyx_test_ex_profile_slot)

/* Returns 1 on success, 0 if the module does not export EX_PROFILE. The
 * caller passes the HMODULE of the exact .pyd copy their Python code imports
 * (e.g. via ctypes.CDLL(module.__file__)._handle), so multiple loaded copies
 * cannot be confused. */
static int _pyx_test_resolve_ex_profile(uintptr_t module_handle) {
    HMODULE h = (HMODULE)module_handle;
    if (!h) return 0;

    void* exported = GetProcAddress(h, "EX_PROFILE");
    if (!exported) return 0;

    uint32_t size_of_image = 0;
    const unsigned char* image = (const unsigned char*)h;
    if (image[0] == 'M' && image[1] == 'Z') {
        uint32_t e_lfanew = *(const uint32_t*)(image + 0x3C);
        const unsigned char* pe = image + e_lfanew;
        if (pe[0] == 'P' && pe[1] == 'E' && pe[2] == 0 && pe[3] == 0) {
            size_of_image = *(const uint32_t*)(pe + 0x50);  /* PE32+ optional header */
        }
    }
    if (!size_of_image) return 0;

    /* Locate the EX_PROFILE pointer variable: it is the unique .data slot
     * initialized to &EX_PROFILE_DEFAULT, a struct whose first bytes are
     * "UTC_NONSTOP_". Dereferencing the slot on every use (see the macro
     * below) tracks profile activation. */
    for (uintptr_t off = 0; off + sizeof(uintptr_t) <= size_of_image; off += sizeof(uintptr_t)) {
        uintptr_t candidate = *(const uintptr_t*)(image + off);
        if (candidate < (uintptr_t)image || candidate >= (uintptr_t)image + size_of_image)
            continue;
        if (memcmp((const void*)candidate, "UTC_NONSTOP_", 12) != 0)
            continue;
        _pyx_test_ex_profile_slot = (const struct exchange_profile**)(image + off);
        break;
    }
    return _pyx_test_ex_profile_slot && *_pyx_test_ex_profile_slot ? 1 : 0;
}

#endif /* _WIN32 */

#endif /* C_EXCHANGE_PROFILE_LINKAGE_SHIM_H */
