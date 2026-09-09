#include <algo_engine/exchange_profile/c_ex_profile_base.h>

/* c_ex_profile_capi.c — Windows consumer-side import of the live profile
 * globals. PE DLLs have no global symbol scope, so each consumer extension
 * carries its own slots (compiled here) and resolves them once at module
 * init from the defining extension's exports via GetProcAddress — the pure
 * OS-level mechanism, no Python. POSIX needs none of this: the ELF loader
 * binds the direct extern references after c_ex_profile_promote_globals().
 * The slots are dereferenced on every use (see the macros in
 * c_ex_profile_base.h), so activation switches propagate to every
 * extension. */

#if defined(_WIN32) || defined(_WIN64)

#include <stdio.h>
#include <windows.h>

const exchange_profile**            EX_PROFILE_SLOT = NULL;
const session_date_range_t**        EX_TRADE_CALENDAR_CACHE_SLOT = NULL;
ex_profile_activation_listener**    EX_PROFILE_ACTIVATION_LISTENERS_SLOT = NULL;

int EX_PROFILE_IMPORT(void) {
    if (EX_PROFILE_SLOT) return 0;

    /* Locate the defining module by the full path derived from this
     * extension's own path: the ABI suffix of this extension's file name
     * is identical for every extension of one build, and the "algo_engine"
     * (or "test") package directory anchors the shared package root. The
     * loader registers modules under their full path, so GetModuleHandleA
     * is asked for the exact path and LoadLibraryA returns the already
     * loaded module if it was imported earlier — always the same copy. */
    HMODULE self = NULL;
    GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR) &EX_PROFILE_SLOT, &self);
    if (!self) return -1;

    char self_name[MAX_PATH];
    if (!GetModuleFileNameA(self, self_name, MAX_PATH)) return -1;

    const char* base = strrchr(self_name, '\\');
    base = base ? base + 1 : self_name;
    const char* suffix = strchr(base, '.');  /* e.g. ".cp313-win_amd64.pyd" */

    const char* anchor = strstr(self_name, "\\algo_engine\\");
    if (!anchor) anchor = strstr(self_name, "\\test\\");
    if (!anchor) return -1;

    char definer_path[MAX_PATH];
    snprintf(definer_path, sizeof(definer_path), "%.*salgo_engine\\exchange_profile\\c_exchange_profile%s",
             (int) (anchor - self_name + 1), self_name, suffix ? suffix : ".pyd");

    HMODULE h = GetModuleHandleA(definer_path);
    if (!h) h = LoadLibraryA(definer_path);
    if (!h) return -1;
    /* No FreeLibrary: the defining extension must outlive every consumer
     * (the slots point into it for the process lifetime). The one-time
     * refcount bump from LoadLibraryA is deliberate. */

    /* The defining extension exports the pointer variables themselves
     * (built with /GL- so the export table binds the variable, not the
     * pointed-to struct); GetProcAddress returns the variable's address. */
    EX_PROFILE_SLOT = (const exchange_profile**) GetProcAddress(h, "EX_PROFILE");
    EX_TRADE_CALENDAR_CACHE_SLOT = (const session_date_range_t**) GetProcAddress(h, "EX_TRADE_CALENDAR_CACHE");
    EX_PROFILE_ACTIVATION_LISTENERS_SLOT = (ex_profile_activation_listener**) GetProcAddress(h, "EX_PROFILE_ACTIVATION_LISTENERS");

    return (EX_PROFILE_SLOT && EX_TRADE_CALENDAR_CACHE_SLOT && EX_PROFILE_ACTIVATION_LISTENERS_SLOT) ? 0 : -1;
}

#endif /* _WIN32 */
