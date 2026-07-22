import os
import platform
import re
import shutil
import sys
from contextlib import suppress
from pathlib import Path

import cbase
import event_engine
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# ==============================
# Setup Configuration
# ==============================

PACKAGE_NAME = "algo_engine"
DISPLAY_NAME = "PyAlgoEngine"

WITH_ANNOTATION = False
COMPILE_FLAGS = ["/Ox", "/std:c17", "/experimental:c11atomics"] if platform.system() == "Windows" else ['-O3'] + ([] if os.environ.get('GITHUB_ACTIONS') == 'true' else ['-march=native'])
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
N_CORES = os.cpu_count() or 1
# MSVC + cythonize process pool conflict on Windows (spawn re-imports setup.py) — build sequentially there.
N_THREADS = 0 if platform.system() == "Windows" else max(1, N_CORES - 2)
__VERSION__ = match.group(1) if (match := re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', (Path(REPO_ROOT) / PACKAGE_NAME / '__init__.py').read_text(), re.MULTILINE)) else "unknown"

ext_modules = []
c_extensions = []
cython_extension = []


# ==============================
# Custom Build Extension Class
# ==============================


class BuildExtWithConfig(build_ext):
    __cy_modules__ = [
        "algo_engine",
        "algo_engine.base",
        "algo_engine.base.c_market_data",
        "algo_engine.exchange_profile",
        "algo_engine.engine"
    ]

    def initialize_options(self):
        super().initialize_options()
        self.parallel = N_THREADS

    def run(self):
        self.pre_compile()

        super().run()

        self.post_compile()
        print(f"[build_py] <{DISPLAY_NAME}> v{__VERSION__} setup complete. Built {len(self.extensions)} Cython extensions.")

    def build_extensions(self):
        macros = []
        for macro in ["DEBUG", "TICKER_SIZE", "BOOK_SIZE", "ID_SIZE", "MAX_WORKERS"]:
            val = os.environ.get(macro)
            if val:
                print(f'[build_py] Compile-time variable {macro} overridden with value {val}')
                macros.append((macro, val))
        for ext in self.extensions:
            ext.define_macros = macros
        super().build_extensions()

    def pre_compile(self):
        self.collect_sources()

    def post_compile(self):
        # Monkey hack the "__init__.pxd" issue:
        self.inject_pxd()

        # Inject the generated includes/ mirror into build_lib so it gets packaged
        self.inject_sources()

    @classmethod
    def collect_sources(cls) -> None:
        project_root = Path(__file__).resolve().parent
        source_root = project_root / PACKAGE_NAME
        include_root = project_root / PACKAGE_NAME / "includes"
        mirror_root = include_root / PACKAGE_NAME

        if mirror_root.exists():
            shutil.rmtree(mirror_root)

        copied = 0
        source_patterns = ["*.h", "*.c", "*.cpp"]
        for pattern in source_patterns:
            for source_file in sorted(source_root.rglob(pattern)):
                # Skip files inside the generated include_root
                if include_root in source_file.parents:
                    continue
                dest = include_root.joinpath(*source_file.relative_to(project_root).parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, dest)
                copied += 1

        print(f"[build_py] <{DISPLAY_NAME}> mirrored {copied} C source file(s) -> {include_root.relative_to(project_root)}")

    def inject_sources(self) -> None:
        project_root = Path(__file__).resolve().parent
        include_root = project_root / PACKAGE_NAME / "includes"
        mirror_root = include_root / PACKAGE_NAME

        if not mirror_root.exists():
            return

        dest_root = Path(self.build_lib, PACKAGE_NAME, "includes", PACKAGE_NAME)
        if dest_root.exists():
            shutil.rmtree(dest_root)

        shutil.copytree(mirror_root, dest_root)
        print(f"[build_py] <{DISPLAY_NAME}> injected includes mirror -> {dest_root.relative_to(Path(self.build_lib))}")

    @classmethod
    def remove_pxd(cls) -> None:
        project_root = Path(__file__).resolve().parent

        for module in cls.__cy_modules__:
            src_dir = project_root.joinpath(*module.split("."))
            init_pxd = src_dir / "__init__.pxd"

            if init_pxd.exists():
                print(f"[build_py] [pre_compile] Removing {init_pxd}")
                with suppress(FileNotFoundError):
                    init_pxd.unlink()

        # Temporarily move the algo_engine symlink/dir from Cython/Includes
        # to prevent cythonize from picking up stale pxd files from an
        # installed version of algo_engine.
        cls._pxd_backup = []
        try:
            import Cython.Includes as _cy_includes
            includes_dir = Path(next(iter(_cy_includes.__path__)))
            algo_link = includes_dir / "algo_engine"
            if algo_link.exists():
                backup = algo_link.with_name("algo_engine.cy_bak")
                print(f"[build_py] [pre_compile] Moving Cython/Includes symlink {algo_link.name} -> {backup.name}")
                algo_link.rename(backup)
                cls._pxd_backup.append((backup, algo_link))
        except Exception as e:
            print(f"[build_py] [pre_compile] Warning: Could not handle Cython/Includes symlink: {e}")

    @classmethod
    def restore_pxd(cls) -> None:
        """Restore algo_engine symlink in Cython/Includes after cythonize."""
        for backup, original in getattr(cls, '_pxd_backup', []):
            if backup.exists():
                print(f"[build_py] [pre_compile] Restoring Cython/Includes symlink {backup.name} -> {original.name}")
                backup.rename(original)
        cls._pxd_backup = []

    def inject_pxd(self) -> None:
        project_root = Path(__file__).resolve().parent

        for module in self.__cy_modules__:
            src_dir = project_root.joinpath(*module.split("."))
            pkg_dir = Path(self.build_lib, *module.split("."))

            infra_pxd = src_dir / "__infra__.pxd"
            if not infra_pxd.exists():
                continue

            # Inject into build_lib (for wheel / deploy)
            pkg_dir.mkdir(parents=True, exist_ok=True)
            init_pxd = pkg_dir / "__init__.pxd"
            print(f"[build_py] [post_compile] Injecting {infra_pxd} -> {init_pxd}")
            shutil.copyfile(infra_pxd, init_pxd)

            # Also restore into src_dir (so source tree keeps __init__.pxd)
            src_init_pxd = src_dir / "__init__.pxd"
            if not src_init_pxd.exists():
                print(f"[build_py] [post_compile] Restoring {infra_pxd} -> {src_init_pxd}")
                shutil.copyfile(infra_pxd, src_init_pxd)


# =============================
# Define Cython Extensions
# =============================


cython_extension.extend([
    # === Base Cython Extensions ===
    Extension(
        name="algo_engine.base.c_allocator_protocol",
        sources=["algo_engine/base/c_allocator_protocol.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    Extension(
        name="algo_engine.base.c_intern_string",
        sources=["algo_engine/base/c_intern_string.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    # === Exchange Profile Cython Extensions ===
    Extension(
        name="algo_engine.exchange_profile.c_exchange_profile",
        sources=["algo_engine/exchange_profile/c_exchange_profile.pyx",
                 "algo_engine/exchange_profile/c_ex_profile_base.c",
                 "algo_engine/exchange_profile/c_ex_profile_cn.c"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    Extension(
        name="algo_engine.exchange_profile.c_profile_dispatcher",
        sources=["algo_engine/exchange_profile/c_profile_dispatcher.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    Extension(
        name="algo_engine.exchange_profile.c_profile_default",
        sources=["algo_engine/exchange_profile/c_profile_default.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    Extension(
        name="algo_engine.exchange_profile.c_profile_cn",
        sources=["algo_engine/exchange_profile/c_profile_cn.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    # === Market Data Cython Extensions ===
    Extension(
        name="algo_engine.base.c_market_data.c_market_data",
        sources=["algo_engine/base/c_market_data/c_market_data.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    Extension(
        name="algo_engine.base.c_market_data.c_internal",
        sources=["algo_engine/base/c_market_data/c_internal.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    Extension(
        name="algo_engine.base.c_market_data.c_transaction",
        sources=["algo_engine/base/c_market_data/c_transaction.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    Extension(
        name="algo_engine.base.c_market_data.c_tick",
        sources=["algo_engine/base/c_market_data/c_tick.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    Extension(
        name="algo_engine.base.c_market_data.c_candlestick",
        sources=["algo_engine/base/c_market_data/c_candlestick.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    Extension(
        name="algo_engine.base.c_market_data.c_trade_utils",
        sources=["algo_engine/base/c_market_data/c_trade_utils.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    Extension(
        name="algo_engine.base.c_market_data.c_market_data_buffer",
        sources=["algo_engine/base/c_market_data/c_market_data_buffer.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    # === MDS Cython Extensions ===
    Extension(
        name="algo_engine.engine.c_market_engine",
        sources=["algo_engine/engine/c_market_engine.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    ),
    # === EventEngine Integration Cython Extensions ===
    Extension(
        name="algo_engine.engine.c_event_engine",
        sources=["algo_engine/engine/c_event_engine.pyx"],
        include_dirs=[REPO_ROOT, *cbase.get_include(), *event_engine.get_include()],
        extra_compile_args=[*COMPILE_FLAGS]
    )
])

BuildExtWithConfig.remove_pxd()

try:
    ext_modules.extend(
        cythonize(
            cython_extension,
            annotate=WITH_ANNOTATION,
            compiler_directives={
                "language_level": "3",
                'embedsignature': True
            },
            force="--force" in sys.argv,
            nthreads=N_THREADS,
        )
    )
finally:
    BuildExtWithConfig.restore_pxd()

# =============================
# Define C Extensions
# =============================

ext_modules.extend(c_extensions)

# =============================
# Setup Function
# =============================

setup(
    name=PACKAGE_NAME,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtWithConfig}
)
