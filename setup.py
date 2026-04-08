import os
import shutil
import sys
from contextlib import suppress
from pathlib import Path

import event_engine
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExtWithConfig(build_ext):
    def run(self):
        self.pre_compile()

        super().run()

        self.post_compile()

    def build_extensions(self):
        macros = []
        for macro in ["DEBUG", "TICKER_SIZE", "BOOK_SIZE", "ID_SIZE", "MAX_WORKERS"]:
            val = os.environ.get(macro)
            if val:
                print(f'Compile-time variable {macro} overridden with value {val}')
                macros.append((macro, val))
        for ext in self.extensions:
            ext.define_macros = macros
        super().build_extensions()

    def pre_compile(self):
        self.remove_pxd(
            [
                "algo_engine.base",
                "algo_engine.base.c_market_data",
                "algo_engine.exchange_profile",
                "algo_engine.engine"
            ]
        )

    def post_compile(self):
        # Monkey hack the "__init__.pxd" issue:
        self.inject_pxd(
            [
                "algo_engine.base",
                "algo_engine.base.c_market_data",
                "algo_engine.exchange_profile",
                "algo_engine.engine"
            ]
        )

    def remove_pxd(self, modules: list[str]) -> None:
        project_root = Path(__file__).resolve().parent

        for module in modules:
            src_dir = project_root.joinpath(*module.split("."))
            init_pxd = src_dir / "__init__.pxd"

            if init_pxd.exists():
                print(f"[pre_compile] Removing {init_pxd}")
                with suppress(FileNotFoundError):
                    init_pxd.unlink()

    def inject_pxd(self, modules: list[str]) -> None:
        for module in modules:
            project_root = Path(__file__).resolve().parent
            src_dir = project_root.joinpath(*module.split("."))
            pkg_dir = Path(self.build_lib, *module.split("."))

            infra_pxd = src_dir / "__infra__.pxd"
            if not infra_pxd.exists():
                continue

            pkg_dir.mkdir(parents=True, exist_ok=True)
            init_pxd = pkg_dir / "__init__.pxd"

            print(f"[build_py] Injecting {infra_pxd} -> {init_pxd}")
            shutil.copyfile(infra_pxd, init_pxd)


# Define the extensions
extensions = [
    # Extension(
    #     name="algo_engine.profile.c_base",
    #     sources=["algo_engine/profile/c_base.pyx"],
    #     extra_compile_args=["-O3"]
    # ),
    # Extension(
    #     name="algo_engine.profile.c_cn",
    #     sources=["algo_engine/profile/c_cn.pyx"],
    #     extra_compile_args=["-O3"]
    # ),
]

if os.name == 'posix':
    extensions.extend([
        # === Base Cython Extensions ===
        Extension(
            name="algo_engine.base.c_shm_allocator",
            sources=["algo_engine/base/c_shm_allocator.pyx"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_heap_allocator",
            sources=["algo_engine/base/c_heap_allocator.pyx"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_intern_string",
            sources=["algo_engine/base/c_intern_string.pyx"],
            extra_compile_args=["-O3"]
        ),
        # === Exchange Profile Cython Extensions ===
        Extension(
            name="algo_engine.exchange_profile.c_exchange_profile",
            sources=["algo_engine/exchange_profile/c_exchange_profile.pyx",
                     "algo_engine/exchange_profile/c_ex_profile_base.c",
                     "algo_engine/exchange_profile/c_ex_profile_cn.c"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.exchange_profile.c_profile_dispatcher",
            sources=["algo_engine/exchange_profile/c_profile_dispatcher.pyx"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.exchange_profile.c_profile_default",
            sources=["algo_engine/exchange_profile/c_profile_default.pyx"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.exchange_profile.c_profile_cn",
            sources=["algo_engine/exchange_profile/c_profile_cn.pyx"],
            extra_compile_args=["-O3"]
        ),
        # === Market Data Cython Extensions ===
        Extension(
            name="algo_engine.base.c_market_data.c_allocator_protocol",
            sources=["algo_engine/base/c_market_data/c_allocator_protocol.pyx"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data.c_market_data",
            sources=["algo_engine/base/c_market_data/c_market_data.pyx"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data.c_internal",
            sources=["algo_engine/base/c_market_data/c_internal.pyx"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data.c_transaction",
            sources=["algo_engine/base/c_market_data/c_transaction.pyx"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data.c_tick",
            sources=["algo_engine/base/c_market_data/c_tick.pyx"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data.c_candlestick",
            sources=["algo_engine/base/c_market_data/c_candlestick.pyx"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data.c_trade_utils",
            sources=["algo_engine/base/c_market_data/c_trade_utils.pyx"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data.c_market_data_buffer",
            sources=["algo_engine/base/c_market_data/c_market_data_buffer.pyx"],
            extra_compile_args=["-O3"]
        ),
        # === MDS Cython Extensions ===
        Extension(
            name="algo_engine.engine.c_market_engine",
            sources=["algo_engine/engine/c_market_engine.pyx"],
            extra_compile_args=["-O3"]
        ),
        # === EventEngine Integration Cython Extensions ===
        Extension(
            name="algo_engine.engine.c_event_engine",
            sources=["algo_engine/engine/c_event_engine.pyx"],
            include_dirs=[*event_engine.get_include()],
            extra_compile_args=["-O3"]
        )
    ])

setup(
    name="algo_engine",
    ext_modules=cythonize(extensions, force="--force" in sys.argv),
    cmdclass={"build_ext": BuildExtWithConfig},
)
