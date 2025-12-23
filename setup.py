import os
import pathlib

from Cython.Build import cythonize
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

import event_engine


class BuildExtWithConfig(build_ext):
    def build_extensions(self):
        macros = []
        for macro in ["DEBUG", "TICKER_SIZE", "BOOK_SIZE", "ID_SIZE", "MAX_WORKERS"]:
            val = os.environ.get(macro)
            if val:
                print(f'Compile-time variable {macro} overridden with value {val}')
                macros.append((macro, val))
        for ext in self.extensions:
            ext.define_macros = macros
        build_ext.build_extensions(self)


# Define the extensions
extensions = [
    Extension(
        name="algo_engine.profile.c_base",
        sources=["algo_engine/profile/c_base.pyx"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        name="algo_engine.profile.c_cn",
        sources=["algo_engine/profile/c_cn.pyx"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        name="algo_engine.base.c_market_data.c_market_data",
        sources=["algo_engine/base/c_market_data/c_market_data.pyx"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        name="algo_engine.base.c_market_data.c_transaction",
        sources=["algo_engine/base/c_market_data/c_transaction.pyx"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        name="algo_engine.base.c_market_data.c_candlestick",
        sources=["algo_engine/base/c_market_data/c_candlestick.pyx"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        name="algo_engine.base.c_market_data.c_tick",
        sources=["algo_engine/base/c_market_data/c_tick.pyx"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        name="algo_engine.base.c_market_data.c_market_data_buffer",
        sources=["algo_engine/base/c_market_data/c_market_data_buffer.pyx"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        name="algo_engine.base.c_market_data.c_trade_utils",
        sources=["algo_engine/base/c_market_data/c_trade_utils.pyx"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        name="algo_engine.engine.c_market_engine",
        sources=["algo_engine/engine/c_market_engine.pyx"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        name="algo_engine.engine.c_event_engine",
        sources=["algo_engine/engine/c_event_engine.pyx"],
        include_dirs=[*event_engine.get_include()],
        extra_compile_args=["-O3"]
    )
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
        # === Market Data Cython Extensions ===
        Extension(
            name="algo_engine.base.c_market_data_ng.c_market_data",
            sources=["algo_engine/base/c_market_data_ng/c_market_data.pyx"],
            include_dirs=["algo_engine/base"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data_ng.c_internal",
            sources=["algo_engine/base/c_market_data_ng/c_internal.pyx"],
            include_dirs=["algo_engine/base"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data_ng.c_transaction",
            sources=["algo_engine/base/c_market_data_ng/c_transaction.pyx"],
            include_dirs=["algo_engine/base"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data_ng.c_tick",
            sources=["algo_engine/base/c_market_data_ng/c_tick.pyx"],
            include_dirs=["algo_engine/base"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data_ng.c_candlestick",
            sources=["algo_engine/base/c_market_data_ng/c_candlestick.pyx"],
            include_dirs=["algo_engine/base"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data_ng.c_trade_utils",
            sources=["algo_engine/base/c_market_data_ng/c_trade_utils.pyx"],
            include_dirs=["algo_engine/base"],
            extra_compile_args=["-O3"]
        ),
        Extension(
            name="algo_engine.base.c_market_data_ng.c_market_data_buffer",
            sources=["algo_engine/base/c_market_data_ng/c_market_data_buffer.pyx"],
            include_dirs=["algo_engine/base"],
            extra_compile_args=["-O3"]
        ),
    ])

setup(
    name="algo_engine",
    ext_modules=cythonize(extensions),
    cmdclass={"build_ext": BuildExtWithConfig},
)
