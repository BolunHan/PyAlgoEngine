import codecs
import os

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


class BuildExtWithFallback(build_ext):
    """Custom build_ext to handle Cython compilation with fallback."""

    def run(self):
        try:
            print("Attempting to compile Cython modules...")
            super().run()
        except Exception as e:
            print("Cython compilation failed:", e)
            print("Falling back to pure Python implementation.")



from Cython.Build import cythonize

cython_ext = [
    Extension(
        name="algo_engine.base.market_data",
        sources=["algo_engine/base/market_data.pyx"],
        extra_compile_args=['-O3', '-flto'],
        extra_link_args=['-flto']
    ),
    Extension(
        name="algo_engine.base.transaction",
        sources=["algo_engine/base/transaction.pyx"],
        extra_compile_args=['-O3', '-flto'],
        extra_link_args=['-flto']
    ),
    Extension(
        name="algo_engine.base.candlestick",
        sources=["algo_engine/base/candlestick.pyx"],
        extra_compile_args=['-O3', '-flto'],
        extra_link_args=['-flto']
    ),
    Extension(
        name="algo_engine.base.tick",
        sources=["algo_engine/base/tick.pyx"],
        extra_compile_args=['-O3', '-flto'],
        extra_link_args=['-flto']
    ),
    Extension(
        name="algo_engine.base.market_data_buffer",
        sources=["algo_engine/base/market_data_buffer.pyx"],
        extra_compile_args=['-O3', '-flto'],
        extra_link_args=['-flto']
    ),
    Extension(
        name="algo_engine.base.trade_utils",
        sources=["algo_engine/base/trade_utils.pyx"],
        extra_compile_args=['-O3', '-flto'],
        extra_link_args=['-flto']
    )
]

ext_modules = cythonize(
    cython_ext,
    compiler_directives={
        'language_level': "3",
        'optimize.use_switch': True,
        'optimize.unpack_method_calls': True,
        'binding': False
    },
    annotate=False
)

long_description = read("README.md")

setuptools.setup(
    name="PyAlgoEngine",
    version=get_version(os.path.join('algo_engine', '__init__.py')),
    author="Bolun.Han",
    author_email="Bolun.Han@outlook.com",
    description="Basic algo engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BolunHan/PyAlgoEngine",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
    install_requires=[
        'numpy',
        'pandas',
        'exchange_calendars',
        'PyEventEngine',
        'cython'
    ],
    extras_require={
        "WebApps": [
            "flask",
            "waitress",
            "bokeh"
        ],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtWithFallback},
    command_options={
        'nuitka': {
            # boolean option, e.g. if you cared for C compilation commands
            '--show-scons': ("setup.py", True),
            # options without value, e.g. enforce using Clang
            '--clang': ("setup.py", None),
            # options with single values, e.g. enable a plugin of Nuitka
            # '--enable-plugin': ("setup.py", "pyside2"),
            # options with several values, e.g. avoiding including modules
            # '--nofollow-import-to': ("setup.py", ["*.tests", "*.distutils"]),
            # disable LTO
            '--lto': ("setup.py", 'yes'),
            # include some common 3rd party packages
            '--include-package': ("setup.py", ['ctypes', 'datetime', 'typing', 'multiprocessing']),
            # '--mode': ("setup.py", 'standalone')
        }
    }
)
