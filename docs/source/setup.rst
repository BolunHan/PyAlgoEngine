Setup & Installation
====================

Requirements
------------

* **Python** 3.12 or higher
* **C compiler** (GCC, Clang, or MSVC)
* **Linux** or **Windows** (macOS may work but is not tested)

Dependencies are declared in ``pyproject.toml`` and installed automatically:

* ``numpy``, ``pandas`` — data handling
* ``exchange_calendars`` — trading calendar support
* ``PyCyBase`` — Cython base utilities (allocators, interned strings)
* ``PyEventEngine`` — event-driven framework
* ``Cython`` — extension compilation

Install from Source
-------------------

Clone the repository and build:

.. code-block:: bash

   git clone https://github.com/BolunHan/PyAlgoEngine.git
   cd PyAlgoEngine

Then choose one of three build methods:

Method 1: build.sh (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ./build.sh -i

This cleans previous artifacts, compiles Cython extensions in-place, and
installs the package. Use ``-v <path>`` to specify a virtual environment.

Method 2: Make
~~~~~~~~~~~~~~~

.. code-block:: bash

   make build && pip install -U . --no-build-isolation

``make build`` runs ``python setup.py build_ext --inplace --verbose --force``
after cleaning.

Method 3: Step-by-step
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python setup.py build_ext --inplace --verbose --force
   pip install -U . --no-build-isolation

Compile-Time Configuration
---------------------------

You can override memory layout constants at compile time via environment
variables. This is useful for tuning the engine to specific market data
requirements without modifying source code.

.. list-table:: Compile-Time Macros
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``TICKER_SIZE``
     - 32
     - Max length of a ticker symbol (bytes)
   * - ``BOOK_SIZE``
     - 10
     - Max depth of the order book (levels)
   * - ``ID_SIZE``
     - 16
     - Max length of ID fields (bytes)
   * - ``MAX_WORKERS``
     - 128
     - Max number of concurrent buffer workers
   * - ``DEBUG``
     - 0
     - Enable debug assertions (set to 1)

Override any of these before building:

.. code-block:: bash

   BOOK_SIZE=20 MAX_WORKERS=256 ./build.sh -i

Verify the compiled values:

.. code-block:: python

   from algo_engine.base import CONFIG
   print(CONFIG)

Optional Dependencies
---------------------

Install extras for additional features:

.. code-block:: bash

   # Web dashboard support
   pip install PyAlgoEngine[WebApps]

   # Documentation build tools
   pip install PyAlgoEngine[Docs]

Platform Notes
--------------

**Linux** (primary target):
    GCC or Clang with ``-O3 -march=native``. Parallel builds use
    ``os.cpu_count() - 2`` threads.

**Windows**:
    MSVC with ``/Ox /std:c17``. Parallel builds are disabled by default
    (set to 1 thread); override with the ``N_THREADS`` environment variable
    if needed.
