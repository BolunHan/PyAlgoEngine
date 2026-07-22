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

This cleans previous artifacts, compiles 15 Cython extensions in-place
(parallel build using ``os.cpu_count() - 2`` threads on Linux), and
installs the package.

Use ``-v <path>`` to activate a virtual environment before building.

Method 2: Make
~~~~~~~~~~~~~~~

.. code-block:: bash

   make build && pip install -U . --no-build-isolation

``make build`` runs ``python setup.py build_ext --inplace --verbose --force``
after cleaning stale artifacts.

Method 3: Step-by-step
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python setup.py build_ext --inplace --verbose --force
   pip install -U . --no-build-isolation

Compile-Time Configuration
---------------------------

You can override memory layout constants at compile time via environment
variables. Probe available macros with:

.. code-block:: bash

   ./build.sh -l        # or: make list-args

This reads all ``#define`` directives from C headers and prints a table
of macro names, default values, and source locations.

Key user-configurable macros:

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``BOOK_SIZE``
     - 10
     - Max order book depth (price levels)
   * - ``ID_SIZE``
     - 16
     - Max size of ``md_id`` fields (order/transaction IDs)
   * - ``LONG_ID_SIZE``
     - 128
     - Max size of ``long_md_id`` fields (report/instruction IDs)
   * - ``MAX_WORKERS``
     - 128
     - *(deprecated, no longer has any effect)*
   * - ``MD_BUF_PTR_DEFAULT_CAP``
     - 16
     - Default pointer capacity for ``MarketDataBuffer``
   * - ``MD_BUF_DATA_DEFAULT_CAP``
     - 1024
     - Default data byte capacity for ``MarketDataBuffer``
   * - ``DEBUG``
     - 0
     - Enable debug assertions (set to 1)

Override before building:

.. code-block:: bash

   BOOK_SIZE=20 ./build.sh -i

Verify the compiled and runtime values:

.. code-block:: python

   from algo_engine.base import CONFIG
   print(CONFIG)
   # DEBUG=False; BOOK_SIZE=10; ID_SIZE=16; LONG_ID_SIZE=128; ...

Runtime configuration (``MD_CFG_*``) can be inspected on ``CONFIG`` as well
but is set programmatically, not at compile time.

Optional Dependencies
---------------------

.. code-block:: bash

   pip install PyAlgoEngine[WebApps]   # Flask, waitress, bokeh
   pip install PyAlgoEngine[Docs]      # sphinx, sphinx-rtd-theme

Platform Notes
--------------

**Linux** (primary target):
    GCC or Clang with ``-O3 -march=native``. Parallel builds use
    ``os.cpu_count() - 2`` threads.

**Windows**:
    MSVC with ``/Ox /std:c17 /experimental:c11atomics``. Parallel builds
    are disabled by default (set to 1 thread).
    A ``build.ps1`` PowerShell script is provided for Windows.
