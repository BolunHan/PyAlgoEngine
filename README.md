# [PyAlgoEngine]("https://github.com/BolunHan/PyAlgoEngine"): Python Algo-Trading Engine

This module is a high-performance market data buffer implementation written in Python, Cython and C, designed for HFT (High Frequency Trading) system.

## 📦 Features

- C-level data structures for fast market data access
- Efficient shared memory support for interprocess communication
- Compile-time configurable parameters for memory and layout tuning

---

## ⚙️ Compile-Time Configuration

This module allows overriding several constants at compile time via environment variables.

### Available Parameters

| Variable      | Default | Description                      |
|---------------|---------|----------------------------------|
| `TICKER_SIZE` | `32`    | Max length of a ticker symbol    |
| `BOOK_SIZE`   | `10`    | Max depth of the order book      |
| `ID_SIZE`     | `16`    | Max length of ID field           |
| `MAX_WORKERS` | `128`   | Max number of concurrent workers |

These values are defined in `c_market_data_config.h`, but can be overridden at build time.

---

## 🚀 Building with Custom Parameters

To override default values, set environment variables before building:

### Using `pip install`

```bash
TICKER_SIZE=64 BOOK_SIZE=20 pip install .
```

### Using `setup.py` directly

```bash
TICKER_SIZE=64 BOOK_SIZE=20 python setup.py build_ext --inplace
```

These environment variables are passed to the C compiler as `-D` flags and will override the fallback values in `c_market_data_config.h`.

---

## 🧪 Verify Compilation

You can verify the values were compiled correctly by running:

```python
from algo_engine.base import C_CONFIG

print(C_CONFIG)
```