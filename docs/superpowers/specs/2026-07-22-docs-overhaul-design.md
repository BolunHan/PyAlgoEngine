# PyAlgoEngine Documentation Overhaul — Design Spec

**Date:** 2026-07-22
**Status:** approved → implementation

## Problem

Current docs are obsolete:
- `index.rst` misidentifies the project as a "decision tree library"
- `setup.rst` is 2 lines (`pip install PyAlgoEngine`, "Python 3.12+")
- `usage.rst` is one undifferentiated `MockData` code block with no narrative
- `apis.rst` / `modules.rst` are stubs relying on `sphinx-apidoc` flat dump
- No architecture overview, backtest guide, strategy guide, or engine docs
- README.md focuses only on compile-time config, missing project identity

Meanwhile, the codebase is v0.12.0 — a full HFT algo-trading engine with 8 sub-packages,
15+ Cython extensions, C-level data structures, shared memory buffers, backtesting,
exchange calendars, web visualization, and more.

## Approach: Curated + Auto-generated Hybrid (Option B)

Manual prose pages for guides/architecture + curated `autodoc` directives for API reference
(not a raw `sphinx-apidoc` dump).

## New RST Structure

```
docs/source/
├── index.rst                    # [U] Landing page — project identity, features, quick links
├── setup.rst                    # [U] Build & install guide (3 methods, compile-time config)
├── usage.rst                    # [U] Quick start with real examples
├── architecture.rst             # [+] Architecture overview — layer diagram, data flow
├── market_data.rst              # [+] Market data types guide
├── backtest.rst                 # [+] Backtesting guide
├── strategy.rst                 # [+] Strategy development guide
├── engines.rst                  # [+] Engine layer guide
├── web_apps.rst                 # [+] WebApp/Bokeh visualization guide
├── api/                         # [+] Curated API reference
│   ├── index.rst                # API overview + toc
│   ├── base.rst                 # algo_engine.base
│   ├── engine.rst               # algo_engine.engine
│   ├── backtest.rst             # algo_engine.backtest
│   ├── exchange_profile.rst     # algo_engine.exchange_profile
│   ├── strategy.rst             # algo_engine.strategy
│   ├── apps.rst                 # algo_engine.apps
│   ├── monitor.rst              # algo_engine.monitor
│   └── utils.rst                # algo_engine.utils
├── conf.py                      # [U] Add intersphinx, autosummary, better config
```

`[U]` = updated, `[+]` = new

## Content Plan

| File | Content |
|------|---------|
| **index.rst** | Project description (HFT engine, not decision trees), feature list, architecture diagram (ASCII), quick install command, link to full setup, toc tree |
| **setup.rst** | 3 build methods (`./build.sh -i`, `make build && pip install`, `setup.py build_ext`), compile-time macros table, requirements, optional deps (WebApps, Docs), venv setup, platform notes (Linux/Windows) |
| **usage.rst** | Quick start: import, create buffer, push ticks, subscribe. Real examples for: reading market data, using exchange profiles, creating a simple strategy. Link to deeper guides |
| **architecture.rst** | Layer diagram (ASCII art), data flow from market data → buffer → engine → strategy → execution, package dependency graph, singleton pattern explanation |
| **market_data.rst** | All market data types: TickData, BarData/DailyBar, OrderBook, TransactionData/OrderData/TradeData, MarketData base. Buffer types: MarketDataBuffer, RingBuffer, ConcurrentBuffer, BufferCache. Code snippets for each |
| **backtest.rst** | Replay engine (SimpleReplay, ProgressReplay), SimMatch order matching, TradeMetrics, StrategyTester, running backtests end-to-end |
| **strategy.rst** | StrategyEngineTemplate, AlgoTemplate, ALGO_ENGINE/ALGO_REGISTRY, writing custom strategies, event handlers |
| **engines.rst** | EventEngine (topics, handlers), MarketDataService/MDS (subscriptions, monitors), AlgoEngine (algo lifecycle), TradeEngine (DMA, positions, risk) |
| **web_apps.rst** | WebApp/start_app, Bokeh DocServer/CandleStick, Tester UI |
| **api/*.rst** | Per sub-package: hand-written overview paragraph, then `.. autoclass::` / `.. autofunction::` directives for key public API |

## Removals

- `modules.rst` and `apis.rst` → replaced by `api/index.rst` + curated sub-pages
- Old auto-generated `algo_engine.rst` → replaced by curated `api/*.rst`
- "Decision tree" description → accurate HFT engine description

## Tools Updated

| Tool | Changes |
|------|---------|
| **`docs/update_docs.sh`** | Add venv activation, better error handling, `make html` with `-W` for CI |
| **`docs/Makefile`** | Unchanged (standard Sphinx Makefile) |
| **`docs/source/conf.py`** | Add `sphinx.ext.intersphinx`, `sphinx.ext.autosummary`, fix `sys.path`, suppress Cython warnings |
| **`README.md`** | Rewrite: project identity, feature grid, quick install + verify, link to docs |

## Build & Verify

1. Run `docs/update_docs.sh` to build HTML
2. Fix all Sphinx warnings
3. Verify key pages render correctly
4. Check `_build/html/` output
