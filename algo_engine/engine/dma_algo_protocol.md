# DMA algo protocol

The exchange-report contract of `algo_engine/engine/c_dma_algo.h`. The exchange
reports are the only source of exchange order facts. This document states what
each report is assumed to be, which field decides each state judgment, and
which checks each function keeps. The state transitions themselves are in
`orderstate_flow.md`.

## Reports

| report                                                         | entry                        | assumption                                    | indication                | writes                   |
| -------------------------------------------------------------- | ---------------------------- | --------------------------------------------- | ------------------------- | ------------------------ |
| order-state event                                              | `c_dma_algo_engine_on_order` | the exchange's current view of one live order | the event's `order_state` | the order state column   |
| trade report, `c_md_side_offset(trade->side) == OFFSET_CANCEL` | `c_dma_algo_engine_on_trade` | a cancel confirmation                         | the side's offset         | the order state column   |
| trade report, any other offset                                 | `c_dma_algo_engine_on_trade` | one execution                                 | the side's offset         | the fill quantities only |

The `volume` of a trade report is a quantity, never a classifier. A cancel
report may carry a positive volume — some exchanges report the canceled amount in
this field — and that amount is not booked as a fill. A fill report carries
the executed quantity.

## The one gate

`c_dma_algo_engine_on_trade` owns the trade-id gate: it probes `ctx->reports`
once and returns `DMA_ALGO_OK` on a hit; a new report is copied into the context
before any effect. No other function dedups, and no report is discarded for its
pre-state or its volume.

## Operations

### Cancel report — a trade report whose side carries `OFFSET_CANCEL`

- Assumption: reached only inside `on_trade`, with the context resolved.
- Indication: `ctx->order.order_state`, the stored order's state column — written
  by the order-state event or by the access's in-place update, never by the
  trade report itself.
- Classification: `STATE_CANCELING`, `STATE_SENT`, `STATE_PLACED` and
  `STATE_PARTFILLED` apply; `STATE_CANCELED` is a repeated confirmation and a
  no-op; every other pre-state is refused.
- Effect: after the gate, writes `STATE_CANCELED` and `ts_canceled`, records
  `STATE_CANCELED` as the last applied state, releases the remaining working
  volume once, leaves the working indexes and dispatches `on_canceled`.

### Trade entry — `c_dma_algo_engine_on_trade`

- Assumption: one report from the exchange.
- Indication: the report's order id resolves the context in the all-orders map
  — one probe, live or done — and the trade side's offset classifies the
  report.
- The gate: one probe of `ctx->reports`; a repeated trade id returns
  `DMA_ALGO_OK` before any effect, and a new report is copied into the context
  first.
- Fill: books the volume, notional and fee inline, adopting the report's
  multiplier when the order's is NaN; releases the fill volume from the
  working set while `ctx->flags == DMA_ORDER_LISTED` (a stop bit means the
  remaining volume was already released); updates the algo aggregates;
  dispatches `on_filled`; mirrors the fill into the hosting position. It
  writes no order state and no timestamp: the state column follows the
  order-state event.
- Cancel report: classifies against `ctx->order.order_state`, writes the cancel,
  releases the remaining working volume once, leaves the working indexes,
  dispatches `on_canceled` and mirrors the release into the position.
- A repeated trade id stops at the gate and touches nothing.

### Order entry — `c_dma_algo_engine_on_order`

- Assumption: one exchange order-state event for a live order — the working
  index.
- The event arrives in its own buffer: its `order_state` is classified against
  the stored order's state column, exactly as the flows in `orderstate_flow.md`
  state. A tolerated event is a no-op; a banned event is refused; an applied
  event is copied into the stored order.
- A caller that hands in the stored order itself (a bound view) has written the
  state already, so the pre-state is gone. The entry prints one line to stderr
  and routes to `c_dma_algo_engine_on_order_inplace`, which runs no transition
  guard because the writer's operation guard already classified the pre-state.
  The access's own writes call that entry directly.
- The working-volume delta, the working-index departure, the callbacks and the
  position mirror follow the state now in the stored order.

### In-place updates — one entry per written state

- The access calls the entry for the state it wrote:
  `..._on_order_pending_inplace`, `..._launched_inplace`, `..._placed_inplace`,
  `..._partially_filled_inplace`, `..._filled_inplace`, `..._canceling_inplace`,
  `..._canceled_inplace`, `..._rejected_inplace`, `..._invalid_inplace`.
- Working volume: an order carries none at creation. The first working state
  (pending or sent) opens the remaining volume once at the algo and its host;
  every later working state moves nothing. A terminal state releases the
  remaining volume only when it was opened, so a never-sent order is never
  deducted.
- Effect: the pending and launched entries open the remaining volume once; the
  placed entry dispatches `on_listed` at the algo and the position; the
  partially-filled and canceling entries apply nothing (the order is working,
  so its volume is open and the host already holds the algo); filled leaves the
  working indexes; canceled, rejected and invalid release the remaining volume
  once, leave the working indexes and dispatch the terminal callback at the
  algo and the position.
- `c_dma_algo_engine_on_order_inplace` dispatches a stored-order view to the
  entry for its state. A state with no entry is a misuse: the dispatcher
  returns `DMA_ALGO_ERR_INVALID_INPUT` and reports it to stderr.

### Order construction — `c_dma_algo_engine_new_order`

- Assumption: a constructed engine and an algo created by
  `c_dma_algo_init` (its ticker and access are set).
- Checks kept: the argument validation — `engine`, `algo`, a finite positive
  volume.
- This is the only order producer: the instruction starts its `dma_order_ctx`
  and the id is that context's address, so the access hands the instruction to
  the engine as the context. A caller never constructs an order.

### Working index — `c_dma_algo_engine_pop_working_order`

- Assumption: a constructed engine. `pop_working_order` is called by the
  entries above.
- `pop_working_order` removes the order from the working indexes and from the
  algo's live map; the history indexes keep the context.

### Market data — `c_dma_algo_engine_on_md`

- Assumption: one market-data observation carrying a ticker.
- Indication: the observation's ticker selects the working algo list.

## Check boundary

A check is kept only where the caller's guarantee does not already cover it.

| function                                                           | may assume                                                                    | checks kept                                                                                                                      |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `c_dma_algo_engine_on_trade`, `on_order`, `on_md`                  | a constructed engine                                                          | the NULL arguments; `on_order` also owns the stored-order detection for a caller that hands one in; `on_trade` also owns the trade-id gate, the fill booking and the cancel classification |
| the per-state in-place entries                                     | the access, which wrote that state; the state is already written              | the NULL arguments                                                                                                               |
| `c_dma_algo_engine_on_order_inplace`                               | a stored-order view handed to the main entry                                  | the NULL arguments; a state with no entry is refused with a stderr line                                                          |
| `c_dma_algo_engine_new_order`                                      | a constructed engine and algo                                                 | the NULL arguments and a finite positive volume                                                                                  |
| `c_dma_algo_engine_pop_working_order`, `free_orders`               | a constructed engine; a context always carries its algo and ticker            | the NULL arguments                                                                                                               |
| `c_dma_algo_engine_new`, `_init`, `_dealloc`, `_free`, `_clear`    | a caller-allocated object                                                     | the constructor boundary; `_dealloc` and `_clear` tolerate a partially initialized engine because `_new` tears down through them |
| registry `_new`, `_init`, `_reg`, `_get`, `_dealloc`, `_free`      | the C API surface                                                             | the argument validation                                                                                                          |

Removed in this sweep, each because the caller's guarantee covers it:

- the order-context store, cancel and fill helpers and the `n_before` length
  comparison: the trade-id gate and the fill booking are stated once, in
  `on_trade`.
- the order-context `last_state` shadow and `note_state`: the stored order's
  state column is the only record; the exchange event writes it and the
  access's in-place writes route through `on_order_inplace`.
- `on_trade`'s working-map probe plus the all-orders fallback and its
  `!engine->orders` re-check: one probe in the all-orders map resolves both a
  live and a late report.
- the fill pre-state ban (`STATE_IDLE`, `STATE_UNKNOWN`, `STATE_PENDING`,
  `STATE_REJECTED`, `STATE_INVALID`), the `volume <= 0` return, the overfill
  refusal and the multiplier-mismatch refusal: the exchange report is
  authoritative and its volume is not a classifier.
- `on_order`'s `!algo` check; the `engine->position`, `position->tickers` and
  `position->working_algos` re-checks; the same map checks in `new_order`,
  `pop_working_order`, `free_orders` and `on_md`; the `factory_ctx->orders` and
  `algo->orders` re-checks: the engine constructor creates those maps and the
  access never clears them.

## Corrections

The earlier convention — a zero-volume trade report is the cancel
confirmation — is withdrawn. A trade report is a cancel report exactly when
its side carries `OFFSET_CANCEL`; its volume is the canceled amount and may be
positive. A fill report never writes the order state: the state column is the
exchange's order-state event.

The `last_state` shadow is withdrawn. It duplicated the stored order's state
column and was only correct while every writer called `note_state`. The stored
order's state column is the record: the exchange event writes it, and the
access's local writes are written in place and applied through
`c_dma_algo_engine_on_order_inplace`.

## The access protocol

The access dispatches through one attached `dma_protocol`; the Python wrapper
binds it in `c_bind_protocol` on construction.

| part     | entries                       | required |
| -------- | ----------------------------- | -------- |
| handlers | launch, cancel                | yes      |
| handlers | balance query                 | no; its absence passes the gate |
| callbacks | rejected, filled             | no       |

A subclass is bound only when it overrides both required handlers; the
rejected/filled callbacks are attached only when those methods are overridden. An
access that is not bound stays inert. The balance query fills the access's
`dma_balance` and reports 0 when it supplied one; the risk profile's balance
gate reads it after the rules pass, and a failed or absent query leaves the
gate passing. The gate values the trade at its multiplier and the ticker's
designated margin rate: NaN checks the buy's notional against available cash
(a sale releases cash), a finite rate checks the notional times the rate
against available margin in either direction.
