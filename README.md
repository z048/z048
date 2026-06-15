# z048

A from-scratch reinforcement-learning engine that learns to play a 2048-like
game through TD(λ) self-play — guiding a depth-limited alpha-beta search in
which the tile spawner is treated as an _adversary_, not as random chance.

## Overview

z048 is a small, dependency-light RL engine written in Rust. It learns a value
function for a 2048-style sliding-tile game and uses that value function to
steer a minimax search.

The twist is in the framing. In ordinary 2048 the spawner places a new tile at
random; here the spawner is modeled as a **minimizing adversary** inside a
depth-limited alpha-beta tree. The player tries to maximize a shaped return
while the spawner tries to minimize it, turning the game into a two-player
minimax problem rather than an expectimax one against chance.

Learning is done by **TD(λ) self-play**. A small two-head value network (built
on [candle](https://github.com/huggingface/candle), CPU) predicts state values,
the search uses those predictions to sample moves, and the resulting
trajectories produce λ-return targets that are regressed back into the network.
Four binaries cover the lifecycle: `mint` initializes a fresh network, `dojo`
trains it via self-play, `duel` evaluates it headlessly, and `term` plays an
interactive game in the terminal (human and/or AI on either side).

## How it works

### The game and the potential function Φ

The board is a `4×4` grid of `u8`, where a cell value `r` encodes the tile `2^r`
(`0` = empty). Sliding compacts and merges tiles toward a direction; two
equal-rank tiles merge into one of rank `r+1`.

A potential function Φ drives reward shaping:

- **escore** = `Σ r · 2^r` over all 16 cells.
- **score** (`Φ`) = `log₂(escore)`.

Every transition contributes a shaped reward `ΔΦ = Φ(child) − Φ(parent)`. The
search and the learner both work in terms of these per-ply ΔΦ increments rather
than raw board scores, which keeps the value targets well-scaled.

### Board encoding

Boards are fed to the network as a **256-wide multi-hot vector**:
`4 × 4 cells × 16 ranks`. For each cell with rank `r`, the index
`cell * 16 + min(r, 15)` is activated (ranks above 15 are capped at index 15).

### The value network

A small fully-connected trunk with two output heads:

- Input `256` → hidden `[128, 32]` (default) → output `2`, with **ReLU after
  every layer except the output head**.
- **Head 0 — `v_after`**: value of the afterstate (used by `minimize` at depth
  0).
- **Head 1 — `v_before`**: value of the pre-slide state (used by `maximize` at
  depth 0).

Weights use He-uniform initialization (`limit = sqrt(6 / fan_in)`) with one
deliberate wrinkle: the input layer's _effective_ fan-in is **16**, not 256 —
because the multi-hot input activates exactly one rank per cell (16 of 256
inputs are ever set), so it scales by `sqrt(6 / 16)`. The **output head is
initialized to zero**, so `V ≡ 0` at the start and the untrained policy is pure
ΔΦ-greedy minimax. Checkpoints are stored as a postcard-serialized list of
per-layer `(weight, bias)` `f32` vectors.

### The search: adversarial alpha-beta over afterstates

Move selection runs a depth-limited alpha-beta minimax over afterstates, with ΔΦ
shaping applied at every ply:

- **`maximize`** (player to move): iterates legal slides; for each child
  evaluates `ΔΦ + minimize(child, depth−1)`. At depth 0 it reads head 1
  (`v_before`). Alpha-beta bounds are shifted by `−ΔΦ`.
- **`minimize`** (spawner to move): iterates legal spawns; for each child
  evaluates `ΔΦ + maximize(child, depth−1)`, or just `ΔΦ` if the spawn ends the
  game. At depth 0 it reads head 0 (`v_after`). Bounds are likewise shifted by
  `−ΔΦ`.

Actual moves during self-play and evaluation are drawn by **softmax sampling**
over the search scores, controlled by a temperature `τ`: `τ = 0` is greedy
(argmax), `τ = ∞` is uniform, and values in between interpolate. The player
samples slides to maximize `ΔΦ + minimize(...)`; the spawner samples spawns to
minimize the same quantity (it negates the score).

### TD(λ) training

Self-play games produce two trajectories of `(board, ΔΦ)` pairs — `befores`
(pre-slide states) and `afters` (afterstates). A backward λ-return recursion
blends bootstrapped network estimates with the Monte-Carlo tail:

```
g_after[t]  = afters[t].Δ  + (if last: 0 else (1−λ)·v_before[t+1] + λ·g_before[t+1])
g_before[t] = befores[t].Δ + (1−λ)·v_after[t]  + λ·g_after[t]
```

Each afterstate becomes a training row targeting head 0 with `g_after`; each
pre-slide state targets head 1 with `g_before`. Rows accumulate in a
fixed-capacity **ring buffer** (default 1,048,576) with FIFO eviction. Optimizer
steps sample uniform minibatches and apply **8-fold symmetry augmentation**
(each board's 8 dihedral symmetries) before computing a masked per-head MSE
loss, optimized with AdamW.

## Project layout

| File              | Role                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| `src/board.rs`    | `Board` type, slide/spawn mechanics, `escore`/`score`, legality checks, random initial board           |
| `src/slide.rs`    | `Slide` enum (U/D/L/R) and the `coord<N>` line→board coordinate mapping                                |
| `src/spawn.rs`    | `Spawn<N, M>` tile-spawn encoding (position + rank packed in a `u16`)                                  |
| `src/dicer.rs`    | `Dicer` RNG wrapper over `SmallRng` plus the temperature-scaled `softmax` sampler                      |
| `src/rater.rs`    | The value network: input encoding, two-head architecture, alpha-beta search, sampling, loss, postcard save/load |
| `src/train.rs`    | `Train`: the TD(λ) self-play loop, replay buffer, augmentation, and the serde JSON training config       |
| `src/lib.rs`      | Crate root tying the modules together                                                                  |
| `src/bin/mint.rs` | Binary: initialize a fresh, untrained checkpoint                                                       |
| `src/bin/dojo.rs` | Binary: runs a JSON list of training stages, snapshotting the net per stage/round                       |
| `src/bin/duel.rs` | Binary: net-vs-net evaluation arena (no learning)                                                      |
| `src/bin/term.rs` | Binary: interactive terminal arena (human and/or AI on each side) |

## Requirements

- A recent Rust toolchain — the crate uses **edition 2024**.
- Dependency stack:
  - `candle-core` / `candle-nn` (0.10.2) — tensors and neural-network modules.
    Runs on **CPU**.
  - `clap` (4.x, `derive`) — CLI parsing for `mint` / `duel` / `term`.
  - `rand` (0.9, `small_rng`) — RNG for self-play and initialization.
  - `serde` / `serde_json` — the `dojo` training config (JSON).
  - `postcard` — checkpoint serialization.

No GPU is required or used; all tensor work runs on the CPU.

## Build

```bash
# Standard release build of all four binaries
cargo build --release

# Recommended: enable native CPU optimizations for faster search and training
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Usage

A typical end-to-end flow: **mint** a fresh network, **train** it with `dojo`,
then **evaluate** with `duel`.

> **Required paths:** `mint` / `dojo` (`--rater`) and `duel` (`--slide-rater` /
> `--spawn-rater`) have no default — pass them explicitly. `dojo` also needs
> `--train <config.json>`.

### 1. Mint a fresh network

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin mint -- \
  --hidden 128 32 \
  --rater model.bin
```

`--seed` is optional; it defaults to `0x2048204820482048`. If you override it,
pass a **decimal** `u64` — the CLI does not accept a `0x…` literal.

### 2. Train via self-play

`dojo` takes a checkpoint (`--rater`) and a JSON **config** (`--train`). The
config is an **array of training stages** run in sequence on the same net; every
field is optional and falls back to the defaults in
[Training config](#training-config). Write a config:

```json
[
  { "num_round": 200, "search_depth": 2 },
  { "num_round": 100, "search_depth": 3, "adamw_lr": 5e-4 }
]
```

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin dojo -- \
  --rater model.bin --train config.json
```

Each round snapshots the net to a sibling file `model.s{stage}.r{round}.bin` and
logs the round's loss trend (mean of the first vs. last 25% of its steps) to
stderr:

```
stage 0 round 12: loss 0.04210 -> 0.03980
```

`num_round` must be set per stage — a stage with `num_round` 0 (or omitted)
trains nothing.

### 3. Evaluate

`duel` plays the `--slide-rater` net (the player) against the `--spawn-rater` net
(the adversarial spawner). Each side is loaded independently, so you can pit
different checkpoints against each other:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin duel -- \
  --slide-rater model.bin --slide-depth 2 \
  --spawn-rater model.bin --spawn-depth 2 \
  --rounds 128
```

Both `--slide-rater` and `--spawn-rater` must point at existing checkpoints; a
missing file is an error.

### 4. Play interactively

`term` runs a screen-refreshing game where the slide side is the player and the
spawn side the adversary. Give a side a `--*-rater` to let a network play it, or
omit it to play that side yourself — so any of PvE / PvP / EvE works:

```bash
# watch two nets play (EvE)
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin term -- \
  --slide-rater model.bin --spawn-rater model.bin

# play the slide side yourself against a net spawner (PvE)
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin term -- \
  --spawn-rater model.bin
```

Human controls: arrow keys slide; for a spawn, arrows move the cursor, `[` / `]`
pick the 2/4 tile, and space places it; `q` quits. Flags: `--slide-depth` /
`--spawn-depth` (default `4`), `--slide-tau` / `--spawn-tau` (default `0`),
`--seed` (defaults to the wall clock), and `--delay` (ms between AI moves,
default `80`). Requires a Unix/macOS terminal (`stty`).

## Training config

`dojo --train` points at a JSON file holding an **array** of stage objects.
Every field is optional and uses the serde default below; stages run top to
bottom on the same net. (`--rater` is a `dojo` CLI flag, not a config field.)

| Field          | Default               | Description                                                   |
| -------------- | --------------------- | ------------------------------------------------------------- |
| `num_round`    | `0`                   | Rounds for this stage (`0` / omitted = trains nothing)        |
| `play_games`   | `64`                  | Self-play games generated per round                           |
| `search_depth` | `2`                   | Alpha-beta search depth used during move sampling             |
| `train_steps`  | `256`                 | AdamW optimizer steps per round                               |
| `batch_size`   | `256`                 | Minibatch size per step (×8 after symmetry augmentation)      |
| `buffer_size`  | `1048576`             | Replay buffer capacity (FIFO eviction)                        |
| `random_seed`  | `2326144701688193096` | PRNG seed for reproducibility                                 |
| `td_lambda`    | `0.8`                 | TD(λ): blends bootstrap (`1−λ`) with Monte-Carlo return (`λ`) |
| `tau_a`        | `1.0`                 | Temperature numerator                                         |
| `tau_h`        | `8.0`                 | Temperature ply offset                                        |
| `tau_k`        | `0.02`                | Temperature floor                                             |
| `adamw_lr`     | `1e-3`                | AdamW learning rate                                           |
| `adamw_beta1`  | `0.9`                 | AdamW first-moment decay                                      |
| `adamw_beta2`  | `0.999`               | AdamW second-moment decay                                     |
| `adamw_eps`    | `1e-8`                | AdamW numerical-stability epsilon                             |
| `adamw_wd`     | `1e-4`                | AdamW weight decay (L2 regularization)                        |

The replay buffer and optimizer are re-created per stage — only the net weights
carry over between stages. The per-ply exploration temperature follows
`τ = tau_a / (ply + tau_h) + tau_k`, decreasing with ply toward the floor `tau_k`.

## Evaluation (duel)

`duel` runs deterministic, greedy (`τ = 0`) games (round `i` uses
`--seed + i`) with no learning. The slide side and the spawn side are **separate
nets, loaded independently**, so you can evaluate a checkpoint against itself or
pit two different checkpoints against each other (e.g. a trained player against a
snapshot spawner).

**Flags:**

- `--slide-rater` (required) — net that picks slides (the player), searched at
  `--slide-depth`.
- `--slide-depth` (default `2`) — alpha-beta depth for the slide side.
- `--spawn-rater` (required) — net that picks spawns (the adversary), searched
  at `--spawn-depth`.
- `--spawn-depth` (default `2`) — alpha-beta depth for the spawn side.
- `--rounds` (default `128`) — number of games to play.
- `--seed` (defaults to the wall clock) — round `i` uses `--seed + i`.

Both checkpoints must exist on disk; pointing at a missing file is an error (no
fresh-net fallback).

**Output:** per-round lines reporting `phi_final`, ply count, and max tile rank,
followed by a summary (the histogram lists only tiles that actually occurred as
a game maximum):

```
round 0: phi_final 11.000 plies 254 max_rank 8
summary: games 100 phi_final mean 14.512 median 14.000 p10 12.000 p90 16.000
max-rank histogram (tile:count): 1024:31 2048:12
```

## Notes

- All computation runs on the **CPU**; there is no GPU backend.
