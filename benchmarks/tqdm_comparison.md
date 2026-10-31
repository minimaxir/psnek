# psnek vs tqdm

A side-by-side comparison of psnek and tqdm on the same workloads, with both
libraries at their default settings unless noted.

**Setup:** Apple M3 Pro, macOS (Darwin 27.2), CPython 3.11.14 (GIL build),
psnek 0.1.0 (commit `1e68ff4`), tqdm 4.70.1, ipykernel 7.4.0, ipywidgets
8.1.9, nbclient 0.11.0. Measured 2026-10-27. Every run is a fresh process or a
fresh kernel. Runs are sequential, never parallel, and the two libraries take
turns in alternating order between rounds.

**Reproducing:** loop overhead (section 1) comes from `bench.py`. Everything
else comes from `footprint.py`, which runs each child process and kernel with
a minimal environment so neither library sees CI or coding-agent variables.
The notebook section also needs `ipykernel` and `ipywidgets` in the
environment.

```bash
uv run --group bench python benchmarks/bench.py
```

```bash
uv run --group bench python benchmarks/footprint.py imports terminal notebook
```

## Summary

| Attribute | tqdm | psnek | Winner |
|---|---:|---:|---|
| Per-item overhead, list loop | 51.8 ns | 3.8 ns | psnek, 13.7x less |
| Per-item overhead, `update()` | 68.8 ns | 19.4 ns | psnek, 3.5x less |
| Per-item overhead in a notebook | 64.9 ns | below noise | psnek |
| Cost of one short top-level bar | 33.2 us | 73.3 us | tqdm, 2.2x less |
| Cost of one short nested bar | 29.8 us | 10.7 us | psnek, 2.8x less |
| Import time, terminal | 21.6 ms | 28.1 ms | tqdm, 1.3x less |
| Import time, notebook (`tqdm.notebook`) | 59.8 ms | 28.1 ms | psnek, 2.1x less |
| Terminal bytes, 3 s bar (color / `NO_COLOR`) | 5,674 B | 7,028 / 4,904 B | mixed |
| Notebook traffic, 3 s bar | 118 msgs, 71.8 KB | 19 msgs, 33.6 KB | psnek, 2.1x less |
| Saved `.ipynb`, 1 bar, 1 run | 12,366 B | 2,680 B | psnek, 4.6x smaller |
| Saved `.ipynb`, 1 bar, 20 runs | 226,307 B | 2,681 B | psnek, 84x smaller |
| Widget models left in the kernel after 20 runs | 220 | 0 | psnek |
| Extra packages needed for notebooks | ipywidgets and 3 more | none | psnek |

## 1. Loop overhead (terminal)

From `bench.py` with its defaults: 5,000,000 items, 6 rounds, best run kept.
Each library writes to `/dev/null` at its default refresh rate (tqdm 10 Hz,
psnek 12 Hz). "Overhead" is the library's time minus the bare loop. CPU time
matched wall time within 0.1 ns in every case, so neither library hides work
on a background thread.

| Case | Bare loop | tqdm | psnek | tqdm overhead | psnek overhead | psnek vs tqdm |
|---|---:|---:|---:|---:|---:|---:|
| Iterate a list | 3.7 ns | 55.5 ns | 7.5 ns | 51.8 ns | 3.8 ns | 13.7x less |
| Iterate a tuple | 3.8 ns | 57.3 ns | 5.0 ns | 53.5 ns | 1.2 ns | 43.5x less |
| Iterate a range | 7.2 ns | 56.0 ns | 8.2 ns | 48.8 ns | 1.0 ns | 52.1x less |
| Iterate a generator | 17.2 ns | 66.5 ns | 20.8 ns | 49.3 ns | 3.6 ns | 13.7x less |
| Manual `update()` | 17.8 ns | 86.6 ns | 37.2 ns | 68.8 ns | 19.4 ns | 3.5x less |

Per-bar cost, measured over 5,000 bars of 10 items each:

| Case | tqdm | psnek | psnek vs tqdm | tqdm output per bar | psnek output per bar |
|---|---:|---:|---:|---:|---:|
| Top-level bar | 33.3 us | 73.4 us | 2.2x more | 93 chars | 517 chars |
| Nested bar | 29.9 us | 10.8 us | 2.8x less | 86 chars | 0.1 chars |

The short top-level bar is psnek's weakest case. Each finished bar keeps a
colored final line, styled cell by cell, so it writes 5.6x more characters
than tqdm. A nested bar that finishes within one render tick writes nothing.

### Parallel and async helpers

The bench uses each library's usual idiom, so these rows are not a pure
like-for-like comparison. tqdm's `thread_map` submits one future per item,
like `Executor.map`, while `psnek.pmap` sends items in chunks. Most of the
`pmap` speedup comes from chunking, not from progress tracking. The `gather`
row is the closest match: both libraries await every call at once.

| Case | stdlib | tqdm | psnek | psnek vs tqdm |
|---|---:|---:|---:|---:|
| Map over 1 thread, per item | 5,167.7 ns | 6,645.8 ns | 17.4 ns | 381.3x faster |
| `gather` every call at once, per item | 5,986.3 ns | 8,044.8 ns | 6,633.7 ns | 1.2x faster |
| Map, 16 calls at a time, per item | 9,669.5 ns | 12,896.6 ns | 1,467.7 ns | 8.8x faster |

## 2. Overhead inside a Jupyter kernel

From `footprint.py notebook`. A 1,000,000-item `range` loop runs inside a real
ipykernel, comparing `tqdm.notebook.tqdm` with `psnek.track`. Each cell times
5 runs and keeps the fastest. The best of 3 fresh kernels is reported.

| Library | Bare loop | Loop with bar | Overhead per item |
|---|---:|---:|---:|
| tqdm.notebook | 8.1 ns | 72.9 ns | 64.9 ns |
| psnek | 8.9 ns | 7.9 ns | below noise (-1.0 ns) |

## 3. Import cost

From `footprint.py imports`. Each import runs in a fresh interpreter with a
warm bytecode cache, 30 rounds. These were measured outside a kernel. Inside
Jupyter, IPython is already loaded, so the extra cost of `tqdm.notebook` there
is mostly ipywidgets.

| Import | Median | Min | Memory added (RSS) | Modules loaded |
|---|---:|---:|---:|---:|
| `from tqdm import tqdm` | 21.6 ms | 20.7 ms | 9.0 MiB | 160 |
| `import psnek` | 28.1 ms | 26.9 ms | 10.5 MiB | 181 |
| `from tqdm.auto import tqdm` | 28.4 ms | 26.8 ms | 11.4 MiB | 206 |
| `from tqdm.notebook import tqdm` | 59.8 ms | 57.6 ms | 16.2 MiB | 312 |

In a terminal, tqdm imports 1.3x faster. For notebooks, psnek's single import
costs 2.1x less than `tqdm.notebook`.

Import time on this laptop varies by run. An earlier run the same day measured
28.6 ms for tqdm and 48.9 ms for psnek. tqdm was faster in both runs.

## 4. Terminal bandwidth

From `footprint.py terminal`. Bytes the child process wrote to a real 100x30
pseudo-terminal (`TERM=xterm-256color`), median of 3 rounds. psnek uses true
color by default; the `NO_COLOR` column shows the same output without color
codes, which is the closest match to tqdm's uncolored bar.

| Case | tqdm | psnek | psnek, `NO_COLOR` |
|---|---:|---:|---:|
| One 3 s bar (300 items x 10 ms) | 5,674 B | 7,028 B (1.24x) | 4,904 B (0.86x) |
| Fast 2M-item loop | 460 B | 338 B (0.73x) | 215 B (0.47x) |
| 30 sequential bars (10 items x 10 ms each) | 15,080 B | 19,641 B (1.30x) | 12,777 B (0.85x) |
| Nested: 10 outer x 30 inner x 10 ms | 10,244 B | 11,767 B (1.15x) | 7,173 B (0.70x) |
| 300 x 10 ms with `print()` every 10 items | 5,947 B | 16,685 B (2.81x) | 11,309 B (1.90x) |

Ratios are relative to tqdm. psnek's color codes are the whole difference:
without them, psnek writes 14–53% fewer bytes in every case except the one
with `print()`.

In the `print()` case, the two libraries are not doing the same work. A bare
`print()` breaks tqdm's bar, which leaves partial bar lines in the scrollback
and only redraws the bar at its next refresh. psnek captures the printed
lines, writes them above the bar, and redraws the bar under them.

## 5. Notebook bandwidth and saved file size

From `footprint.py notebook`. Each case runs as one cell under nbclient in a
fresh kernel. Traffic counts every message the kernel sends to the frontend
for that cell (excluding status messages), with each message measured as
compact JSON. The saved size is the whole `.ipynb` in UTF-8, as nbclient
writes it, which includes widget state. The same cell is then rerun in the
same kernel, for 20 runs in total, and the notebook is saved again.

### Traffic for one run

| Case | tqdm.notebook msgs | psnek msgs | tqdm.notebook bytes | psnek bytes | psnek vs tqdm (bytes) |
|---|---:|---:|---:|---:|---:|
| One 3 s bar (300 x 10 ms) | 118 | 19 | 71,795 | 33,551 | 2.1x less |
| 50 sequential bars (20 x 2 ms each) | 950 | 14 | 863,121 | 140,013 | 6.2x less |
| Nested: 10 outer x 30 inner x 5 ms | 266 | 11 | 221,068 | 28,252 | 7.8x less |
| 300 x 10 ms with `print()` every 10 | 133 | 47 | 78,759 | 46,161 | 1.7x less |

### Saved notebook size and the rerun bloat bug

| Case | tqdm, 1 run | psnek, 1 run | tqdm, 20 runs | psnek, 20 runs | tqdm widget models after 20 runs |
|---|---:|---:|---:|---:|---:|
| One 3 s bar | 12,366 B | 2,680 B | 226,307 B | 2,681 B | 220 |
| 50 sequential bars | 581,339 B | 32,549 B | 11,280,050 B | 32,550 B | 11,000 |
| Nested bars | 128,449 B | 2,658 B | 2,481,182 B | 2,659 B | 2,420 |
| `print()` every 10 | 14,334 B | 5,779 B | 228,275 B | 5,954 B | 220 |

tqdm.notebook builds each bar from 11 ipywidgets models and never closes the
ones it leaves on screen. The models stay alive in the kernel, and any
frontend that saves widget state writes every one into the notebook. This
includes nbclient, `nbconvert --execute`, papermill, and JupyterLab with "Save
Widget State Automatically" turned on. As a result the file grows with every
rerun, about 11 KB per bar per run in the cases above. Even after a single
run, widget state is at least 57% of tqdm's saved file (measured as compact
JSON, so it understates the share).

psnek uses no widgets. Each cell gets one HTML output (with a plain-text
fallback) that is updated in place, so a rerun replaces it, and the saved size
stays the same however many times the cell runs. Between the first and the
20th run, the `print()` case changed by 175 bytes and every other case by 1
byte. This is covered by
`test_rerunning_a_cell_in_a_kernel_saves_no_widgets_and_no_growth` in
`tests/test_notebook_cells.py`.

### What a saved notebook shows without a live kernel

This matters on GitHub, nbviewer, or after a kernel restart, where widgets
can't render and only the saved text remains:

| Case | tqdm.notebook saved text | psnek saved text |
|---|---|---|
| One 3 s bar | `train:   0%\|          \| 0/300 [00:00<?, ?it/s]` | `# train ####...  100.0%  300/300  0:03  83.1it/s  done` |
| Nested bars | `inner:   0%\|          \| 0/30 [00:00<?, ?it/s]` | `# outer ####...  100.0%  10/10  0:01  5.4it/s  done` |

tqdm's saved text is the bar's first frame, so a finished loop reads as 0%.
psnek saves the final state as both colored HTML and plain text.

The libraries keep different things by default. tqdm.notebook keeps every bar,
including inner bars of a nested loop (unless you pass `leave=False`). psnek
keeps the outer bar, plus the last 20 finished bars in a cell, and shows a
count such as "+30 earlier bars" for the rest. Some of psnek's size advantage
in the 50-bar and nested cases comes from keeping fewer bars. Even so, tqdm's
widget state alone for 50 bars (407,283 B as compact JSON) is 12.5x psnek's
whole saved file.

## 6. Package footprint

| Attribute | tqdm | psnek |
|---|---:|---:|
| Required dependencies | none (colorama on Windows) | none |
| Extra packages for notebook bars | ipywidgets, widgetsnbextension (4.4 MiB), jupyterlab_widgets, comm, plus a frontend extension | none |
| Source size | 162.6 KB in 31 files | 224.3 KB in 18 files |

tqdm's file count includes integrations psnek doesn't have (Keras, Dask,
Telegram, Slack, Discord, Tk, rich).

## Where tqdm is better

- **Import time in a terminal script:** 21.6 ms vs 28.1 ms.
- **Many short top-level bars:** 33 us vs 73 us per bar, with 5.6x less output.
- **Terminal bytes with color on:** psnek writes 15–30% more in every case
  except the fast loop, because of its true-color styling. Set `NO_COLOR` to
  make it smaller than tqdm.
- **Interleaved `print()` in a terminal:** psnek writes 1.9–2.8x more bytes,
  because it redraws the bar under each printed line instead of letting the
  line break it.
