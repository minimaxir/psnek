![](docs/psnek.svg)

A modern reimagining of the Python progress*sss* bar.

![](docs/psnek_hello.gif)

- Track any loop with near-zero overhead: no Python code runs per item!
- Bars that turn amber when a loop stalls and red when it's stuck, judged against the loop's own pace.
- An ETA that is marked with a `~` when it can't be trusted.
- Plain status lines in CI and for AI coding agents, instead of a mess of carriage returns.
- One output per Jupyter Notebook cell, updated in place.
- A panel below the bars for `print()`, logging, and live tables, including polars and pandas DataFrames.
- Progress for threads, processes, and asyncio, with support for free-threaded Python.
- Taller bars that span multiple rows.

## Installation

psnek can be installed from PyPI (Python 3.11+):

```sh
pip3 install psnek
```

```sh
uv pip install psnek
```

## Demo

Wrap any iterable with `track()` and optionally give it a description.

```py3
import psnek

for path in psnek.track(paths, "resize"):
    resize(path)
```

For anything else, you can update a `Bar` manually, from any thread. Use `with` so the bar is always closed, and so an exception marks it as failed:

```py3
options = psnek.Options(unit="B", scale="iec")

with psnek.Bar(total=size, desc="upload", options=options) as bar:
    for chunk in chunks:
        send(chunk)
        bar.update(len(chunk))
```

You can add status fields after the metrics with `status()`, either as keywords or as a dict whose keys don't need to be valid Python names. Setting a field to `None` removes it.

```py3
bar.status(loss=0.137, lr=3e-4)
bar.status({"val loss": 0.142, "lr": None})
```

Bars started inside another bar's loop are drawn as a tree below it. By default, only the top-level bar's final line is kept once it finishes.

```py3
for epoch in psnek.track(3, "epochs"):
    for batch in psnek.track(batches, f"epoch {epoch + 1}"):
        ...
```

```
⣄ epochs    ⣿⣿⣿⣿⣿⣿⣿⣿⡆⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀   33.3%        1/3    0:21<0:22   11.1s/it
└─⠋ epoch 2 ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀   41.2%   412/1000    0:09<0:12   45.8it/s
  └─⢰ eval  ⣿⣿⣿⣿⣿⣿⡏⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀   26.6%      17/64    0:00<0:01   41.0it/s
```

To print a line above whatever bars are live:

```py3
psnek.write("finished shard", 3)
```

A bar's progress can be read at any time with `bar.n`, `bar.total`, `bar.rate`, and `bar.elapsed`, and once it closes, `bar.outcome` is one of `"done"`, `"failed"`, or `"interrupted"`. A bar closed by an exception shows its cause after its state, shortened in a narrow terminal so the state and a short bar still fit: `interrupted Ctrl-C` for `KeyboardInterrupt`, `interrupted exit` for `sys.exit()`, `interrupted cancelled` for a cancelled asyncio task, `interrupted closed` for a generator closed early, and the exception's type otherwise, such as `failed ValueError`.

[Jupyter Notebook with a tour of everything above](notebooks/01_tour.ipynb).

## The Snek

Each cell of the bar is an 8-dot Braille character that fills in eight steps, so consecutive cells read as one winding coil: `⠀ ⡀ ⡄ ⡆ ⡇ ⡏ ⡟ ⡿ ⣿`. When the total is unknown, the snake slithers back and forth, led by a face that reflects the run's mood: it smiles (`⢌⣀⡡`) while running, stares wide-eyed (`⠛⣤⠛`) when stalled, and winces (`⠕⣀⠪`) once stuck. If progress stops for an unusually long time for that loop, the snake slows down and the bar turns amber (`stalled 12s`), then red (`stuck 48s`).

## The Panel

While a bar is live in a terminal, anything you `print()` or log is routed into a panel drawn below the bars instead of breaking them. The panel shows at most 8 lines, and older lines move above the bars, so no output is ever lost. You can also write to the panel directly:

```py3
bar.log("processed", path)  # append a message to the panel, like logging
bar.log("loss spiked", level=logging.WARNING)  # with a logging level
bar.panel("worker 1: idle\nworker 2: busy")  # replace the panel's content
bar.panel(None)  # clear it
bar.write("saved model")  # print permanently above the bars, like print()
```

`log()` is for messages about the run: each one starts with its level (`debug`, `info`, `warn`, `error`, or `fatal`), and rises in salience with it, from a faint debug line to bold warnings and errors in the bar's own colors. `Options(log_time=True)` starts each message with the time, and `psnek.LogHandler(log_time=True)` gives records from the `logging` module the same look. Logged messages stay indented under the bars once the panel pushes them above, while `write()` prints its text as-is, flush left:

```
  16:30:33 info  batch 10 ingested
checkpoint saved: ingest-015.parquet
⢰ ingest ⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣀   90.0%      27/30    0:00<0:00   38.5it/s  rows=27000
⡇ 16:30:33 warn  batch 18: 3 malformed rows skipped
⡇ 16:30:33 error upload of batch 24 failed
⡇                retrying in 1s
```

`panel()` also lays out structured data as a borderless table: a dict, dataclass, or named tuple becomes bold keys beside their values, while a list of them (or a polars or pandas `DataFrame`) becomes rows under a bold header. Only the rows that fit are formatted, so it's cheap to call every step.

```py3
bar.panel({"loss": 0.13761, "learning_rate": 3e-4, "epoch": 3})
bar.panel([{"model": "llama", "acc": 0.912}, {"model": "gpt2", "acc": 0.6, "note": "base"}])
```

```
⡇ loss           0.1376
⡇ learning_rate  0.0003000
⡇ epoch          3

⡇ model  acc     note
⡇ llama  0.9120
⡇ gpt2   0.6000  base
```

[Jupyter Notebook](notebooks/02_panel_tables.ipynb)

## CI and AI Agents

Without a terminal (CI, pipes, and AI coding agents such as Claude Code or Codex), psnek prints plain, newline-terminated status lines with no escape codes: a start line, one every 10 seconds, one whenever a bar stalls, gets stuck, or recovers, and a final summary.

```
ingest: 0.0% 0/100 0:00<--:-- --it/s
ingest: 50.0% 50/100 0:06<~0:06 stalled 5s 40.6it/s
ingest: 53.0% 53/100 0:07<~0:06 7.3it/s (recovered)
ingest: 100.0% 100/100 0:08 done 13.9it/s
```

psnek detects common CI and agent environment variables automatically. ([Jupyter Notebook](notebooks/04_resilience.ipynb))

## Parallel and Async

`pmap()` maps a function over items with progress and returns the results in input order. It uses threads by default, but you can pass any executor, such as a `ProcessPoolExecutor`, for CPU-bound work.

```py3
results = psnek.pmap(fetch, urls, "fetch", workers=16)
```

For asyncio, `atrack()` wraps an async iterable, `gather()` works like `asyncio.gather()`, and `amap()` works like `pmap()` with at most `workers` calls running at once. Both `gather()` and `amap()` count each call as it completes, and cancel the remaining calls if one fails.

```py3
async for message in psnek.atrack(stream, "messages"):
    ...

pages = await psnek.gather(*(fetch_page(url) for url in urls), desc="fetch")
pages = await psnek.amap(fetch_page, urls, "fetch", workers=16)
```

Wrapping the inputs of `asyncio.gather()` in `track()` does not work as you'd expect: the bar fills up instantly, since every coroutine is created before any of them runs. ([Jupyter Notebook](notebooks/05_async.ipynb))

On a free-threaded build of Python, such as `python3.14t`, CPU-bound `pmap()` on plain threads scales across cores, and `update()` stays cheap even when many threads call it at once. ([Jupyter Notebook](notebooks/06_free_threading.ipynb))

## Options

`psnek.Options` holds the presentation settings, and can be shared by any number of bars. ([Jupyter Notebook](notebooks/07_options.ipynb))

Some userful options include:

- `unit`: Label for counts and rates (default: `"it"`)
- `scale`: `"si"` (1000) or `"iec"` (1024) number prefixes (default: `"none"`)
- `height`: Rows the bar spans (default: 1)
- `gradient`: Color the bar with its gradient, with your own list or tuple of `"#RRGGBB"` colors, or (False) in the terminal's text color, keeping the stalled, stuck, and failed colors (default: True)
- `disable`: Draw nothing, while counting and text still work (default: False)
- `stream`: Where the bar is drawn (default: `sys.stderr`)
- `notify`: Allow desktop notifications about the bar (default: True)

## Miscellaneous Notes

- The terminal GIFs/videos were made using [vhs](https://github.com/charmbracelet/vhs), which apparently Opus 5.5 can use well. An additional postprocessing step was added to crop the resulting video to only salient dimensions in order to reduce file size. The demo files and `vhs` tapes are available in the [demo folder](/demo).
- A bar is considered stalled once the wait since its last progress is past the smoothed mean of its gaps plus 3 standard deviations (and at least 5 seconds), and stuck past 9 standard deviations (and at least 30 seconds). The first 3 gaps are never flagged, so a slow first step such as loading a model is fine.
- The ETA drops its `~` only when psnek is 99% confident it is within 10% (or a second). During warm-up, irregular loops, changes in pace, and stalls, it keeps the `~`.
- In a terminal, psnek sends a desktop notification when a bar gets stuck, fails, or a long run finishes, on terminals that support it (Ghostty, WezTerm, foot, iTerm2, and kitty). Nothing is sent inside tmux or screen.
- All text psnek draws is sanitized, so captured output can't move the cursor or spoof the display.

## Development

```sh
uv sync --group dev
uv run pytest
uv run mypy
uv run ruff check && uv run ruff format --check
```

To rerun the notebooks:

```sh
uv sync --group notebooks
uv pip install ipykernel
uv run jupyter execute --inplace notebooks/0[1-57]_*.ipynb
```

The `06_free_threading` notebook needs a kernel running a free-threaded build of Python, set up as its first cell describes.

## Maintainer/Creator

Max Woolf ([@minimaxir](https://minimaxir.com))

_Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir) and [GitHub Sponsors](https://github.com/sponsors/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use._

## License

MIT
