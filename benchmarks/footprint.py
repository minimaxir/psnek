"""Import cost, terminal bytes, and notebook traffic of psnek versus tqdm.

These complement ``bench.py``'s loop overhead with what each library costs
around the loop, printed as Markdown tables:

- ``imports``: time, resident memory, and modules added by each import, each
  in a fresh interpreter.
- ``terminal``: bytes each library writes to a real 100x30 pseudo-terminal
  (``TERM=xterm-256color``) for the same loops, psnek with and without color.
- ``notebook``: kernel messages and bytes sent to the frontend for one cell,
  the saved ``.ipynb`` size (widget state included, as nbclient, papermill,
  and JupyterLab's "Save Widget State Automatically" store it) after one run
  and after `RERUNS` runs in the same kernel, and per-item overhead inside a
  kernel; ``tqdm.notebook`` against ``psnek``.

Every run starts a fresh process or kernel with a minimal environment, so
neither library sees CI or coding-agent variables; runs are sequential, and
the libraries take turns in alternating order. The terminal section needs a
POSIX pseudo-terminal, and the notebook section needs ``ipykernel`` and
``ipywidgets`` installed beside the ``bench`` group.

Usage::

    uv run --group bench python benchmarks/footprint.py [SECTION ...]
"""

import argparse
import fcntl
import os
import pty
import statistics
import struct
import subprocess
import sys
import termios
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import nbformat
import orjson
from nbclient import NotebookClient
from nbformat import NotebookNode

KEPT_VARIABLES = ("HOME", "LANG", "TMPDIR")
RSS_UNIT = 1 if sys.platform == "darwin" else 1024
MIB = 1024 * 1024
IMPORT_ROUNDS, TERMINAL_ROUNDS, OVERHEAD_KERNELS, RERUNS = 30, 3, 3, 20
TERMINAL_SIZE = struct.pack("HHHH", 30, 100, 0, 0)
WIDGET_STATE = "application/vnd.jupyter.widget-state+json"
UNCOUNTED_MESSAGES = ("status", "execute_input")
TEXT_SHOWN = 90

IMPORTS = {
    "nothing": "pass",
    "from tqdm import tqdm": "from tqdm import tqdm",
    "import psnek": "import psnek",
    "from tqdm.auto import tqdm": "from tqdm.auto import tqdm",
    "from tqdm.notebook import tqdm": "from tqdm.notebook import tqdm",
}
IMPORT_PROBE = """\
import resource, sys, time
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
start = time.perf_counter()
{statement}
elapsed = time.perf_counter() - start
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(elapsed, after - before, len(sys.modules))
"""

TERMINAL_LOOPS = {
    "one 3 s bar (300 x 10 ms)": "for _ in BAR(range(300)):\n    time.sleep(0.01)",
    "fast 2M-item loop": "for _ in BAR(range(2_000_000)):\n    pass",
    "30 sequential bars (10 x 10 ms each)": (
        "for epoch in range(30):\n"
        "    for _ in BAR(range(10), desc=f'epoch {epoch}'):\n"
        "        time.sleep(0.01)"
    ),
    "nested: 10 outer x 30 inner x 10 ms": (
        "for _ in BAR(range(10), desc='outer'):\n"
        "    for _ in BAR(range(30), desc='inner'):\n"
        "        time.sleep(0.01)"
    ),
    "300 x 10 ms with print() every 10": (
        "for step in BAR(range(300)):\n"
        "    time.sleep(0.01)\n"
        "    if step % 10 == 0:\n"
        "        print(f'step {step}')"
    ),
}
TERMINAL_LIBRARIES = {
    "tqdm": ("from tqdm import tqdm as BAR", {}),
    "psnek": ("from psnek import track as BAR", {}),
    "psnek, NO_COLOR": ("from psnek import track as BAR", {"NO_COLOR": "1"}),
}

NOTEBOOK_CELLS = {
    "one 3 s bar (300 x 10 ms)": (
        "for _ in BAR(range(300), desc='train'):\n    time.sleep(0.01)"
    ),
    "50 sequential bars (20 x 2 ms each)": (
        "for epoch in range(50):\n"
        "    for _ in BAR(range(20), desc=f'epoch {epoch}'):\n"
        "        time.sleep(0.002)"
    ),
    "nested: 10 outer x 30 inner x 5 ms": (
        "for _ in BAR(range(10), desc='outer'):\n"
        "    for _ in BAR(range(30), desc='inner'):\n"
        "        time.sleep(0.005)"
    ),
    "300 x 10 ms with print() every 10": (
        "for step in BAR(range(300), desc='train'):\n"
        "    time.sleep(0.01)\n"
        "    if step % 10 == 0:\n"
        "        print(f'step {step}')"
    ),
}
NOTEBOOK_LIBRARIES = {
    "tqdm.notebook": "from tqdm.notebook import tqdm as BAR",
    "psnek": "from psnek import track as BAR",
}
OVERHEAD_CELL = """\
def fastest(loop, items=1_000_000):
    times = []
    for _ in range(5):
        start = time.perf_counter()
        loop(items)
        times.append(time.perf_counter() - start)
    return min(times) / items * 1e9

def bare(items):
    for _ in range(items):
        pass

def barred(items):
    for _ in BAR(range(items)):
        pass

print(fastest(bare), fastest(barred))
"""


class ImportCost(NamedTuple):
    """What one import added to a fresh interpreter."""

    seconds: float
    rss_bytes: float
    modules: int


@dataclass(slots=True)
class CellReport:
    """What one notebook cell sent and saved."""

    messages: int
    sent_bytes: int
    saved_bytes: int
    widget_bytes: int
    widget_models: int
    rerun_saved_bytes: int
    rerun_widget_models: int
    final_text: str


class TrafficClient(NotebookClient):
    """A notebook client that tallies the kernel messages its cells receive."""

    messages = 0
    sent_bytes = 0

    def process_message(
        self, msg: dict[str, Any], cell: NotebookNode, cell_index: int
    ) -> NotebookNode | None:
        """Count ``msg`` and its JSON size, then process it as usual.

        Args:
            msg: A message from the kernel's output channel.
            cell: The cell being executed.
            cell_index: The cell's position in the notebook.

        Returns:
            The output the message added, if any.
        """
        if msg["msg_type"] not in UNCOUNTED_MESSAGES:
            self.messages += 1
            self.sent_bytes += message_bytes(msg)
        return super().process_message(msg, cell, cell_index)


def human_environment(**extra: str) -> dict[str, str]:
    """Return a minimal environment like a person's shell, plus ``extra``.

    Args:
        **extra: Variables to add.

    Returns:
        ``PATH``, `KEPT_VARIABLES` that are set, and ``extra``; no CI, agent,
        or library configuration variables.
    """
    kept = {name: os.environ[name] for name in KEPT_VARIABLES if name in os.environ}
    return {"PATH": os.defpath, "PYTHONHASHSEED": "0", **kept, **extra}


def in_turns(names: Sequence[str], rounds: int) -> Iterator[str]:
    """Yield ``names`` ``rounds`` times, reversing the order every other round.

    Args:
        names: What takes turns.
        rounds: How many rounds to run.

    Yields:
        Each name once per round.
    """
    for round_index in range(rounds):
        yield from names if round_index % 2 == 0 else reversed(names)


def measure_import(statement: str) -> ImportCost:
    """Time ``statement`` in a fresh interpreter, with the memory it adds.

    Args:
        statement: Python code that imports something.

    Returns:
        The import's cost.

    Raises:
        subprocess.CalledProcessError: If the interpreter fails.
    """
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", IMPORT_PROBE.format(statement=statement)],
        capture_output=True,
        text=True,
        check=True,
        env=human_environment(),
    )
    seconds, rss, modules = result.stdout.split()
    return ImportCost(float(seconds), float(rss) * RSS_UNIT, int(modules))


def bytes_in_terminal(code: str, environment: dict[str, str]) -> int:
    """Run ``code`` on a fresh 100x30 pseudo-terminal and count what it writes.

    Args:
        code: Python source to run in a new interpreter.
        environment: The interpreter's environment.

    Returns:
        Bytes the interpreter wrote to the terminal.
    """
    leader, follower = pty.openpty()
    fcntl.ioctl(follower, termios.TIOCSWINSZ, TERMINAL_SIZE)
    with subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", code],
        stdin=follower,
        stdout=follower,
        stderr=follower,
        env=environment,
    ) as process:
        os.close(follower)
        written = 0
        try:
            while chunk := os.read(leader, 65536):
                written += len(chunk)
        except OSError:  # the terminal closes once the interpreter exits
            pass
        process.wait()
    os.close(leader)
    return written


def message_bytes(msg: dict[str, Any]) -> int:
    """Return the size of a kernel message as JSON, plus its binary buffers.

    Args:
        msg: A message from the kernel's output channel.

    Returns:
        Its size in bytes.
    """
    parts = [msg["header"], msg["parent_header"], msg["metadata"], msg["content"]]
    encoded = orjson.dumps(parts, default=str)
    return len(encoded) + sum(len(buffer) for buffer in msg.get("buffers", ()))


def saved_size(notebook: NotebookNode, client: NotebookClient) -> tuple[int, int, int]:
    """Store the kernel's widget state in ``notebook`` and measure the file.

    Args:
        notebook: The executed notebook.
        client: The client that executed it.

    Returns:
        The saved notebook's bytes, its widget state's bytes, and the number of
        widget models stored.
    """
    client.set_widgets_metadata()
    widgets = notebook.metadata.get("widgets", {})
    models = len(widgets.get(WIDGET_STATE, {}).get("state", {}))
    widget_bytes = len(orjson.dumps(widgets)) if widgets else 0
    return len(nbformat.writes(notebook).encode()), widget_bytes, models


def final_text(cell: NotebookNode) -> str:
    """Return the last line of plain text a cell's last display saves.

    Args:
        cell: An executed code cell.

    Returns:
        Up to `TEXT_SHOWN` characters of the last line of its last display's
        ``text/plain``, or nothing without a display.
    """
    displays = [
        output.get("data", {}).get("text/plain", "")
        for output in cell.outputs
        if output.output_type == "display_data"
    ]
    lines = displays[-1].splitlines() if displays else []
    return lines[-1][:TEXT_SHOWN] if lines else ""


def one_cell(source: str) -> NotebookNode:
    """Return a notebook holding one code cell of ``source``."""
    notebook: NotebookNode = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source)]
    )
    return notebook


def run_cell(source: str) -> CellReport:
    """Run ``source`` as one cell in a fresh kernel, then rerun it there.

    Args:
        source: The cell's code.

    Returns:
        What the first run sent and saved, and what the file held after
        `RERUNS` runs.
    """
    notebook = one_cell(source)
    client = TrafficClient(notebook, kernel_name="python3", timeout=300)
    cell = notebook.cells[0]
    with client.setup_kernel(env=human_environment()):
        client.execute_cell(cell, 0)
        messages, sent = client.messages, client.sent_bytes
        saved, widget_bytes, models = saved_size(notebook, client)
        text = final_text(cell)
        for _ in range(RERUNS - 1):
            client.execute_cell(cell, 0)
        rerun_saved, _, rerun_models = saved_size(notebook, client)
    return CellReport(
        messages, sent, saved, widget_bytes, models, rerun_saved, rerun_models, text
    )


def kernel_overhead(header: str) -> tuple[float, float]:
    """Time a bare and a barred loop in a fresh kernel, nanoseconds per item.

    Args:
        header: Code importing the bar as ``BAR``.

    Returns:
        The bare and the barred loop's fastest time per item.
    """
    source = f"import time\n{header}\n{OVERHEAD_CELL}"
    notebook = one_cell(source)
    client = NotebookClient(notebook, kernel_name="python3", timeout=300)
    with client.setup_kernel(env=human_environment()):
        client.execute_cell(notebook.cells[0], 0)
    printed = "".join(
        output.text
        for output in notebook.cells[0].outputs
        if output.output_type == "stream" and output.name == "stdout"
    )
    bare, barred = map(float, printed.split()[-2:])
    return bare, barred


def _table(
    header: Sequence[str], rows: Sequence[Sequence[object]], labels: int = 1
) -> None:
    """Print a Markdown table, right-aligning the columns after ``labels``.

    Args:
        header: Column names.
        rows: Cells; integers get thousands separators.
        labels: How many leading columns are left-aligned labels.
    """
    print(f"| {' | '.join(header)} |")
    print(f"|{'---|' * labels}{'---:|' * (len(header) - labels)}")
    for row in rows:
        cells = (f"{cell:,}" if isinstance(cell, int) else str(cell) for cell in row)
        print(f"| {' | '.join(cells)} |")


def imports() -> None:
    """Print each import's cost over `IMPORT_ROUNDS` fresh interpreters."""
    costs: dict[str, list[ImportCost]] = {name: [] for name in IMPORTS}
    for name in in_turns(list(IMPORTS), IMPORT_ROUNDS):
        costs[name].append(measure_import(IMPORTS[name]))
    rows = []
    for name, runs in costs.items():
        seconds = [cost.seconds for cost in runs]
        rss = statistics.median(cost.rss_bytes for cost in runs) / MIB
        rows.append(
            (
                f"`{name}`",
                f"{statistics.median(seconds) * 1e3:.1f} ms",
                f"{min(seconds) * 1e3:.1f} ms",
                f"{rss:.1f} MiB",
                runs[0].modules,
            )
        )
    _table(("import", "median", "min", "RSS added", "modules loaded"), rows)


def terminal() -> None:
    """Print the bytes each library writes per loop, median of `TERMINAL_ROUNDS`."""
    rows = []
    for loop, body in TERMINAL_LOOPS.items():
        written: dict[str, list[int]] = {name: [] for name in TERMINAL_LIBRARIES}
        for name in in_turns(list(TERMINAL_LIBRARIES), TERMINAL_ROUNDS):
            header, extra = TERMINAL_LIBRARIES[name]
            environment = human_environment(TERM="xterm-256color", **extra)
            code = f"import time\n{header}\n{body}\n"
            written[name].append(bytes_in_terminal(code, environment))
        medians = [round(statistics.median(sizes)) for sizes in written.values()]
        rows.append((loop, *medians))
    _table(("loop", *TERMINAL_LIBRARIES), rows)


def notebook() -> None:
    """Print each cell's traffic, saved size, and text, then kernel overhead."""
    reports = {
        (case, library): run_cell(f"import time\n{header}\n{body}")
        for case, body in NOTEBOOK_CELLS.items()
        for library, header in NOTEBOOK_LIBRARIES.items()
    }
    header = (
        "cell", "library", "messages", "bytes sent", "saved .ipynb",
        "widget state", "widget models", f"saved after {RERUNS} runs",
        f"widget models after {RERUNS} runs",
    )  # fmt: skip
    rows = [
        (
            *names,
            report.messages,
            report.sent_bytes,
            report.saved_bytes,
            report.widget_bytes,
            report.widget_models,
            report.rerun_saved_bytes,
            report.rerun_widget_models,
        )
        for names, report in reports.items()
    ]
    _table(header, rows, labels=2)
    print()
    texts = [
        (*names, f"`{report.final_text}`".replace("|", "\\|"))
        for names, report in reports.items()
    ]
    _table(("cell", "library", "last saved text line"), texts, labels=3)
    print()
    runs: dict[str, list[tuple[float, float]]] = {
        library: [] for library in NOTEBOOK_LIBRARIES
    }
    for library in in_turns(list(NOTEBOOK_LIBRARIES), OVERHEAD_KERNELS):
        runs[library].append(kernel_overhead(NOTEBOOK_LIBRARIES[library]))
    overheads = []
    for library, found in runs.items():
        bare, barred = min(found, key=lambda run: run[1] - run[0])
        overheads.append(
            (library, f"{bare:.1f} ns", f"{barred:.1f} ns", f"{barred - bare:.1f} ns")
        )
    _table(("library", "bare loop", "loop with bar", "overhead per item"), overheads)


SECTIONS: dict[str, Callable[[], None]] = {
    "imports": imports,
    "terminal": terminal,
    "notebook": notebook,
}


def main() -> None:
    """Run the sections named on the command line (default: all), in order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sections", nargs="*", help=", ".join(SECTIONS))
    chosen = parser.parse_args().sections or list(SECTIONS)
    if unknown := [name for name in chosen if name not in SECTIONS]:
        parser.error(f"unknown sections: {', '.join(unknown)}")
    for index, name in enumerate(chosen):
        if index:
            print()
        print(f"{name.capitalize()}:\n")
        SECTIONS[name]()


if __name__ == "__main__":
    main()
