"""Tests for the benchmark harness in ``benchmarks/``."""

import importlib.util
import io
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import TextIO, cast

import pytest

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"


def _load(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load("bench")


@pytest.fixture
def worker() -> types.ModuleType:
    pytest.importorskip("tqdm")
    return _load("_worker")


def test_ratio_and_chars_formatting() -> None:
    # Act
    cheaper = bench._ratio(10.0, 5.0, ("less", "more"))
    dearer = bench._ratio(5.0, 10.0, ("less", "more"))
    noisy = [bench._ratio(*costs, ("less", "more")) for costs in ((0, 5), (5, -1))]
    chars = bench._chars(1234.5)

    # Assert
    assert (cheaper, dearer) == ("2.0x less", "2.0x more")
    assert noisy == ["n/a (below noise)"] * 2
    assert chars == "1.23e+03 ch"


def test_best_cycles_through_orders_and_keeps_each_fastest_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    calls: list[str] = []

    def measure(library: str, kind: str, items: int) -> tuple[float, float, float]:
        calls.append(library)
        return float(len(calls)), 0.0, 0.0

    monkeypatch.setattr(bench, "measure", measure)

    # Act
    best = bench._best("list", 100, 2)

    # Assert
    assert calls == ["baseline", "tqdm", "psnek", "baseline", "psnek", "tqdm"]
    assert best == {
        "baseline": (1.0, 0.0, 0.0),
        "tqdm": (2.0, 0.0, 0.0),
        "psnek": (3.0, 0.0, 0.0),
    }


def test_measure_scrubs_the_environment_and_parses_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    seen: dict[str, object] = {}

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        seen.update(options, command=command)
        return subprocess.CompletedProcess(command, 0, stdout="1.5 2.5 3.0\n")

    monkeypatch.setenv("PSNEK_TICK", "1")
    monkeypatch.setenv("TQDM_MININTERVAL", "1")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(bench.subprocess, "run", run)

    # Act
    result = bench.measure("psnek", "list", 1000)

    # Assert
    environment = cast("dict[str, str]", seen["env"])
    assert result == (1.5, 2.5, 3.0)
    assert seen["command"] == [
        sys.executable,
        str(bench.WORKER),
        "psnek",
        "list",
        "1000",
    ]
    assert "PSNEK_TICK" not in environment
    assert "TQDM_MININTERVAL" not in environment
    assert "COLUMNS" not in environment
    assert environment["PSNEK_MODE"] == "live"


def test_main_rejects_too_few_items(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(sys, "argv", ["bench.py", "--items", "10"])

    # Act / Assert
    with pytest.raises(SystemExit):
        bench.main()


def test_counting_sink_counts_and_forwards(worker: types.ModuleType) -> None:
    # Arrange
    stream = io.StringIO()
    sink = worker.CountingSink(stream)

    # Act
    sink.write("abc")
    sink.write("de")

    # Assert
    assert sink.written == 5
    assert stream.getvalue() == "abcde"
    assert sink.getvalue() == "abcde"


def test_worker_rejects_bad_item_counts(
    worker: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for count in ("\u00b20", "-50", "5", "many"):
        # Arrange
        monkeypatch.setattr(sys, "argv", ["_worker.py", "psnek", "list", count])

        # Act / Assert
        with pytest.raises(SystemExit):
            worker.main()


def test_every_case_processes_every_item(worker: types.ModuleType) -> None:
    # Arrange
    kinds = (
        *("list", "tuple", "range", "generator", "update", "short", "nested"),
        *("pmap", "gather", "amap"),
    )
    stream = cast("TextIO", io.StringIO())

    # Act
    processed = {
        (library, kind): worker._case(library, kind, list(range(40)), stream)()
        for library in ("baseline", "tqdm", "psnek")
        for kind in kinds
    }

    # Assert
    assert set(processed.values()) == {40}


def test_main_prints_every_table(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    costs = {"baseline": 1.0, "tqdm": 5.0, "psnek": 3.0}

    def measure(library: str, kind: str, items: int) -> tuple[float, float, float]:
        return costs[library], costs[library] * 2, 0.5

    monkeypatch.setattr(bench, "measure", measure)
    monkeypatch.setattr(sys, "argv", ["bench.py", "--items", "500", "--repeats", "1"])

    # Act
    bench.main()

    # Assert
    output = capsys.readouterr().out
    assert "| iterate a list | 1.0 ns | 5.0 ns | 3.0 ns | 2.0x less |" in output
    assert "| nested bar of 10 items | 0.0 us | 0.0 us | 0.0 us |" in output
    assert "| map over 1 thread | 1.0 ns | 5.0 ns | 3.0 ns | 1.7x faster |" in output
    assert "| map, 16 calls at a time | 1.0 ns | 5.0 ns | 3.0 ns | 1.7x faster |" in (
        output
    )


def test_worker_prints_its_cost_per_unit(
    worker: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    monkeypatch.setattr(sys, "argv", ["_worker.py", "baseline", "short", "100"])

    # Act
    worker.main()

    # Assert
    wall, cpu, written = map(float, capsys.readouterr().out.split())
    assert wall >= 0
    assert cpu >= 0
    assert written == 0


def test_worker_rejects_a_run_that_skipped_items(
    worker: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(sys, "argv", ["_worker.py", "baseline", "list", "100"])
    monkeypatch.setattr(worker, "_case", lambda *args: lambda: 99)

    # Act / Assert
    with pytest.raises(RuntimeError, match="processed 99 of 100"):
        worker.main()


@pytest.fixture
def footprint() -> types.ModuleType:
    for module in ("tqdm", "nbclient", "orjson"):
        pytest.importorskip(module)
    return _load("footprint")


def test_human_environment_drops_agent_ci_and_library_variables(
    footprint: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    for name in ("CLAUDECODE", "CI", "PSNEK_MODE", "TQDM_MININTERVAL"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("HOME", "/home/someone")

    # Act
    environment = footprint.human_environment(NO_COLOR="1")

    # Assert
    assert environment["PATH"] == os.defpath
    assert environment["HOME"] == "/home/someone"
    assert environment["NO_COLOR"] == "1"
    assert not {"CLAUDECODE", "CI", "PSNEK_MODE", "TQDM_MININTERVAL"} & set(environment)


def test_in_turns_reverses_the_order_every_other_round(
    footprint: types.ModuleType,
) -> None:
    # Act
    order = list(footprint.in_turns(["a", "b", "c"], 3))

    # Assert
    assert order == ["a", "b", "c", "c", "b", "a", "a", "b", "c"]


def test_measure_import_parses_the_probe_in_a_clean_interpreter(
    footprint: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    seen: dict[str, object] = {}

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        seen.update(options, command=command)
        return subprocess.CompletedProcess(command, 0, stdout="0.025 2048 160\n")

    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setattr(footprint.subprocess, "run", run)

    # Act
    cost = footprint.measure_import("import psnek")

    # Assert
    command = cast("list[str]", seen["command"])
    assert cost == (0.025, 2048 * footprint.RSS_UNIT, 160)
    assert command[:2] == [sys.executable, "-c"]
    assert "import psnek" in command[2]
    assert "CLAUDECODE" not in cast("dict[str, str]", seen["env"])


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX pseudo-terminal")
def test_bytes_in_terminal_counts_what_a_process_writes_to_a_terminal(
    footprint: types.ModuleType,
) -> None:
    # Arrange
    code = (
        "import shutil, sys\n"
        "if sys.stdout.isatty() and shutil.get_terminal_size() == (100, 30):\n"
        "    sys.stdout.write('x' * 5000)\n"
    )

    # Act
    written = footprint.bytes_in_terminal(code, footprint.human_environment())

    # Assert
    assert written == 5000


def test_message_bytes_counts_the_json_parts_and_buffers(
    footprint: types.ModuleType,
) -> None:
    # Arrange
    message = {
        "header": {"msg_type": "stream"},
        "parent_header": {},
        "metadata": {},
        "content": {"text": "hi"},
        "buffers": [b"1234"],
    }
    expected = len('[{"msg_type":"stream"},{},{},{"text":"hi"}]') + 4

    # Act
    size = footprint.message_bytes(message)

    # Assert
    assert size == expected


def test_traffic_client_counts_messages_other_than_status(
    footprint: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(footprint.NotebookClient, "process_message", lambda *args: None)
    client = footprint.TrafficClient(footprint.one_cell(""))
    parts = ("header", "parent_header", "metadata", "content")
    content: dict[str, object] = {part: {} for part in parts}
    cell = client.nb.cells[0]

    # Act
    for msg_type in ("status", "execute_input", "display_data", "stream"):
        client.process_message({**content, "msg_type": msg_type}, cell, 0)

    # Assert
    assert client.messages == 2
    assert client.sent_bytes == 2 * footprint.message_bytes(content)


def test_saved_size_includes_the_widget_state_the_client_stores(
    footprint: types.ModuleType,
) -> None:
    # Arrange
    notebook = footprint.one_cell("pass")
    state: dict[str, object] = {footprint.WIDGET_STATE: {"state": {"a": {}, "b": {}}}}

    class Client:
        def set_widgets_metadata(self) -> None:
            notebook.metadata["widgets"] = state

    plain = len(footprint.nbformat.writes(footprint.one_cell("pass")).encode())

    # Act
    saved, widget_bytes, models = footprint.saved_size(notebook, Client())

    # Assert
    assert models == 2
    assert widget_bytes == len(footprint.orjson.dumps(state))
    assert saved > plain + widget_bytes


def test_final_text_is_the_last_line_of_the_last_display(
    footprint: types.ModuleType,
) -> None:
    # Arrange
    cell = footprint.one_cell("").cells[0]
    nbformat = footprint.nbformat
    cell.outputs = [
        nbformat.v4.new_output("display_data", {"text/plain": "first"}),
        nbformat.v4.new_output("display_data", {"text/plain": "+3 earlier\nlast"}),
        nbformat.v4.new_output("stream", name="stdout", text="printed"),
    ]

    # Act
    text = footprint.final_text(cell)
    empty = footprint.final_text(footprint.one_cell("").cells[0])

    # Assert
    assert (text, empty) == ("last", "")


def test_sections_print_markdown_tables(
    footprint: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    report = footprint.CellReport(118, 76_922, 12_354, 8_906, 11, 226_067, 220, "0%|")
    cost = footprint.ImportCost(0.0286, 2 * 2**20, 160)
    monkeypatch.setattr(footprint, "measure_import", lambda statement: cost)
    monkeypatch.setattr(footprint, "bytes_in_terminal", lambda code, env: 5674)
    monkeypatch.setattr(footprint, "run_cell", lambda source: report)
    monkeypatch.setattr(footprint, "kernel_overhead", lambda header: (8.3, 78.1))

    # Act
    footprint.imports()
    footprint.terminal()
    footprint.notebook()

    # Assert
    output = capsys.readouterr().out
    assert "| `import psnek` | 28.6 ms | 28.6 ms | 2.0 MiB | 160 |" in output
    assert "| fast 2M-item loop | 5,674 | 5,674 | 5,674 |" in output
    assert (
        "| one 3 s bar (300 x 10 ms) | psnek | 118 | 76,922 | 12,354 | 8,906 | 11 "
        "| 226,067 | 220 |"
    ) in output
    assert "| one 3 s bar (300 x 10 ms) | tqdm.notebook | `0%\\|` |" in output
    assert "| psnek | 8.3 ns | 78.1 ns | 69.8 ns |" in output


def test_main_runs_the_chosen_sections_in_order(
    footprint: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    ran: list[str] = []
    sections = {name: lambda name=name: ran.append(name) for name in footprint.SECTIONS}
    monkeypatch.setattr(footprint, "SECTIONS", sections)
    monkeypatch.setattr(sys, "argv", ["footprint.py", "notebook", "imports"])

    # Act
    footprint.main()

    # Assert
    assert ran == ["notebook", "imports"]
    assert capsys.readouterr().out == "Notebook:\n\n\nImports:\n\n"


def test_main_runs_every_section_by_default_and_rejects_unknown_ones(
    footprint: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    ran: list[str] = []
    sections = {name: lambda name=name: ran.append(name) for name in footprint.SECTIONS}
    monkeypatch.setattr(footprint, "SECTIONS", sections)
    monkeypatch.setattr(sys, "argv", ["footprint.py"])

    # Act
    footprint.main()
    monkeypatch.setattr(sys, "argv", ["footprint.py", "imports", "speed"])

    # Assert
    assert ran == ["imports", "terminal", "notebook"]
    with pytest.raises(SystemExit):
        footprint.main()
