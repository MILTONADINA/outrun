"""specs/14-app.md (CLI-*): parsing, dispatch and the thin serve / export-static / disk handlers.

Handlers that do real work (fetch, train, ...) belong to other areas; here they are replaced with fakes.
"""

from __future__ import annotations

import argparse
from collections import namedtuple
from pathlib import Path

import pytest

import outrun
from outrun import cli

HAZARDS = ["storm", "tornado", "flood", "wildfire", "ice", "smoke"]
ROUTE_MIN = ["route", "--place", "Tulsa, OK", "--origin", "-95.99", "36.15", "--nearest-shelter"]
MIN_ARGS = {
    "fetch": ["fetch"],
    "prep": ["prep", "sevir"],
    "train": ["train", "storm"],
    "eval": ["eval"],
    "scenario": ["scenario"],
    "route": ROUTE_MIN,
    "serve": ["serve"],
    "export-static": ["export-static"],
    "disk": ["disk"],
}
Usage = namedtuple("Usage", "total used free")


def _parse(argv):
    return cli.build_parser().parse_args(argv)


def _exit_code(fn, *a):
    with pytest.raises(SystemExit) as e:
        fn(*a)
    return e.value.code


class Recorder:
    def __init__(self, rc=0):
        self.rc = rc
        self.calls: list[argparse.Namespace] = []

    def __call__(self, args):
        self.calls.append(args)
        return self.rc


# ----------------------------------------------------------------- CLI-01
@pytest.mark.req("CLI-01")
@pytest.mark.parametrize("cmd", list(MIN_ARGS))
def test_every_command_parses(cmd):
    assert _parse(MIN_ARGS[cmd]).command == cmd


@pytest.mark.req("CLI-01")
def test_exact_command_set_and_prog():
    p = cli.build_parser()
    assert p.prog == "outrun"
    subs = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(subs) == 1
    assert set(subs[0].choices) == set(MIN_ARGS) == set(cli.COMMANDS)


@pytest.mark.req("CLI-01")
@pytest.mark.parametrize("argv", [["fly-to-the-moon"], []])
def test_unknown_or_missing_command_exits_2(argv):
    assert _exit_code(_parse, argv) == 2


@pytest.mark.req("CLI-01")
def test_version(capsys):
    assert _exit_code(_parse, ["--version"]) == 0
    assert outrun.__version__ in capsys.readouterr().out


# ----------------------------------------------------------------- CLI-02
@pytest.mark.req("CLI-02")
def test_defaults():
    f = _parse(["fetch"])
    assert (f.sources, f.manifest, f.dry_run) == ([], "data/manifest.yaml", False)
    f2 = _parse(["fetch", "--source", "sevir", "--source", "tornet", "--dry-run"])
    assert f2.sources == ["sevir", "tornet"] and f2.dry_run is True
    pr = _parse(["prep", "mrms", "--limit", "5"])
    assert pr.dataset == "mrms" and pr.limit == 5 and _parse(["prep", "x"]).limit is None
    t = _parse(["train", "ice"])
    assert (t.hazard, t.epochs, t.device, t.seed, t.resume) == ("ice", None, "auto", None, False)
    t2 = _parse(["train", "smoke", "--epochs", "3", "--device", "mps", "--seed", "7", "--resume"])
    assert (t2.epochs, t2.device, t2.seed, t2.resume) == (3, "mps", 7, True)
    e = _parse(["eval"])
    assert (e.hazard, e.split, e.routing, e.final) == ("all", "val", False, False)
    s = _parse(["scenario"])
    assert (s.scenario_id, s.all, s.out) == (None, False, "scenarios")
    assert _parse(["scenario", "el-reno-2019"]).scenario_id == "el-reno-2019"
    r = _parse(ROUTE_MIN)
    assert (r.profile, r.risk_source, r.lam, r.scenario, r.out) == ("civilian", "model", None, None, None)
    sv = _parse(["serve"])
    assert (sv.host, sv.port, sv.demo, sv.scenarios, sv.reports) == (
        "127.0.0.1",
        8000,
        False,
        "scenarios",
        "reports",
    )
    assert _parse(["serve", "--port", "9001", "--demo"]).port == 9001
    x = _parse(["export-static"])
    assert (x.out, x.scenarios, x.reports) == ("docs", "scenarios", "reports")
    d = _parse(["disk"])
    assert d.need_gb == 0.0 and _parse(["disk", "--need-gb", "12.5"]).need_gb == pytest.approx(12.5)


@pytest.mark.req("CLI-02")
@pytest.mark.parametrize("hz", HAZARDS)
def test_hazard_choices(hz):
    assert _parse(["train", hz]).hazard == hz
    assert _parse(["eval", "--hazard", hz]).hazard == hz


@pytest.mark.req("CLI-02")
@pytest.mark.parametrize(
    "argv",
    [
        ["train", "hail"],
        ["train"],
        ["eval", "--hazard", "hail"],
        ["eval", "--split", "train"],
        ["train", "storm", "--device", "cuda"],
        ["serve", "--port", "abc"],
        ["prep"],
    ],
)
def test_invalid_choices_exit_2(argv):
    assert _exit_code(_parse, argv) == 2


# ----------------------------------------------------------------- CLI-03
@pytest.mark.req("CLI-03")
def test_handlers_cover_every_command():
    assert set(cli.HANDLERS) == set(cli.COMMANDS)
    assert all(callable(h) for h in cli.HANDLERS.values())


@pytest.mark.req("CLI-03")
@pytest.mark.parametrize("cmd", list(MIN_ARGS))
def test_main_dispatches_at_call_time(monkeypatch, cmd):
    rec = Recorder(rc=7)
    monkeypatch.setitem(cli.HANDLERS, cmd, rec)
    assert cli.main(MIN_ARGS[cmd]) == 7
    assert len(rec.calls) == 1 and rec.calls[0].command == cmd


@pytest.mark.req("CLI-03")
def test_python_dash_m_forwards(monkeypatch):
    import outrun.__main__ as entry

    seen = []
    monkeypatch.setattr(cli, "main", lambda argv=None: seen.append(argv) or 3)
    assert entry.main(["disk"]) == 3
    assert seen == [["disk"]]


# ----------------------------------------------------------------- CLI-04
@pytest.mark.req("CLI-04")
def test_test_split_needs_final(monkeypatch):
    rec = Recorder()
    monkeypatch.setitem(cli.HANDLERS, "eval", rec)
    assert _exit_code(cli.main, ["eval", "--split", "test"]) == 2
    assert rec.calls == []
    assert cli.main(["eval", "--split", "test", "--final"]) == 0
    assert cli.main(["eval", "--split", "val"]) == 0
    assert cli.main(["eval"]) == 0
    assert len(rec.calls) == 3 and rec.calls[0].final is True


# ----------------------------------------------------------------- CLI-05
@pytest.mark.req("CLI-05")
def test_route_parses_place_and_bbox():
    r = _parse(ROUTE_MIN + ["--profile", "ev", "--risk-source", "current-only", "--lambda", "0.5"])
    assert r.place == "Tulsa, OK" and r.origin == pytest.approx([-95.99, 36.15]) and r.nearest_shelter is True
    assert (r.profile, r.risk_source, r.lam) == ("ev", "current-only", pytest.approx(0.5))
    b = _parse(
        [
            "route",
            "--bbox",
            "-97.6",
            "35.4",
            "-97.4",
            "35.6",
            "--origin",
            "-97.5",
            "35.5",
            "--dest",
            "-97.45",
            "35.55",
        ]
    )
    assert b.bbox == pytest.approx([-97.6, 35.4, -97.4, 35.6]) and b.dest == pytest.approx([-97.45, 35.55])
    assert b.place is None and b.nearest_shelter is False


@pytest.mark.req("CLI-05")
@pytest.mark.parametrize(
    "argv",
    [
        ["route", "--origin", "-97.5", "35.5", "--nearest-shelter"],  # neither place nor bbox
        [
            "route",
            "--place",
            "X",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
            "--origin",
            "0.5",
            "0.5",
            "--nearest-shelter",
        ],
        ["route", "--place", "X", "--origin", "-97.5", "35.5"],  # neither dest nor nearest-shelter
        ["route", "--place", "X", "--origin", "-97.5", "35.5", "--dest", "1", "2", "--nearest-shelter"],
        ["route", "--place", "X", "--nearest-shelter"],  # no origin
        ROUTE_MIN + ["--profile", "truck"],
        ROUTE_MIN + ["--risk-source", "oracle"],
        ["route", "--bbox", "1", "2", "3", "--origin", "0", "0", "--nearest-shelter"],
    ],
)
def test_route_argument_violations_exit_2(argv):
    assert _exit_code(_parse, argv) == 2


@pytest.mark.req("CLI-05")
@pytest.mark.parametrize(
    "bbox",
    [
        ["-97.4", "35.4", "-97.6", "35.6"],
        ["-97.6", "35.6", "-97.4", "35.4"],
        ["-97.5", "35.4", "-97.5", "35.6"],
    ],
)
def test_route_bad_bbox_order_exits_before_handler(monkeypatch, bbox):
    rec = Recorder()
    monkeypatch.setitem(cli.HANDLERS, "route", rec)
    argv = ["route", "--bbox", *bbox, "--origin", "-97.5", "35.5", "--nearest-shelter"]
    assert _exit_code(cli.main, argv) == 2
    assert rec.calls == []


# ----------------------------------------------------------------- CLI-06
@pytest.mark.req("CLI-06")
@pytest.mark.parametrize("demo_flag,expected_demo", [(True, True), (False, None)])
def test_serve_wires_create_app_and_uvicorn(monkeypatch, demo_flag, expected_demo):
    import uvicorn

    import outrun.app.server as server

    sentinel = object()
    made, ran = [], []
    monkeypatch.setattr(server, "create_app", lambda **kw: made.append(kw) or sentinel)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: ran.append((app, kw)))
    argv = ["serve", "--port", "8123", "--scenarios", "sc", "--reports", "rp"] + (
        ["--demo"] if demo_flag else []
    )
    assert cli.cmd_serve(_parse(argv)) == 0
    assert made == [{"scenarios_dir": Path("sc"), "reports_dir": Path("rp"), "demo": expected_demo}]
    assert len(ran) == 1 and ran[0][0] is sentinel
    assert ran[0][1] == {"host": "127.0.0.1", "port": 8123}


# ----------------------------------------------------------------- CLI-07
@pytest.mark.req("CLI-07")
def test_export_static_handler(monkeypatch, capsys):
    import outrun.app.export as export

    calls = []
    monkeypatch.setattr(
        export, "export_static", lambda s, r, o: calls.append((s, r, o)) or ["index.html", "a.json", "b.png"]
    )
    assert cli.cmd_export_static(_parse(["export-static", "--out", "site"])) == 0
    assert calls == [(Path("scenarios"), Path("reports"), Path("site"))]
    assert "3" in capsys.readouterr().out


@pytest.mark.req("CLI-07")
def test_export_static_handler_foreign_file(monkeypatch, capsys):
    import outrun.app.export as export

    def boom(*a):
        raise FileExistsError("docs/index.html is not ours")

    monkeypatch.setattr(export, "export_static", boom)
    assert cli.cmd_export_static(_parse(["export-static"])) == 1
    assert "docs/index.html is not ours" in capsys.readouterr().err


# ----------------------------------------------------------------- CLI-08
@pytest.mark.req("CLI-08")
def test_check_disk(monkeypatch, tmp_path):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(500e9, 300e9, 200e9))
    st = cli.check_disk(tmp_path, 50.0, 375.0)
    assert isinstance(st, cli.DiskStatus)
    assert st.path == tmp_path
    assert (st.total_gb, st.used_gb, st.free_gb) == pytest.approx((500.0, 300.0, 200.0))
    assert (st.need_gb, st.max_used_gb, st.ok) == (50.0, 375.0, True)
    assert cli.check_disk(tmp_path, 75.0, 375.0).ok is True  # exactly at the limit
    assert cli.check_disk(tmp_path, 80.0, 375.0).ok is False  # 380 > 375
    assert cli.check_disk(tmp_path, 210.0, 1000.0).ok is False  # more than free space
    with pytest.raises(ValueError):
        cli.check_disk(tmp_path, -1.0, 375.0)
    with pytest.raises(FileNotFoundError):
        cli.check_disk(tmp_path / "missing", 1.0, 375.0)


@pytest.mark.req("CLI-08")
@pytest.mark.parametrize("oks,expected", [((True, True), 0), ((True, False), 1), ((False, True), 1)])
def test_cmd_disk_exit_code(monkeypatch, tmp_dirs, oks, expected):
    seen = []

    def fake_check(path, need_gb, max_used_gb):
        ok = oks[len(seen)] if len(seen) < 2 else True
        seen.append(Path(path))
        return cli.DiskStatus(Path(path), 500.0, 100.0, 400.0, need_gb, max_used_gb, ok)

    monkeypatch.setattr(cli, "check_disk", fake_check)
    assert cli.cmd_disk(_parse(["disk", "--need-gb", "1"])) == expected
    assert {p.resolve() for p in seen} == {tmp_dirs["raw"].resolve(), tmp_dirs["proc"].resolve()}
