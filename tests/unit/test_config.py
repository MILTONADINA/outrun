"""specs/01-data.md DATA-15, DATA-16: paths, YAML config, seeds, device."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from outrun import config
from outrun.config import Paths, get_device, get_paths, load_config, seed_everything

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- DATA-15
@pytest.mark.req("DATA-15")
def test_repo_root_contains_pyproject():
    assert config.REPO_ROOT == ROOT
    assert (config.REPO_ROOT / "pyproject.toml").exists()


@pytest.mark.req("DATA-15")
def test_default_paths_without_env():
    p = get_paths({})
    assert isinstance(p, Paths)
    assert p.raw == Path("/Volumes/OutrunRaw")
    assert p.proc == ROOT / "data" / "proc"
    assert p.configs == ROOT / "configs"
    assert p.root == ROOT
    for name in ("splits", "scenarios", "models", "logs"):
        assert getattr(p, name) == ROOT / name


@pytest.mark.req("DATA-15")
def test_env_overrides_and_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = {
        "OUTRUN_RAW": "rawdir",
        "OUTRUN_PROC": str(tmp_path / "proc"),
        "OUTRUN_CONFIG_DIR": str(tmp_path / "c"),
    }
    p = get_paths(env)
    assert p.raw == (tmp_path / "rawdir").resolve()
    assert p.proc == (tmp_path / "proc").resolve()
    assert p.configs == (tmp_path / "c").resolve()
    assert all(getattr(p, f).is_absolute() for f in Paths.__dataclass_fields__)
    assert not (tmp_path / "rawdir").exists()  # nothing created


@pytest.mark.req("DATA-15")
def test_empty_env_value_is_unset_and_tilde_expanded():
    p = get_paths({"OUTRUN_RAW": "", "OUTRUN_PROC": "~/outrun_proc_x"})
    assert p.raw == Path("/Volumes/OutrunRaw")
    assert p.proc == (Path.home() / "outrun_proc_x").resolve()


@pytest.mark.req("DATA-15")
def test_reads_os_environ_when_env_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTRUN_RAW", str(tmp_path))
    assert get_paths().raw == tmp_path.resolve()


# ---------------------------------------------------------------- DATA-16
@pytest.mark.req("DATA-16")
def test_load_config_reads_yaml(tmp_path):
    (tmp_path / "x.yaml").write_text("a: 1\nb: {c: [1, 2]}\n")
    assert load_config("x", config_dir=tmp_path) == {"a": 1, "b": {"c": [1, 2]}}
    assert load_config("x.yaml", config_dir=tmp_path) == {"a": 1, "b": {"c": [1, 2]}}


@pytest.mark.req("DATA-16")
def test_load_config_fresh_copy(tmp_path):
    (tmp_path / "x.yaml").write_text("a: {b: 1}\n")
    first = load_config("x", config_dir=tmp_path)
    first["a"]["b"] = 99
    assert load_config("x", config_dir=tmp_path)["a"]["b"] == 1


@pytest.mark.req("DATA-16")
def test_load_config_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config("nope", config_dir=tmp_path)
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        load_config("list", config_dir=tmp_path)


@pytest.mark.req("DATA-16")
def test_committed_configs_load(monkeypatch):
    monkeypatch.delenv("OUTRUN_CONFIG_DIR", raising=False)
    assert load_config("domains")["grid"]["crs"] == "EPSG:5070"
    assert "hazards" in load_config("hazards")
    assert isinstance(load_config("planning"), dict)


@pytest.mark.req("DATA-16")
def test_seed_everything_reproducible():
    seed_everything(123)
    a = (random.random(), np.random.rand(3).tolist())
    seed_everything(123)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b
    seed_everything(124)
    assert (random.random(), np.random.rand(3).tolist()) != a


@pytest.mark.req("DATA-16")
def test_get_device_env_override():
    assert get_device({"OUTRUN_DEVICE": "cpu"}) == "cpu"
    assert get_device({"OUTRUN_DEVICE": "mps"}) == "mps"
    with pytest.raises(ValueError):
        get_device({"OUTRUN_DEVICE": "cuda"})
    assert get_device({}) in ("cpu", "mps")
