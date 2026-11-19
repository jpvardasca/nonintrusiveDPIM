from pathlib import Path

import pytest

from nistep.config import ConfigError, config_from_dict, load_config

MIN = {"step": {"d_ref": 1.0}}


def test_minimal_defaults(tmp_path):
    cfg = config_from_dict(MIN, base_dir=tmp_path)
    assert cfg.step.d_ref == 1.0 and cfg.step.eps == 0.24
    assert cfg.step.c == pytest.approx(0.24)
    assert cfg.backend.kind == "analytic"
    assert cfg.engine.mode == "lazy" and cfg.engine.debug is False
    assert cfg.rom.masters == (1,)
    assert cfg.output.dir == (tmp_path.parent / "runs" / "run").resolve()
    assert cfg.cache.dir is None and cfg.cache.lru_items == 64


def test_load_toml_and_relative_paths(tmp_path):
    inp = tmp_path / "deck.inp"
    inp.write_text("*Heading\n")
    (tmp_path / "cfg").mkdir()
    p = tmp_path / "cfg" / "m.toml"
    p.write_text(
        'name = "m"\n'
        "[model]\ninp = \"../deck.inp\"\nclamped_nset = \"Set-1\"\n"
        "[step]\nd_ref = 20\n"
        "[backend]\nkind = \"pysvk\"\n"
        "[rom]\nmasters = [1, 2]\norder = 5\n"
        "[cache]\ndir = \"../cache\"\n"
        "[abaqus]\nuser_lib_dir = \"D:/host/uel_lib\"\n"
    )
    cfg = load_config(p)
    assert cfg.name == "m" and cfg.path == p.resolve()
    assert cfg.model.inp == inp.resolve()
    assert cfg.model.clamped_nset == "Set-1"
    assert cfg.rom.masters == (1, 2) and cfg.rom.order == 5
    assert cfg.cache.dir == (tmp_path / "cache").resolve()
    assert cfg.abaqus.user_lib_dir == "D:/host/uel_lib"  # host path: kept verbatim
    assert cfg.output.dir == (tmp_path / "runs" / "m").resolve()


@pytest.mark.parametrize(
    "patch, msg",
    [
        ({"step": {"d_ref": 0.0}}, "d_ref"),
        ({"step": {"d_ref": 1.0, "eps": 0.25}}, "eps"),
        ({"step": {"d_ref": 1.0, "epz": 0.1}}, "unknown keys"),
        ({"stepp": {}}, "unknown top-level"),
        ({"backend": {"kind": "ansys"}}, "kind"),
        ({"backend": {"kind": "abaqus"}}, "inp is required"),
        ({"rom": {"masters": []}}, "masters"),
        ({"rom": {"masters": [1, 1]}}, "unique"),
        ({"forcing": {"enabled": True}}, "forcing_order"),
        ({"engine": {"mode": "eager"}}, "mode"),
        ({"cache": {"lru_items": 0}}, "lru_items"),
        ({"damping": {"alpha": -1.0}}, "damping"),
        ({"output": {"monitor_dof": 4}}, "monitor_dof"),
    ],
)
def test_validation_errors(tmp_path, patch, msg):
    data = {**MIN, **patch}
    with pytest.raises(ConfigError, match=msg):
        config_from_dict(data, base_dir=tmp_path)


def test_missing_step_section(tmp_path):
    with pytest.raises(ConfigError, match=r"\[step\]"):
        config_from_dict({}, base_dir=tmp_path)


def test_utf8_bom_and_bad_encoding(tmp_path):
    # Windows editors and PowerShell 5.1 write UTF-8 with a byte-order mark
    p = tmp_path / "bom.toml"
    p.write_bytes(b"\xef\xbb\xbf" + 'name = "bom"\n[step]\nd_ref = 2.0\n'.encode())
    cfg = load_config(p)
    assert cfg.name == "bom" and cfg.step.d_ref == 2.0
    p.write_bytes(b"name = \"\xff\"\n[step]\nd_ref = 2.0\n")
    with pytest.raises(ConfigError, match="not UTF-8"):
        load_config(p)


def test_missing_input_file(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        config_from_dict({**MIN, "model": {"inp": "nope.inp"}}, base_dir=tmp_path)


def test_repo_configs_load():
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    files = sorted(cfg_dir.glob("analytic_*.toml"))
    assert files, "expected configs/analytic_*.toml"
    for f in files:
        cfg = load_config(f)
        assert cfg.backend.kind == "analytic"
