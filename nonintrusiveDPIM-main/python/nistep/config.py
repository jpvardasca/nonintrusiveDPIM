"""Run configuration: frozen dataclasses loaded from a TOML file.

Rules
-----
* Every section is optional except ``[step]`` (``d_ref`` has no default).
* Unknown keys are an error, so typos are never silently ignored.
* Local paths are resolved relative to the directory of the config file.
  Paths that live on the Abaqus host (``abaqus.user_lib_dir``, ``abaqus.mass_lib_dir``)
  are kept as plain strings and are never resolved or checked here.

Example (see ``configs/*.toml``)::

    name = "analytic_two_dof"
    [step]
    d_ref = 1.0
    [backend]
    kind = "analytic"
    [analytic]
    model = "two_dof"
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

BACKEND_KINDS = ("analytic", "pysvk", "replay", "abaqus")
RUNNER_KINDS = ("local", "spool", "ssh")
ANALYTIC_MODELS = ("two_dof", "random_sparse", "spring_chain", "duffing")
ENGINE_MODES = ("lazy", "record")


class ConfigError(ValueError):
    """Raised for any invalid or inconsistent configuration."""


# --------------------------------------------------------------------------- sections


@dataclass(frozen=True)
class ModelCfg:
    inp: Path | None = None  # base Abaqus deck (needed for pysvk / abaqus)
    clamped_nset: str = "Set-2"  # name of the clamped node set in the deck
    stiffness_mtx: Path | None = None  # optional processed "i j v" stiffness (all nodes)
    mass_mtx: Path | None = None  # optional processed "i j v" mass (all nodes)


@dataclass(frozen=True)
class DampingCfg:
    alpha: float = 0.0  # C = alpha*M + beta*K
    beta: float = 0.0


@dataclass(frozen=True)
class StepCfg:
    d_ref: float  # reference displacement (thesis "minimum distance")
    eps: float = 0.24  # c = eps * d_ref
    zero_tol: float = 1e-13  # part treated as zero if max|part| <= zero_tol*max|a|
    match_tol: float = 1e-9  # canonical-direction matching tolerance (inf-norm)
    seed: int = 12345  # seed of the sign-fixing random vector w (CONVENTIONS 4.5)

    @property
    def c(self) -> float:
        """Probe amplitude c = eps * d_ref (CONVENTIONS 4.3)."""
        return self.eps * self.d_ref


@dataclass(frozen=True)
class EngineCfg:
    mode: str = "lazy"  # "lazy" | "record"
    debug: bool = False  # shape/finiteness checks + comparison with backend.exact_*
    # per-order batches of tangent jobs before the solve (PLAN.md P9):
    # "auto" = on for the abaqus backend, off otherwise | "on" | "off"
    prefetch: str = "auto"


@dataclass(frozen=True)
class BackendCfg:
    kind: str = "analytic"
    # replay only: fingerprint (or its first 16 characters) of the recorded model, needed
    # when the cache directory holds more than one model
    replay_fingerprint: str | None = None
    delay_s: float = 0.0  # test aid: sleep this long per tangent job (ThrottledBackend)


@dataclass(frozen=True)
class AnalyticCfg:
    model: str = "two_dof"  # two_dof | random_sparse | spring_chain | duffing
    n: int = 40  # number of DOFs (random_sparse, spring_chain)
    seed: int = 0  # random_sparse
    density: float = 0.1  # random_sparse: generalised springs per DOF pair
    nl_scale: float = 1.0  # multiplies all quadratic and cubic coefficients


@dataclass(frozen=True)
class AbaqusCfg:
    runner: str = "local"  # local | spool | ssh
    # executable plus leading arguments; a string or a list in TOML (use a list for arguments)
    command: tuple[str, ...] = ("abaqus",)
    max_parallel: int = 1  # concurrent Abaqus jobs (limited by licence tokens)
    spool_dir: Path | None = None  # shared folder of the spool runner
    work_dir: Path | None = None  # job directories; default: <config dir>/../work/<name>
    user_lib_dir: str | None = None  # host path: directory of the precompiled stiffness UEL
    mass_lib_dir: str | None = None  # host path: directory of the precompiled mass UEL
    user_source: str | None = None  # host path: Fortran source passed as user= (no library)
    mass_source: str | None = None  # host path: Fortran source of the mass UEL
    mass_from: str = "uel"  # "uel": Abaqus job with the mass UEL | "processed": model.mass_mtx
    timeout_s: float = 900.0  # per attempt
    retries: int = 1  # extra attempts after a failed job
    keep_job_dirs: str = "failed"  # failed | all | none
    poll_s: float = 0.5  # spool runner: polling interval
    heartbeat_stale_s: float = 120.0  # spool runner: worker considered down after this


@dataclass(frozen=True)
class RomCfg:
    masters: tuple[int, ...] = (1,)  # 1-based mode numbers
    order: int = 3
    forcing_order: int = 0
    style: str = "complex_normal_form"
    resonance_tol: float = 0.05  # absolute detuning threshold; 0 = MORFE's default for the style
    n_modes: int = 0  # modes computed for the spectrum; 0 = automatic (max(masters) + 8)


@dataclass(frozen=True)
class ForcingCfg:
    enabled: bool = False
    amplitude: float = 0.0
    mode: int = 1  # forcing shape: amplitude * M * phi_mode


@dataclass(frozen=True)
class OutputCfg:
    dir: Path | None = None  # default: <config dir>/../runs/<name>
    monitor_node: int | None = None
    monitor_dof: int | None = None


@dataclass(frozen=True)
class CacheCfg:
    dir: Path | None = None  # None -> in-memory only
    lru_items: int = 64


@dataclass(frozen=True)
class Config:
    name: str
    path: Path | None
    step: StepCfg
    model: ModelCfg = field(default_factory=ModelCfg)
    damping: DampingCfg = field(default_factory=DampingCfg)
    engine: EngineCfg = field(default_factory=EngineCfg)
    backend: BackendCfg = field(default_factory=BackendCfg)
    analytic: AnalyticCfg = field(default_factory=AnalyticCfg)
    abaqus: AbaqusCfg = field(default_factory=AbaqusCfg)
    rom: RomCfg = field(default_factory=RomCfg)
    forcing: ForcingCfg = field(default_factory=ForcingCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    cache: CacheCfg = field(default_factory=CacheCfg)

    @property
    def use_prefetch(self) -> bool:
        """Whether the driver should run the per-order job batches (``[engine] prefetch``)."""
        if self.engine.prefetch == "auto":
            return self.backend.kind == "abaqus"
        return self.engine.prefetch == "on"


_SECTIONS: dict[str, type] = {
    "model": ModelCfg,
    "damping": DampingCfg,
    "step": StepCfg,
    "engine": EngineCfg,
    "backend": BackendCfg,
    "analytic": AnalyticCfg,
    "abaqus": AbaqusCfg,
    "rom": RomCfg,
    "forcing": ForcingCfg,
    "output": OutputCfg,
    "cache": CacheCfg,
}

# local paths, resolved relative to the config directory
_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "model": ("inp", "stiffness_mtx", "mass_mtx"),
    "abaqus": ("spool_dir", "work_dir"),
    "output": ("dir",),
    "cache": ("dir",),
}
# local input files that must exist when given
_MUST_EXIST: dict[str, tuple[str, ...]] = {
    "model": ("inp", "stiffness_mtx", "mass_mtx"),
}
_TUPLE_FIELDS: dict[str, tuple[str, ...]] = {"rom": ("masters",)}
# a string or a list of strings in TOML, stored as a tuple of strings
_ARGV_FIELDS: dict[str, tuple[str, ...]] = {"abaqus": ("command",)}
KEEP_KINDS = ("failed", "all", "none")
PREFETCH_KINDS = ("auto", "on", "off")
MASS_FROM = ("uel", "processed")


# --------------------------------------------------------------------------- loading


def load_config(path: str | Path) -> Config:
    """Load and validate a TOML run configuration."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    # utf-8-sig: accept the byte-order mark that Windows editors (and PowerShell 5.1) write
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"config file is not UTF-8: {path} ({exc})") from exc
    return config_from_dict(data, base_dir=path.parent, path=path)


def config_from_dict(
    data: dict[str, Any], base_dir: str | Path = ".", path: Path | None = None
) -> Config:
    """Build a validated :class:`Config` from a plain dict (as parsed from TOML)."""
    base_dir = Path(base_dir).resolve()
    allowed_top = set(_SECTIONS) | {"name"}
    unknown = set(data) - allowed_top
    if unknown:
        raise ConfigError(f"unknown top-level keys: {sorted(unknown)}")
    name = data.get("name") or (path.stem if path is not None else "run")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("'name' must be a non-empty string")
    if "step" not in data:
        raise ConfigError("missing required section [step] (needs d_ref)")

    sections = {key: _build_section(key, data.get(key), base_dir) for key in _SECTIONS}
    if sections["output"].dir is None:
        out = (base_dir.parent / "runs" / name).resolve()
        sections["output"] = OutputCfg(
            dir=out,
            monitor_node=sections["output"].monitor_node,
            monitor_dof=sections["output"].monitor_dof,
        )
    if sections["abaqus"].work_dir is None:
        sections["abaqus"] = replace(sections["abaqus"],
                                     work_dir=(base_dir.parent / "work" / name).resolve())
    cfg = Config(name=name, path=path, **sections)
    validate(cfg)
    return cfg


def _build_section(key: str, raw: Any, base_dir: Path):
    cls = _SECTIONS[key]
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"[{key}] must be a table")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"[{key}] unknown keys: {sorted(unknown)} (allowed: {sorted(known)})")
    kwargs: dict[str, Any] = {}
    for k, v in raw.items():
        if k in _PATH_FIELDS.get(key, ()):
            if not isinstance(v, str):
                raise ConfigError(f"[{key}] {k} must be a path string")
            p = Path(v)
            v = (p if p.is_absolute() else base_dir / p).resolve()
        elif k in _ARGV_FIELDS.get(key, ()):
            if isinstance(v, str):
                v = (v,)
            if (not isinstance(v, (list, tuple)) or not v
                    or not all(isinstance(x, str) and x for x in v)):
                raise ConfigError(f"[{key}] {k} must be a non-empty string or list of strings")
            v = tuple(v)
        elif k in _TUPLE_FIELDS.get(key, ()):
            if not isinstance(v, (list, tuple)):
                raise ConfigError(f"[{key}] {k} must be a list")
            v = tuple(v)
        kwargs[k] = v
    try:
        return cls(**kwargs)
    except TypeError as exc:  # missing required field
        raise ConfigError(f"[{key}] {exc}") from None


# --------------------------------------------------------------------------- validation


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate(cfg: Config) -> None:
    """Raise :class:`ConfigError` if the configuration is invalid."""
    s = cfg.step
    if not _is_num(s.d_ref) or s.d_ref <= 0:
        raise ConfigError(f"[step] d_ref must be > 0, got {s.d_ref!r}")
    if not _is_num(s.eps) or not (0 < s.eps < 0.25):
        raise ConfigError(f"[step] eps must be in (0, 0.25), got {s.eps!r}")
    for nm in ("zero_tol", "match_tol"):
        v = getattr(s, nm)
        if not _is_num(v) or not (0 < v < 1e-3):
            raise ConfigError(f"[step] {nm} must be in (0, 1e-3), got {v!r}")
    if not _is_int(s.seed):
        raise ConfigError("[step] seed must be an integer")

    if cfg.engine.mode not in ENGINE_MODES:
        raise ConfigError(f"[engine] mode must be one of {ENGINE_MODES}")
    if not isinstance(cfg.engine.debug, bool):
        raise ConfigError("[engine] debug must be true/false")
    if cfg.engine.prefetch not in PREFETCH_KINDS:
        raise ConfigError(f"[engine] prefetch must be one of {PREFETCH_KINDS}")
    if not _is_num(cfg.backend.delay_s) or cfg.backend.delay_s < 0:
        raise ConfigError("[backend] delay_s must be >= 0")

    kind = cfg.backend.kind
    if kind not in BACKEND_KINDS:
        raise ConfigError(f"[backend] kind must be one of {BACKEND_KINDS}, got {kind!r}")
    if kind in ("pysvk", "abaqus") and cfg.model.inp is None:
        raise ConfigError(f"[model] inp is required for backend kind {kind!r}")
    for nm in _MUST_EXIST["model"]:
        p = getattr(cfg.model, nm)
        if p is not None and not p.is_file():
            raise ConfigError(f"[model] {nm} does not exist: {p}")
    if not cfg.model.clamped_nset:
        raise ConfigError("[model] clamped_nset must be non-empty")

    d = cfg.damping
    if not (_is_num(d.alpha) and _is_num(d.beta)) or d.alpha < 0 or d.beta < 0:
        raise ConfigError("[damping] alpha and beta must be finite and >= 0")

    a = cfg.analytic
    if a.model not in ANALYTIC_MODELS:
        raise ConfigError(f"[analytic] model must be one of {ANALYTIC_MODELS}")
    if not _is_int(a.n) or a.n < 1:
        raise ConfigError("[analytic] n must be a positive integer")
    if not _is_num(a.density) or a.density <= 0:
        raise ConfigError("[analytic] density must be > 0")
    if not _is_num(a.nl_scale):
        raise ConfigError("[analytic] nl_scale must be a finite number")

    q = cfg.abaqus
    if q.runner not in RUNNER_KINDS:
        raise ConfigError(f"[abaqus] runner must be one of {RUNNER_KINDS}")
    if not _is_int(q.max_parallel) or q.max_parallel < 1:
        raise ConfigError("[abaqus] max_parallel must be >= 1")
    if not _is_num(q.timeout_s) or q.timeout_s <= 0:
        raise ConfigError("[abaqus] timeout_s must be > 0")
    if not _is_int(q.retries) or q.retries < 0:
        raise ConfigError("[abaqus] retries must be >= 0")
    if kind == "abaqus" and q.runner == "spool" and q.spool_dir is None:
        raise ConfigError("[abaqus] spool_dir is required for runner = 'spool'")
    if q.keep_job_dirs not in KEEP_KINDS:
        raise ConfigError(f"[abaqus] keep_job_dirs must be one of {KEEP_KINDS}")
    if q.mass_from not in MASS_FROM:
        raise ConfigError(f"[abaqus] mass_from must be one of {MASS_FROM}")
    if kind == "abaqus" and q.mass_from == "processed" and cfg.model.mass_mtx is None:
        raise ConfigError("[abaqus] mass_from = 'processed' needs [model] mass_mtx")
    for nm in ("poll_s", "heartbeat_stale_s"):
        v = getattr(q, nm)
        if not _is_num(v) or v <= 0:
            raise ConfigError(f"[abaqus] {nm} must be > 0")
    if kind == "replay" and cfg.cache.dir is None:
        raise ConfigError("[cache] dir is required for backend kind 'replay'")

    r = cfg.rom
    if not r.masters or not all(_is_int(m) and m >= 1 for m in r.masters):
        raise ConfigError("[rom] masters must be a non-empty list of positive integers")
    if len(set(r.masters)) != len(r.masters):
        raise ConfigError("[rom] masters must be unique")
    if not _is_int(r.order) or r.order < 1:
        raise ConfigError("[rom] order must be >= 1")
    if not _is_int(r.forcing_order) or r.forcing_order < 0:
        raise ConfigError("[rom] forcing_order must be >= 0")
    if not isinstance(r.style, str) or not r.style:
        raise ConfigError("[rom] style must be a non-empty string")
    if not _is_num(r.resonance_tol) or r.resonance_tol < 0:
        raise ConfigError("[rom] resonance_tol must be >= 0")
    if not _is_int(r.n_modes) or r.n_modes < 0:
        raise ConfigError("[rom] n_modes must be >= 0 (0 = automatic)")
    if r.n_modes and r.n_modes < max(r.masters):
        raise ConfigError("[rom] n_modes must be >= the largest master mode number")

    f = cfg.forcing
    if not isinstance(f.enabled, bool):
        raise ConfigError("[forcing] enabled must be true/false")
    if not _is_num(f.amplitude):
        raise ConfigError("[forcing] amplitude must be a finite number")
    if not _is_int(f.mode) or f.mode < 1:
        raise ConfigError("[forcing] mode must be a positive integer")
    if f.enabled and r.forcing_order < 1:
        raise ConfigError("[forcing] enabled requires [rom] forcing_order >= 1")

    o = cfg.output
    for nm in ("monitor_node", "monitor_dof"):
        v = getattr(o, nm)
        if v is not None and (not _is_int(v) or v < 1):
            raise ConfigError(f"[output] {nm} must be a positive integer")
    if o.monitor_dof is not None and o.monitor_dof > 3:
        raise ConfigError("[output] monitor_dof must be 1, 2 or 3")

    if not _is_int(cfg.cache.lru_items) or cfg.cache.lru_items < 1:
        raise ConfigError("[cache] lru_items must be >= 1")
