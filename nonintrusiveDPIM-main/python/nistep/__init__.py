"""nistep: non-intrusive EED-StEP closure engine (the FE side of the MORFE.jl interface).

Public entry points
-------------------
* :func:`build_engine` takes a TOML config path and returns an ``EEDEngine``. MORFEStEP.jl uses it.
* :func:`build_backend` takes a ``Config`` and returns a ``TangentBackend``.
* :func:`load_config` loads a TOML file into a validated ``Config``.

The math conventions (signs, the factors 2/3/6, complex handling) are in ``CONVENTIONS.md``.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config, ConfigError, load_config

__version__ = "0.1.0"

__all__ = ["__version__", "build_engine", "build_backend", "load_config", "Config", "ConfigError"]


def build_backend(cfg: Config):
    """Instantiate the tangent-stiffness backend described by ``cfg.backend.kind``."""
    backend = _build_backend(cfg)
    if cfg.backend.delay_s > 0:
        from .backends.wrappers import ThrottledBackend

        backend = ThrottledBackend(backend, delay_s=cfg.backend.delay_s)
    return backend


def _build_backend(cfg: Config):
    kind = cfg.backend.kind
    if kind == "analytic":
        from .backends.analytic import from_config

        return from_config(cfg.analytic)
    if kind == "pysvk":
        from .backends.pysvk import PySVKBackend

        return PySVKBackend.from_config(cfg)
    if kind == "abaqus":
        from .backends.abaqus import AbaqusBackend

        return AbaqusBackend.from_config(cfg)
    if kind == "replay":
        from .backends.replay import ReplayBackend

        return ReplayBackend(cfg.cache.dir, cfg.backend.replay_fingerprint)
    raise NotImplementedError(f"backend kind {kind!r} is not implemented")


def build_engine(config_path: str | Path):
    """Load a config and return a ready ``EEDEngine`` (backend + store + step settings)."""
    from .engine import EEDEngine
    from .store import KtgStore

    cfg = load_config(config_path)
    backend = build_backend(cfg)
    store = KtgStore(backend.fingerprint(), cache_dir=cfg.cache.dir, lru_items=cfg.cache.lru_items)
    engine = EEDEngine(backend, cfg.step, store, mode=cfg.engine.mode, debug=cfg.engine.debug)
    engine.config = cfg
    return engine
