"""Session configuration files: portable JSON snapshots of GUI parameters.

A configuration stores the microscope parameters, the fine-structure
definition, the constraint rows and the processing parameters (including
the component colors). It deliberately excludes session state that cannot
be restored meaningfully (selected folders, results folder, despiking and
masking, which depend on the GUI mode).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

CONFIG_KIND = "eelsnmf-gui-session-config"
CONFIG_VERSION = 1


@dataclass
class SessionConfig:
    """Parameters worth persisting across GUI sessions."""

    e0: float = 300.0
    alpha: float = 15.4
    beta: float = 18.46
    # Fine-structure rows: {"name", "min", "max", "checked"}.
    edges: List[Dict[str, Any]] = field(default_factory=list)
    # Constraint rows: [edge, component, state].
    fixed_weights: List[List[Any]] = field(default_factory=list)
    n_components: int = 2
    max_iters: int = 2000
    tol: float = 1e-8
    error_skip_step: int = 100
    decomposition_method: str = "default_decomposition"
    regularization_tv: float = 0.1
    regularization_sr: float = 0.1
    n_power_laws: int = 50
    # Component colors as [r, g, b] floats in [0, 1].
    colors: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": CONFIG_KIND,
            "version": CONFIG_VERSION,
            "config": asdict(self),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionConfig:
        if not isinstance(data, dict) or data.get("kind") != CONFIG_KIND:
            raise ValueError("Not an eelsnmf-gui session configuration file")
        if data.get("version") != CONFIG_VERSION:
            raise ValueError(
                f"Unsupported configuration version: {data.get('version')!r}"
            )
        payload = data.get("config")
        if not isinstance(payload, dict):
            raise ValueError("Configuration payload is missing")
        return cls(
            e0=float(payload.get("e0", 300.0)),
            alpha=float(payload.get("alpha", 15.4)),
            beta=float(payload.get("beta", 18.46)),
            edges=list(payload.get("edges", [])),
            fixed_weights=list(payload.get("fixed_weights", [])),
            n_components=int(payload.get("n_components", 2)),
            max_iters=int(payload.get("max_iters", 2000)),
            tol=float(payload.get("tol", 1e-8)),
            error_skip_step=int(payload.get("error_skip_step", 100)),
            decomposition_method=str(
                payload.get("decomposition_method", "default_decomposition")
            ),
            regularization_tv=float(payload.get("regularization_tv", 0.1)),
            regularization_sr=float(payload.get("regularization_sr", 0.1)),
            n_power_laws=int(payload.get("n_power_laws", 50)),
            colors=list(payload.get("colors", [])),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> SessionConfig:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)
