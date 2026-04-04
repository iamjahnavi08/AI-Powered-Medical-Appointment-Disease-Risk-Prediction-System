from __future__ import annotations

"""
Backwards-compatible facade for shared backend utilities.

This module intentionally *proxies* attribute access into the split modules so mutable module-level
state (e.g., `DB_CONN`) continues to behave like it used to when everything lived in one file.
"""

from . import shared_core as _core
from . import shared_domain as _domain
from . import shared_models as _models


def __getattr__(name: str):  # noqa: ANN001
    for module in (_core, _models, _domain):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    names: set[str] = set()
    for module in (_core, _models, _domain):
        names.update(getattr(module, "__all__", []) or [])
        names.update([n for n in dir(module) if not n.startswith("_")])
    return sorted(names)


__all__ = __dir__()
