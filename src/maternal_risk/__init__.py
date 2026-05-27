"""Maternal health risk classification package.

Exposes the package version and the public service entry points. Submodules are
imported lazily by callers to keep import-time side effects (e.g. loading a
model artifact) out of simple ``import maternal_risk`` statements.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
