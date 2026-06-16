"""Smoke test: package imports."""
from __future__ import annotations


def test_import_package() -> None:
    import rppg_sa

    assert rppg_sa.__version__
