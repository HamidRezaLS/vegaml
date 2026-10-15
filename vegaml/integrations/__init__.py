"""Integrations: vegaml decisions as nodes inside agent frameworks.

Importing this package costs nothing and requires none of the frameworks; each module imports
its framework lazily and degrades to a plain callable where the framework is absent.
"""
from __future__ import annotations

__all__ = ["langchain"]
