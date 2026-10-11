"""The image path's own imports, read off the shipped sources: no checkpoint, no network.

`VegaVision._decide` reached for the thermal helpers through a bare module name that no part of this
package ships, so every installed copy raised ModuleNotFoundError on the first `decide_image` -- the
documented entry point of the `vision` extra. These tests pin the two halves of that bug: where the
import points, and that the names are there.
"""
from __future__ import annotations

import ast
import pathlib

import vegaml

PKG = pathlib.Path(vegaml.__file__).parent

# What _decide needs at call time. They live in _common, which is vendored from the training tree.
HELPERS = ("QTYPES", "route_adapters", "softmax_np", "thermal_features", "thermal_tau")


def test_vision_decide_reaches_for_the_package_not_for_the_training_tree():
    """A bare name resolves against whatever is installed, so it cannot be how the package finds itself."""
    tree = ast.parse((PKG / "_vision.py").read_text(encoding="utf-8"))
    decide = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_decide")
    for node in ast.walk(decide):
        if isinstance(node, ast.ImportFrom):
            assert node.level >= 1, "`from %s import ...` escapes the package" % node.module


def test_the_handlers_vision_needs_are_in_the_package():
    from vegaml import _common

    for name in HELPERS:
        assert hasattr(_common, name), "vegaml._common is missing %s" % name
