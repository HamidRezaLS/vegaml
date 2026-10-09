"""Metadata that silently drifts: the version in two places, the Python versions in three."""
from __future__ import annotations

import pathlib
import re
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:                                      # tomllib is stdlib only from 3.11
    import tomli as tomllib

import vegaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_version_matches_the_package():
    assert PYPROJECT["project"]["version"] == vegaml.__version__


def test_classifiers_match_requires_python():
    declared = {c.rsplit(" :: ", 1)[1] for c in PYPROJECT["project"]["classifiers"]
                if c.startswith("Programming Language :: Python :: 3.")}
    floor = PYPROJECT["project"]["requires-python"]
    assert floor == ">=3.10"
    assert declared == {"3.10", "3.11", "3.12", "3.13"}


def test_ci_tests_every_classified_python():
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    matrix = re.search(r'python: \[([^\]]*)\]', ci).group(1)
    tested = set(re.findall(r'"([\d.]+)"', matrix))
    declared = {c.rsplit(" :: ", 1)[1] for c in PYPROJECT["project"]["classifiers"]
                if c.startswith("Programming Language :: Python :: 3.")}
    assert tested == declared, "CI matrix and classifiers disagree"


def test_licence_is_apache_2_and_shipped():
    assert PYPROJECT["project"]["license"] == "Apache-2.0"
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text and "Version 2.0" in text


def test_readme_quickstart_uses_the_real_entry_point():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "vegaml.load(" in readme
    assert "pip install vegaml" in readme
