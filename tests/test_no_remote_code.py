"""A checkpoint is data from a third party. It must never be able to carry code that runs here.

`vegaml.load("someone/their-model")` downloads a repository the caller may not control. If the
runtime imported a `.py` out of it, loading a model would be arbitrary code execution by design.
The control is the download allow-list, and these tests are what keep it honest.
"""
from __future__ import annotations

import pathlib
import re

import vegaml
from vegaml import hub

PKG = pathlib.Path(vegaml.__file__).parent
EXECUTABLE = (".py", ".pyc", ".so", ".dll", ".dylib", ".sh", ".bash", ".pyd", ".pth")


def test_allow_list_has_no_executable_extension():
    for pattern in hub.ALLOW_PATTERNS:
        assert not pattern.endswith(EXECUTABLE), pattern
        assert "*" not in pathlib.Path(pattern).suffix, "a wildcard suffix could match a .py: " + pattern


def test_allow_list_is_data_only():
    # Every entry resolves to json or safetensors. Widening this list is a security decision and
    # should fail here until the test is updated deliberately.
    for pattern in hub.ALLOW_PATTERNS:
        assert pattern.endswith((".json", ".safetensors")), pattern


def test_download_is_restricted_to_the_allow_list(monkeypatch):
    seen = {}

    def fake_snapshot_download(repo_id, **kw):
        seen.update(kw, repo_id=repo_id)
        return "/tmp/not-a-real-checkpoint"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    hub.fetch("someone/their-model")
    assert seen["allow_patterns"] == hub.ALLOW_PATTERNS, "fetch must constrain what it downloads"


def test_local_directory_passes_through(tmp_path):
    assert hub.fetch(str(tmp_path)) == str(tmp_path)


def test_package_never_adds_a_downloaded_path_to_sys_path():
    # The one line that would undo all of the above.
    for path in PKG.rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert "sys.path.insert" not in src, "%s inserts into sys.path" % path.name
        assert "sys.path.append" not in src, "%s appends to sys.path" % path.name


def test_no_unsafe_deserialization_in_the_package():
    bad = re.compile(r"\bpickle\b|torch\.load\(|joblib\.load\(|yaml\.load\(")
    for path in PKG.rglob("*.py"):
        hits = [ln for ln in path.read_text(encoding="utf-8").splitlines() if bad.search(ln)]
        assert not hits, "%s: %s" % (path.name, hits)


def test_no_dynamic_execution_in_the_package():
    bad = re.compile(r"os\.system\(|subprocess\.|shell=True|\beval\(|\bexec\(")
    for path in PKG.rglob("*.py"):
        hits = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                if bad.search(ln) and ".eval()" not in ln]
        assert not hits, "%s: %s" % (path.name, hits)
