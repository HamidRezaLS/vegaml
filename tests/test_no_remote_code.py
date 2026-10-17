"""A checkpoint is data from a third party. It must never be able to carry code that runs here.

`vegaml.load("someone/their-model")` downloads a repository the caller may not control. If the
runtime imported a `.py` out of it, loading a model would be arbitrary code execution by design.
The control is the download allow-list, and these tests are what keep it honest.
"""
from __future__ import annotations

import pathlib
import re
import sys
import types

import vegaml
from vegaml import hub

PKG = pathlib.Path(vegaml.__file__).parent
EXECUTABLE = (".py", ".pyc", ".so", ".dll", ".dylib", ".sh", ".bash", ".pyd", ".pth")


def test_every_variant_scopes_the_allow_list_to_its_own_folder():
    """A size name must not widen what gets downloaded, only move where it is read from."""
    for name, sub in hub.VARIANTS.items():
        pats = hub.patterns_for(sub)
        assert len(pats) == len(hub.ALLOW_PATTERNS), name
        for pat in pats:
            assert pat.endswith((".json", ".safetensors", ".npz")), (name, pat)
            # The download counter watches one file at the repository root, so that one is exempt
            # from the subfolder rule. Nothing else may escape its own folder.
            if sub and pat != hub.COUNTER_FILE:
                assert pat.startswith(sub + "/"), (name, pat)


def test_allow_list_has_no_executable_extension():
    for pattern in hub.ALLOW_PATTERNS:
        assert not pattern.endswith(EXECUTABLE), pattern
        assert "*" not in pathlib.Path(pattern).suffix, "a wildcard suffix could match a .py: " + pattern


def test_allow_list_is_data_only():
    # Every entry resolves to json, safetensors, or npz. Widening this list is a security
    # decision and should fail here until the test is updated deliberately. The one npz is a
    # vega-U checkpoint's feature statistics; it is arrays only, and the test below pins that
    # every np.load in the package spells out allow_pickle=False.
    for pattern in hub.ALLOW_PATTERNS:
        assert pattern.endswith((".json", ".safetensors", ".npz")), pattern


def test_every_np_load_refuses_pickles():
    # npz on the allow-list is only data because numpy is told not to unpickle. Spelling it out
    # at every call site keeps that true even if numpy's default ever changes.
    for path in PKG.rglob("*.py"):
        for ln in path.read_text(encoding="utf-8").splitlines():
            if "np.load(" in ln or "numpy.load(" in ln:
                assert "allow_pickle=False" in ln, "%s: %s" % (path.name, ln.strip())


def test_download_is_restricted_to_the_allow_list(monkeypatch):
    """What `fetch` actually asks the Hub for.

    A stub stands in for huggingface_hub so this gate needs nothing installed beyond the package
    itself: a security check that only runs when the full ML stack resolves is a check that stops
    running the first time an unrelated dependency breaks.
    """
    seen = {}

    def fake_snapshot_download(repo_id, **kw):
        seen.update(kw, repo_id=repo_id)
        return "/tmp/not-a-real-checkpoint"

    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    hub.fetch("someone/their-model")
    assert seen["allow_patterns"] == hub.ALLOW_PATTERNS, "fetch must constrain what it downloads"
    assert seen["repo_id"] == "someone/their-model"


def test_the_counter_file_is_requested_for_every_size():
    """The Hub counts a download only when the file it watches is actually fetched.

    The 4B lives in a subfolder, so its patterns are prefixed. The counter file must stay at the
    repository root in both cases or loading the 4B would register nothing.
    """
    for subfolder in ("", "4b"):
        pats = hub.patterns_for(subfolder)
        assert hub.COUNTER_FILE in pats, subfolder
        assert pats.count(hub.COUNTER_FILE) == 1, "asked for twice"
    assert "4b/" + hub.COUNTER_FILE not in hub.patterns_for("4b"), "must not be subfolder-scoped"


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
