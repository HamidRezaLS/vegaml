"""Fetch a Vega checkpoint from the Hugging Face Hub: weights and config only.

Deliberately narrow: `allow_patterns` lists data files and no `*.py`. The inference code lives in
this package and is reviewed with it. A checkpoint repository can hold whatever its author put
there, so nothing downloaded here is ever imported or executed, and the engine is read through
safetensors only, never through Python's unpickler.
"""
from __future__ import annotations

import os
from typing import Optional

#: The checkpoint repository. It holds more than one size.
DEFAULT_REPO = "nandakishorm/vega-08b-public-intents"

#: Size name to the folder inside that repository. "" means the repository root.
#: 0.8B sits at the root because it was there first and existing installs point at it.
VARIANTS = {
    "0.8b": "",
    "08b": "",
    "4b": "4b",
}

#: What the runtime reads. No executable file is on this list.
ALLOW_PATTERNS = [
    "vega_config.json",
    "engine.safetensors",
    "adapters/*.safetensors",
    "adapters/*.json",
]


def patterns_for(subfolder: str) -> list[str]:
    """The allow-list, scoped to one model inside the repository."""
    if not subfolder:
        return list(ALLOW_PATTERNS)
    return [f"{subfolder}/{p}" for p in ALLOW_PATTERNS]


def fetch(repo_id: Optional[str] = None, revision: Optional[str] = None,
          token: Optional[str] = None, cache_dir: Optional[str] = None) -> str:
    """Return a local directory holding a checkpoint.

    `repo_id` may be a size name (``"0.8b"``, the default, or ``"4b"``), a Hub repository id, or a
    local path. A local path is passed through unchanged, so this is safe to call on either.
    """
    name = (repo_id or "0.8b").strip()
    if os.path.isdir(name):
        return name

    key = name.lower()
    if key in VARIANTS:
        repo, subfolder = DEFAULT_REPO, VARIANTS[key]
    else:
        repo, subfolder = name, ""

    from huggingface_hub import snapshot_download

    local = snapshot_download(repo, revision=revision, token=token, cache_dir=cache_dir,
                              allow_patterns=patterns_for(subfolder))
    return os.path.join(local, subfolder) if subfolder else local
