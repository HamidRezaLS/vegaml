"""Fetch a Vega checkpoint from the Hugging Face Hub: weights and config only.

Deliberately narrow: `allow_patterns` lists data files and no `*.py`. The inference code lives in this
package and is reviewed with it. A checkpoint repository can hold whatever its author put there, so
nothing downloaded here is ever imported or executed, and the engine is read through safetensors
only - never through Python's unpickler.
"""
from __future__ import annotations

import os
from typing import Optional

#: Everything the runtime reads. No executable file is on this list.
ALLOW_PATTERNS = [
    "vega_config.json",
    "engine.safetensors",
    "adapters/*.safetensors",
    "adapters/*.json",
]


def fetch(repo_id: str, revision: Optional[str] = None, token: Optional[str] = None,
          cache_dir: Optional[str] = None) -> str:
    """Download a checkpoint and return the local directory holding it.

    A local path is passed through unchanged, so `fetch` is safe to call on either.
    """
    if os.path.isdir(repo_id):
        return repo_id
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, revision=revision, token=token, cache_dir=cache_dir,
                             allow_patterns=ALLOW_PATTERNS)
