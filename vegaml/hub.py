"""Fetch a Vega checkpoint from the Hugging Face Hub: weights and config only.

Deliberately narrow: `allow_patterns` lists data files and no `*.py`. The inference code lives in
this package and is reviewed with it. A checkpoint repository can hold whatever its author put
there, so nothing downloaded here is ever imported or executed, and the engine is read through
safetensors only, never through Python's unpickler.
"""
from __future__ import annotations

import os
from typing import Optional, Union

#: The checkpoint repository: the universal (vega-U) model, one engine behind per-reader
#: ports. The earlier per-size checkpoints stay loadable by their full repository id
#: ("nandakishorm/vega-08b-public-intents").
DEFAULT_REPO = "nandakishorm/vega-U"

#: Size name to the folder inside that repository. "" means the repository root, where the
#: default reader lives so that ``load()`` needs no name at all.
VARIANTS = {
    "0.8b": "",
    "08b": "",
    "4b": "Qwen3.5-4B",
    "gemma-1b": "gemma-3-1b-it",
}

#: Reader names as a universal checkpoint's config stores them (`backbone_name`, the bare
#: model name) mapped to the Hub id those weights actually live under. A checkpoint dir names
#: its reader without the owning namespace, which is right for a dir and wrong for a download.
READER_HUB_IDS = {
    "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Qwen3.8-27B": "Qwen/Qwen3.8-27B",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
}


def backbone_hub_id(name: Optional[str]) -> Optional[str]:
    """A loadable location for a checkpoint's `backbone_name`: a bare reader name becomes its
    Hub id when it is a known reader; paths, Hub ids and unknown names pass through unchanged."""
    if name and "/" not in name and not os.path.isdir(name):
        return READER_HUB_IDS.get(name, name)
    return name


def looks_like_hub_id(name: Optional[str]) -> bool:
    """Whether `name` is shaped like a repository id (``owner/model``) rather than a filesystem
    path. A training machine stores its backbone as an absolute path ("/kaggle/input/..."), which
    must never be mistaken for a repository; a repository id has exactly one slash and no path
    anatomy around it."""
    import re

    return bool(name) and name.count("/") == 1 and not name.startswith((".", "/", "~")) \
        and re.fullmatch(r"[\w.\-]+/[\w.\-]+", name) is not None


#: Where a token is read from when the caller passes none, in that order. `HF_TOKEN` is the name
#: the Hugging Face CLI and the Hub libraries use; the other two are older spellings still in use.
TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def default_token() -> Optional[str]:
    """The Hub token found in the environment, or None.

    Nothing here is mandatory. The published checkpoints are public and load with no token at all.
    A token is what makes a *private* or gated repository load, and what lets a fine-tune push its
    adapter back to the Hub under your own account, so it is picked up automatically when it is
    there rather than having to be threaded through every call.
    """
    for var in TOKEN_ENV:
        value = (os.environ.get(var) or "").strip()
        if value:
            return value
    return None


#: The Hub counts a repository download when this file is requested, and counts nothing when a
#: repository has no file it recognises. It carries no settings: the runtime reads vega_config.json.
COUNTER_FILE = "config.json"

#: What the runtime reads. No executable file is on this list.
ALLOW_PATTERNS = [
    COUNTER_FILE,
    "vega_config.json",
    "engine.safetensors",
    "adapters/*.safetensors",
    "adapters/*.json",
    # a vega-U (universal) checkpoint adds the backbone's port and its extraction statistics
    "port.safetensors",
    "stats.npz",
]


def patterns_for(subfolder: str) -> list[str]:
    """The allow-list, scoped to one model inside the repository.

    The counter file is always asked for at the repository root, whichever size is being loaded,
    because that is the path the Hub's download counter watches. It is a few hundred bytes and
    nothing reads it.
    """
    if not subfolder:
        return list(ALLOW_PATTERNS)
    return [COUNTER_FILE] + [f"{subfolder}/{p}" for p in ALLOW_PATTERNS if p != COUNTER_FILE]


def fetch(repo_id: Optional[str] = None, revision: Optional[str] = None,
          token: Union[str, bool, None] = None, cache_dir: Optional[str] = None,
          reader: Optional[str] = None) -> str:
    """Return a local directory holding a checkpoint.

    `repo_id` may be a size name (``"0.8b"``, the default, or ``"4b"``), a Hub repository id, or a
    local path. A local path is passed through unchanged, so this is safe to call on either.

    `reader` is the short name of a reader inside a universal (vega-U) repository, such as
    ``"gemma-3-1b-it"``. Its subfolder is downloaded alongside the root with the same data-only
    allow-list; a repository without that subfolder simply matches nothing extra.

    `token` defaults to whatever :func:`default_token` finds in the environment, which is how a
    private or gated repository loads without the caller passing anything. It stays optional: with
    no token set, a public repository still loads. Pass ``token=False`` to ignore the environment
    and fetch anonymously, or a string to override it.
    """
    name = (repo_id or "0.8b").strip()
    if os.path.isdir(name):
        return name

    key = name.lower()
    if key in VARIANTS:
        repo, subfolder = DEFAULT_REPO, VARIANTS[key]
    else:
        repo, subfolder = name, ""

    patterns = patterns_for(subfolder)
    if reader and not subfolder:
        patterns = patterns + [f"{reader}/{p}" for p in ALLOW_PATTERNS if p != COUNTER_FILE]

    explicit = token is not None and token is not False
    if token is None:
        token = default_token()

    from huggingface_hub import snapshot_download

    try:
        local = snapshot_download(repo, revision=revision, token=token, cache_dir=cache_dir,
                                  allow_patterns=patterns)
    except Exception as exc:
        # Matched by name rather than by class: importing huggingface_hub.errors here would make
        # the whole library a hard import just to describe a failure.
        if _is_auth_error(exc):
            raise _auth_error(repo, name, token, explicit) from exc
        raise
    return os.path.join(local, subfolder) if subfolder else local


def reader_dir(path: str, backbone: str) -> str:
    """Inside a universal (vega-U) checkpoint, the dir carrying the port for `backbone`.

    `backbone` is whatever the caller would pass to load the reader itself: a Hub id like
    ``"google/gemma-3-1b-it"``, or a local path to those weights; its last path component is the
    reader's name. The repo root is one reader, every other one is a subfolder of the same name.
    A classic (non-universal) checkpoint comes back unchanged, so ``backbone=`` keeps its old
    meaning there: an alternate location of the SAME backbone.
    """
    import json

    cfg_path = os.path.join(path, "vega_config.json")
    if not os.path.isfile(cfg_path):
        return path
    with open(cfg_path) as f:
        cfg = json.load(f)
    if not cfg.get("port"):
        return path
    name = os.path.basename(backbone.rstrip("/"))
    if cfg.get("backbone_name") == name or cfg.get("backbone_name") == backbone:
        return path
    sub = os.path.join(path, name)
    if os.path.isfile(os.path.join(sub, "vega_config.json")):
        return sub
    have = [cfg.get("backbone_name", "the root reader")] + sorted(
        os.path.basename(d) for d in os.listdir(path)
        if os.path.isfile(os.path.join(path, d, "vega_config.json")))
    raise ValueError(
        "this universal checkpoint has no port for %r. Readers it ships: %s. A new reader is "
        "attached by fitting its port, never by retraining the engine: vegaml.attach(%r, "
        "texts=..., out_dir=...) does it locally from your own text."
        % (name, ", ".join(have), backbone))


def _is_auth_error(exc: BaseException) -> bool:
    """Whether the Hub refused for want of credentials, rather than failing some other way."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & {"RepositoryNotFoundError", "GatedRepoError"}:
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403)


def _auth_error(repo: str, asked_for: str, token, explicit: bool) -> PermissionError:
    """The Hub says not-found-or-not-authorised. Say which of the two it probably is.

    The Hub deliberately returns the same answer for "no such repository" and "exists but you may
    not see it", so the raw error sends people hunting for a typo when the real answer is a missing
    token. No token value is ever put in the message.
    """
    if explicit:
        where = "a token was passed to this call"
    elif token:
        found = next((v for v in TOKEN_ENV if (os.environ.get(v) or "").strip()), "the environment")
        where = "a token was read from %s" % found
    elif token is False:
        where = "token=False was passed, so no token was sent"
    else:
        where = "no token was found: none of %s is set" % ", ".join(TOKEN_ENV)

    hint = ("Set HF_TOKEN to a token with read access to it, or pass token=... explicitly."
            if not token else
            "Check that this token has read access to that repository and has not been revoked.")
    asked = "" if asked_for == repo else " (asked for %r)" % asked_for
    return PermissionError(
        "cannot read the checkpoint repository %r%s. The Hub returns the same answer for a "
        "repository that does not exist and one you are not allowed to see, so either the id is "
        "wrong or it is private or gated. Here, %s. %s" % (repo, asked, where, hint))
