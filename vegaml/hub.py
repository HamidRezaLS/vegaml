"""Fetch a Vega checkpoint from the Hugging Face Hub: weights and config only.

Deliberately narrow: `allow_patterns` lists data files and no `*.py`. The inference code lives in
this package and is reviewed with it. A checkpoint repository can hold whatever its author put
there, so nothing downloaded here is ever imported or executed, and the engine is read through
safetensors only, never through Python's unpickler.
"""
from __future__ import annotations

import os
from typing import Optional, Union

#: The checkpoint repository. It holds more than one size.
DEFAULT_REPO = "nandakishorm/vega-08b-public-intents"

#: Size name to the folder inside that repository. "" means the repository root.
#: 0.8B sits at the root because it was there first and existing installs point at it.
VARIANTS = {
    "0.8b": "",
    "08b": "",
    "4b": "4b",
}

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
          token: Union[str, bool, None] = None, cache_dir: Optional[str] = None) -> str:
    """Return a local directory holding a checkpoint.

    `repo_id` may be a size name (``"0.8b"``, the default, or ``"4b"``), a Hub repository id, or a
    local path. A local path is passed through unchanged, so this is safe to call on either.

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

    explicit = token is not None and token is not False
    if token is None:
        token = default_token()

    from huggingface_hub import snapshot_download

    try:
        local = snapshot_download(repo, revision=revision, token=token, cache_dir=cache_dir,
                                  allow_patterns=patterns_for(subfolder))
    except Exception as exc:
        # Matched by name rather than by class: importing huggingface_hub.errors here would make
        # the whole library a hard import just to describe a failure.
        if _is_auth_error(exc):
            raise _auth_error(repo, name, token, explicit) from exc
        raise
    return os.path.join(local, subfolder) if subfolder else local


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
