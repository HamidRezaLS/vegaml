"""The Kaggle notebook must stay in step with the script that was actually tested.

A notebook that embeds a copy of a script rots: the copy is edited, or the script is, and the thing
people run stops being the thing anyone verified. The build step reads the script at build time;
these tests fail if the committed notebook no longer matches it.
"""
from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "vega_finetune_2xT4_kaggle.ipynb"
SCRIPT = ROOT / "notebooks" / "vega_finetune.py"


def cells():
    return json.loads(NB.read_text(encoding="utf-8"))["cells"]


def source(cell):
    return "".join(cell["source"])


def test_notebook_is_valid_nbformat_4():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    for c in nb["cells"]:
        assert c["cell_type"] in ("code", "markdown")
        assert isinstance(c["source"], list)


def test_embedded_script_is_byte_identical_to_the_tested_file():
    writefile = [source(c) for c in cells()
                 if c["cell_type"] == "code" and source(c).startswith("%%writefile vega_finetune.py")]
    assert len(writefile) == 1, "expected exactly one %%writefile cell"
    embedded = writefile[0].split("\n", 1)[1]
    assert embedded == SCRIPT.read_text(encoding="utf-8"), \
        "notebook and vega_finetune.py have diverged -- rerun notebooks/build_finetune_nb.py"


def shell_cells():
    """Code cells that run something, excluding the one that merely writes the script to disk."""
    return [source(c) for c in cells()
            if c["cell_type"] == "code" and not source(c).startswith("%%writefile")]


def test_all_three_stages_are_invoked_in_order():
    code = "\n".join(shell_cells())
    extract = code.index("vega_finetune.py extract")
    train = code.index("vega_finetune.py train")
    ev = code.index("vega_finetune.py eval")
    assert extract < train < ev, "stages must run extract -> train -> eval"


def test_extract_uses_both_gpus_and_the_rest_do_not():
    code = shell_cells()
    extract = [c for c in code if "vega_finetune.py extract" in c][0]
    assert "torchrun --nproc_per_node=2" in extract
    for stage in ("train", "eval"):
        cell = [c for c in code if "vega_finetune.py " + stage in c][0]
        assert "torchrun" not in cell, "%s runs on cached features; a second process buys nothing" % stage


def test_a_hugging_face_token_is_optional_never_required():
    """A token unlocks private checkpoints and pushing. It must never be needed to run the thing."""
    text = NB.read_text(encoding="utf-8") + SCRIPT.read_text(encoding="utf-8")
    assert 'os.environ.get("HF_TOKEN")' in text, "the token should be read, not demanded"
    for needle in ('os.environ["HF_TOKEN"]\n', "huggingface-cli login", "login(token",
                   'os.environ["HF_TOKEN"])'):
        assert needle not in text, "a missing token must not be an error: " + needle
    assert re.search(r'HF_TOKEN = os\.environ\.get\("HF_TOKEN"\) or None', SCRIPT.read_text()), \
        "the script should fall back to None, so the public path needs nothing"


def test_no_credential_is_baked_into_either_file():
    text = NB.read_text(encoding="utf-8") + SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{16,}", text), "a literal Hub token is committed"


def test_the_size_is_a_variable_defaulting_to_the_small_one():
    script = SCRIPT.read_text(encoding="utf-8")
    assert '"0.8b": "Qwen/Qwen3.5-0.8B"' in script and '"4b": "Qwen/Qwen3.5-4B"' in script, \
        "both published sizes should be selectable"
    assert 'os.environ.get("VEGA_SIZE") or "0.8b"' in script, "0.8b is the documented default"
    # the two backbones pool to different widths, so a shared cache would silently mix them
    assert 'CACHE = os.path.join(WORK, "features-" + SIZE)' in script
    assert 'OUT = os.path.join(WORK, "adapter-" + SIZE)' in script
    code = "\n".join(shell_cells())
    assert 'os.environ["VEGA_SIZE"] = "0.8b"' in code, "the notebook should set it explicitly"


def test_the_notebook_declares_the_accelerator_it_asserts():
    code = "\n".join(shell_cells())
    assert "torch.cuda.device_count() >= 2" in code, \
        "the notebook should fail loudly on a single-GPU runtime, not silently run half as fast"
