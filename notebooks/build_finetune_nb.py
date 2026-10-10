"""Build vega_finetune_2xT4_kaggle.ipynb.

The training script is NOT retyped into the notebook: the %%writefile cell is read from
vega_finetune.py at build time, so the file that was smoke-tested is byte-for-byte the file the
notebook writes. tests/test_notebook.py fails if the two ever drift.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "vega_finetune.py")
OUT = os.path.join(HERE, "vega_finetune_2xT4_kaggle.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def writefile(name, body):
    """A %%writefile cell carrying `body` unchanged -- trailing newline included, so the file the
    notebook writes is byte-for-byte the file the tests ran."""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": ("%%writefile " + name + "\n" + body).splitlines(True)}


def main():
    script = open(SCRIPT, encoding="utf-8").read()

    cells = [
        md("""
# Fine-tuning Vega on two T4s

A gated adapter for **[vegaml](https://pypi.org/project/vegaml/)** on a public dataset, start to
finish, on Kaggle's 2×T4.

* **Model:** `nandakishorm/vega-08b-public-intents`. Two sizes live in it: **0.8B** (the default
  here) and **4B**. Change `VEGA_SIZE` in the settings cell to switch; each size caches separately,
  so switching never mixes the two. While that repository is private you need `HF_TOKEN` set in the
  settings cell, otherwise the Hub answers 401 and the run stops at stage 1.
* **Task:** `dair-ai/emotion`, 16k messages over six emotions, written as a typed `choice` question.
* **Gate negatives:** `fancyzhx/ag_news`, a different task, so the adapter's gate learns when *not*
  to fire. An adapter that always fires is not routable.
* **Trained:** a rank-32 adapter on seven engine projections plus the gate, about 0.84M parameters.
  The language model is frozen and never trained.

**Before you run:** Settings → Accelerator → **GPU T4 ×2**, and Internet → **On**.

### Why only one stage uses both GPUs

The frozen backbone is the expensive part, and it runs exactly once per row. That read is sharded
across both T4s. What is actually trained then runs on the cached features, where an epoch is
seconds on a single GPU, so the second GPU is used where it helps and nowhere else.
"""),
        code("""
!pip install -q vegaml datasets
"""),
        md("""
## Settings

`VEGA_SIZE` is the only thing most runs change. `HF_TOKEN` is read from Kaggle Secrets if you have
added one, and skipped if you have not. You need it for a checkpoint that is private or gated, and
to push the adapter you train to your own account at the end. The datasets are public either way.
"""),
        code("""
import os

os.environ["VEGA_SIZE"] = "0.8b"          # "0.8b" (default) or "4b"

# A token is needed only for a checkpoint that is private or gated, or to push what you train.
# Add-ons -> Secrets -> add one named HF_TOKEN and this picks it up; with none, this is a no-op
# and the public path below runs unchanged.
try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:
    print("no HF_TOKEN from Kaggle Secrets (%s): continuing without one" % type(exc).__name__)

print("size:", os.environ["VEGA_SIZE"], "| token:", "set" if os.environ.get("HF_TOKEN") else "none")
"""),
        code("""
import torch

print(torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("  cuda:%d  %s  %.0f GB" % (i, p.name, p.total_memory / 2**30))
assert torch.cuda.device_count() >= 2, "set Accelerator to GPU T4 x2"
"""),
        md("""
## The script

Written to disk rather than run inline, because stage 1 is launched with `torchrun` as two
processes. It is the same file that is version-controlled and tested in the repository.
"""),
        writefile("vega_finetune.py", script),
        md("""
## Stage 1: frozen features on both GPUs

One read per row, sharded across the two T4s and cached to `/kaggle/working/features-$VEGA_SIZE`. This is the
long part: roughly 24k rows in total (16k train, 2k validation, 2k test, 4k gate negatives).
"""),
        code("""
!torchrun --nproc_per_node=2 vega_finetune.py extract --batch 32
"""),
        md("""
## Stage 2: the adapter

On cached features, so this is minutes at most. Thermal calibration and the conformal thresholds are
fitted afterwards on the validation split, which training never sees.
"""),
        code("""
!python vega_finetune.py train --epochs 3 --batch 64 --rank 32 --lr 3e-4
"""),
        md("""
## Stage 3: did it help?

Base engine against the fine-tuned adapter on the held-out test split: accuracy and expected
calibration error, plus how far the gate opens.
"""),
        code("""
!python vega_finetune.py eval --batch 64
"""),
        code("""
import json
r = json.load(open("/kaggle/working/finetune_results.json"))
print("test items:", r["n_test"])
print("base engine       acc %.4f   ECE %.2f" % (r["base"]["accuracy"], r["base"]["ece"]))
print("+ emotion adapter acc %.4f   ECE %.2f   mean gate %.3f"
      % (r["adapter"]["accuracy"], r["adapter"]["ece"], r["adapter"]["mean_gate"]))
print("delta             acc %+.4f  ECE %+.2f"
      % (r["adapter"]["accuracy"] - r["base"]["accuracy"], r["adapter"]["ece"] - r["base"]["ece"]))
"""),
        md("""
## Using the adapter

`/kaggle/working/adapter-$VEGA_SIZE/` now holds `emotion.safetensors` and `emotion.json` in the layout the
package already reads. Drop both into a checkpoint's `adapters/` directory and they load
automatically. The gate decides per question whether they apply, and the adapter's own calibration
travels with it.

```python
import vegaml
v = vegaml.load("your-copy-of-the-checkpoint")     # adapters/ picked up automatically
v.decide("I can't believe they remembered my birthday",
         {"emotion": {"type": "choice",
                      "instructions": "Which emotion does this message express?",
                      "criteria": {"sadness": "...", "joy": "...", "love": "...",
                                   "anger": "...", "fear": "...", "surprise": "..."}}})
```

The adapter records the SHA-256 of the base engine it was trained against, and the loader refuses it
against a different one rather than silently producing nonsense.
"""),
        code("""
!ls -la /kaggle/working/adapter-$VEGA_SIZE/ && cat /kaggle/working/adapter-$VEGA_SIZE/emotion.json
"""),
        md("""
## Optional: push it to your own account

Needs `HF_TOKEN` from the settings cell. Only the adapter and its metadata are uploaded, a few
hundred kilobytes, and the repository is created private unless you pass `--public`.
"""),
        code("""
# !python vega_finetune.py eval --batch 64 --push-to your-name/vega-emotion-adapter
"""),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
        fh.write("\n")
    print("wrote %s: %d cells, %d bytes" % (OUT, len(cells), os.path.getsize(OUT)))


if __name__ == "__main__":
    main()
