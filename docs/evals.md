# Evals on your labels

Nobody's test split is your traffic, and the published numbers for any decision model, this one
included, are measured on its own. `vegaml-eval` measures the model on your labelled decisions
instead, which is the number that should decide whether you ship it.

```bash
vegaml-eval decisions.jsonl --questions @questions.json
```

Each JSONL row is one labelled state:

```json
{"state": "I was billed twice, refund me or I cancel.", "labels": {"churn": "true"}}
```

`--questions` is the same sheet `decide()` takes, inline JSON or `@file`. A row may carry its
own `"questions"` where the instruction varies per example. From Python it is one call:

```python
from vegaml.evals import evaluate
report = evaluate(v, rows, questions)
```

## What the report says

Per question and pooled overall:

* `accuracy`, against your labels;
* `ece`, the gap between claimed confidence and measured accuracy, the honesty number;
* `coverage` and `mean_set_size`: how often the label was inside the conformal set, and how
  wide the set had to be to achieve that. At `alpha=0.1` coverage should sit near 0.9, and a
  set that covers by always holding every option is measured, not hidden;
* `abstain_rate`, plus `latency_ms_p50` / `p95` and `n_tokens_per_decision` overall.

## What to do with a bad number

The three numbers point at three different fixes. Low `accuracy` with honest `ece` means the
engine does not know your task: fit heads on the same labels with `v.fit()`. Decent accuracy
with a bad `ece` means it knows but is miscalibrated on your traffic: refit with
[calibration](calibration.md), then eval again on held-out rows. Coverage far from `1 - alpha`
with sane accuracy usually means the same thing, and the calibration page's conformal refit is
the cure. Thirty to a hundred labelled rows per question is enough to start trusting the
direction of these numbers; the eval costs one `decide()` per row.
