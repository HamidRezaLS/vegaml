# The cascade

Most decisions are easy. A cascade answers everything with the 0.8B and pays for the 4B only on
the questions the small engine cannot settle, so a stream of easy traffic runs at small-model
latency and the hard cases still get the big model.

```python
from vegaml.router import Cascade

c = Cascade()                      # "0.8b" first, "4b" for what it cannot settle
out = c.decide("I was billed twice, refund me or I cancel.",
               {"churn": {"type": "boolean",
                          "instructions": "Does the customer threaten to leave?"}})

out["answers"]["churn"]["tier"]    # which checkpoint answered: "0.8b" or "4b"
out["escalated"]                   # the question ids the first tier could not settle
out["unsettled"]                   # what even the last tier could not settle
```

## What "settled" means

Escalation is read off the engine's own calibrated machinery, never guessed from the raw
probability alone. An answer escalates when any of these hold:

* the engine **abstained**, or flagged the state as out of distribution (`unbound`);
* the **conformal set** is wider than `max_set` (default 1: the engine must be able to commit
  to a single option at the requested coverage);
* the **confidence** is under `floor` (default 0.75).

Both knobs are constructor arguments:

```python
Cascade(floor=0.9, max_set=2)      # stricter confidence, but a 2-option set may pass
```

`unsettled` is also importable on its own, if you want the same rule over a plain
`decide()` result.

## The honest end of the line

A question the *last* tier cannot settle keeps that tier's calibrated answer, and its id is
listed in `out["unsettled"]`. That list is the point of the cascade: it is the set of decisions
to hand to a human or a bigger system, instead of letting an unsure answer travel as if it were
a sure one.

## Costs

* Checkpoints load lazily. A cascade over easy traffic never loads the 4B at all; `c.loaded()`
  says which tiers have actually been paid for.
* `out["latency_ms"]` and `out["n_tokens"]` cover every tier that ran.
* Tiers take the same names as `vegaml.load()`, and an already-loaded model can be passed in
  place of a name. Any number of tiers works, smallest first; a one-tier cascade is just that
  model plus the `unsettled` list.
