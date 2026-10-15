# Questions

A request is a state and a mapping of question ids to questions. Three types exist, and every
answer carries calibrated probabilities, a confidence, a conformal prediction set at your
`alpha`, and an abstain flag at the configured risk.

## choice

Pick one option. `criteria` maps each option to the words that define it; writing real
definitions is worth more than clever instructions, because the option text is what the model
reads.

```python
{"team": {"type": "choice",
          "instructions": "Which team handles this?",
          "criteria": {"billing": "payments, invoices and refunds",
                       "technical": "bugs and outages",
                       "sales": "pricing and plans"}}}
```

A bare list (`"criteria": ["billing", "technical"]`) is accepted and means options without
definitions.

## boolean

Does the statement hold? The answer's `boolean` field is the probability of yes. Optional
criteria sharpen both sides:

```python
{"churn": {"type": "boolean",
           "instructions": "Does the customer threaten to leave?",
           "criteria": {"true": "shows intent to cancel", "false": "no such signal"}}}
```

## score

An integer level against ordered criteria. `criteria` is a list, one description per level, in
order; the answer carries the level (`score`) and the probability-weighted `expected_score`.

```python
{"severity": {"type": "score",
              "instructions": "How severe is the problem?",
              "criteria": ["cosmetic", "degraded for some users", "an outage"]}}
```

## Reading an answer

```python
a = out["answers"]["team"]
a["choice"]            # the picked option
a["confidence"]        # top probability, thermally calibrated
a["prediction_set"]    # the options the conformal guarantee keeps at your alpha
a["abstain"]           # True when confidence is below the risk threshold: route to a human
a["unbound"]           # True when the particle settled far from every well: likely out of distribution
```

Gate on `abstain` before acting; that flag is what the calibration is for.
