# Calibration on your labels

The engine's published calibration is measured on its own test split. Your traffic is not that
split, and a probability that is honest there can be over- or under-confident here. This page is
the fix: fit a one-parameter correction per question on a few dozen of your own labelled
examples, and the probabilities, prediction sets and abstain flags hold on data like yours.

```python
from vegaml.calibrate import calibrate, Calibration

examples = [                                  # the same shape Vega.fit() takes
    ("I was billed twice, refund me or I cancel.", {"churn": "true"}),
    ("The export button does nothing on Safari.",  {"churn": "false"}),
    # ... a few dozen of these
]

cal = calibrate(v, examples, questions, alpha=0.1)
print(cal.report)                             # per question: temperature, ECE before and after

out = cal.apply(v.decide(state, questions, mode="engine"))
cal.save("calibration.json")                  # Calibration.load() in the serving process
```

## What it fits

Per question, two numbers, in order:

1. **A temperature** on the engine's probabilities, minimising likelihood on your labels. Above
   one it softens overconfidence, below one it sharpens timidity.
2. **A conformal threshold** on the *corrected* probabilities, so the prediction sets cover
   your labels at `1 - alpha` on data exchangeable with the examples. Where the engine is wrong
   often, the sets widen; that honesty is the feature.

## What it never does

`apply()` moves probabilities, confidence, `prediction_set` and `abstain`. It never changes
which option wins: the argmax is invariant under temperature. If the *answers* are wrong,
calibration cannot help; fit a head on the same labels with `v.fit()` instead, and the two
compose (calibrate the engine's readout, fit heads for the questions with enough signal).

## How many labels

`calibrate()` refuses a question with fewer than `min_examples` (default 10) and leaves it
exactly as the engine answered it, saying so in the report. Thirty to a hundred labels per
question is where the temperature stops moving. The report's ECE figures are measured on the
fitting examples themselves; hold some out if you want an unbiased number.
