"""Refit the probabilities on your own labels, and keep the result next to the model."""
import vegaml
from vegaml.calibrate import calibrate

v = vegaml.load()

questions = {
    "churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"},
}
examples = [
    ("I was billed twice this month, refund me or I cancel.", {"churn": "true"}),
    ("The export button does nothing on Safari 18.", {"churn": "false"}),
    ("Cancel everything, I am done with this product.", {"churn": "true"}),
    ("How do I add a teammate to my workspace?", {"churn": "false"}),
    ("Third outage this week. We are evaluating alternatives.", {"churn": "true"}),
    ("Invoice 4411 lists a seat we removed in March.", {"churn": "false"}),
    ("If this is not fixed by Friday we walk.", {"churn": "true"}),
    ("Love the new dashboard, just wondering about API limits.", {"churn": "false"}),
    ("Our CFO asked me to review whether we still need this.", {"churn": "true"}),
    ("Password reset emails land in spam for one user.", {"churn": "false"}),
]

cal = calibrate(v, examples, questions, alpha=0.1)
print(cal.report["churn"])              # temperature, ECE before and after, on your labels

out = cal.apply(v.decide("Fix the invoice or we cancel the contract.", questions,
                         mode="engine"))
print(out["answers"]["churn"])

cal.save("calibration.json")            # Calibration.load("calibration.json") where you serve
