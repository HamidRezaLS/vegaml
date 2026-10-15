"""Fit small heads on your own labelled examples; auto mode uses them only where they help."""
import vegaml

v = vegaml.load()
questions = {"intent": {"type": "choice", "instructions": "What does the customer want?",
                        "criteria": {"refund": "money back", "cancel": "end the subscription",
                                     "question": "information only"}}}
examples = [
    ("I want my money back for the duplicate charge.", {"intent": "refund"}),
    ("Please close my account at the end of the month.", {"intent": "cancel"}),
    ("Which plan includes the API?", {"intent": "question"}),
    ("Refund the second charge please.", {"intent": "refund"}),
    ("How do I export my data?", {"intent": "question"}),
    ("Cancel everything, I am done.", {"intent": "cancel"}),
]

report = v.fit(examples, questions)
print(report["intent"])          # fitted?, cv_accuracy, whether auto will use it

# decide() is unchanged; each answer says which readout produced it.
out = v.decide("I would like a refund for March.", questions)
print(out["answers"]["intent"].get("readout"), out["answers"]["intent"]["choice"])
