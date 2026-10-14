"""States can be structured; the record is serialized for the model, keys and all."""
import vegaml

v = vegaml.load()
state = {"subject": "Invoice 4821 charged twice",
         "customer_since": "2023-04",
         "turns": ["I was billed twice this month.",
                   "Agent: I can see two charges, looking into it.",
                   "If this is not fixed today I am cancelling."]}
out = v.decide(state, {"churn": {"type": "boolean",
                                 "instructions": "Does the customer threaten to leave?"}})
print(out["answers"]["churn"])
