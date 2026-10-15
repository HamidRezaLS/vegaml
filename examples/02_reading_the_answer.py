"""Everything an answer carries, and which fields to gate on before acting."""
import vegaml

v = vegaml.load()
out = v.decide("The dashboard returns 500 for every user since 09:00.",
               {"severity": {"type": "score", "instructions": "How severe is the problem?",
                             "criteria": ["cosmetic", "degraded for some users", "an outage"]}})

a = out["answers"]["severity"]
print("level:", a["score"])                  # the picked level, 0-based
print("expected:", a["expected_score"])      # probability-weighted level
print("probs:", a["probs"])                  # calibrated, they are meant to be believed
print("set:", a["prediction_set"])           # conformal set at alpha=0.1
print("abstain:", a["abstain"])              # True: send it to a human instead
print("unbound:", a["unbound"])              # True: likely out of distribution
