"""Steadier answers and tighter guarantees: test-time views, and the conformal alpha."""
import vegaml

v = vegaml.load()
q = {"team": {"type": "choice", "instructions": "Which team handles this?",
              "criteria": {"billing": "payments", "technical": "bugs", "sales": "pricing"}}}

# views=2 re-reads with options reversed, views=3 adds a prose rendering; probabilities
# average across the readings. Cost scales with views, steadiness is what you buy.
out = v.decide("Why was I charged before the trial ended?", q, views=2)
print(out["answers"]["team"]["view_agreement"])      # did the readings agree on the top option

# alpha is the conformal miscoverage: alpha="0.05" makes prediction_set carry the true
# answer 95% of the time, at the price of larger sets.
out = v.decide("Why was I charged before the trial ended?", q, alpha="0.05")
print(out["answers"]["team"]["prediction_set"])
