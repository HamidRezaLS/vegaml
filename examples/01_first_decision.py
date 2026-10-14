"""One state, two questions, one forward pass. The smallest complete call."""
import vegaml

v = vegaml.load()                                    # the 0.8B; load("4b") for the 4B

out = v.decide(
    "I was billed twice this month, refund me or I cancel.",
    {"team": {"type": "choice", "instructions": "Which team handles this?",
              "criteria": {"billing": "payments, invoices and refunds",
                           "technical": "bugs and outages",
                           "sales": "pricing and plans"}},
     "churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"}})

print(out["answers"]["team"]["choice"])              # billing
print(out["answers"]["churn"]["boolean"])            # the probability the statement holds
print(out["latency_ms"], "ms for both questions")    # both answered from one read of the state
