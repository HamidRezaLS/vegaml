"""The cascade: the 0.8B answers what it can settle, the 4B gets the rest."""
from vegaml.router import Cascade

c = Cascade()                      # loads nothing yet; tiers load on first use

tickets = [
    "I was billed twice this month, refund me or I cancel.",
    "The export button does nothing on Safari 18.",
    "Thinking about the enterprise plan, who do I talk to?",
]
questions = {
    "team": {"type": "choice", "instructions": "Which team handles this?",
             "criteria": {"billing": "payments and refunds", "technical": "bugs and errors",
                          "sales": "plans and pricing"}},
    "churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"},
}

for state in tickets:
    out = c.decide(state, questions)
    for qid, ans in out["answers"].items():
        print("%-5s %-10s tier=%-5s p=%.2f" % (
            qid, ans.get("prediction", "?"), ans["tier"], ans.get("confidence", 0.0)))
    if out["unsettled"]:
        print("  hand these to a human: %s" % ", ".join(out["unsettled"]))
print("tiers actually loaded: %s" % ", ".join(c.loaded()))
