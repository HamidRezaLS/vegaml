"""Call a running vegaml-serve from any language; this is the python shape of it."""
import json
import urllib.request

req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/decide",
    data=json.dumps({
        "state": "I was billed twice, refund me or I cancel.",
        "questions": {"churn": {"type": "boolean",
                                "instructions": "Does the customer threaten to leave?"}},
    }).encode(),
    headers={"content-type": "application/json"})
with urllib.request.urlopen(req) as r:
    out = json.load(r)
print(out["answers"]["churn"])                       # same fields as the Python API
