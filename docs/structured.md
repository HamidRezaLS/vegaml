# Structured decisions

```bash
pip install "vegaml[structured]"
```

A pydantic model is the question sheet and the answer:

```python
from typing import Literal
from pydantic import BaseModel, Field
from vegaml.structured import decide_model

class Ticket(BaseModel):
    team: Literal["billing", "technical", "sales"] = Field(description="Which team handles this?")
    churn: bool = Field(description="Does the customer threaten to leave?")
    severity: int = Field(ge=1, le=3, description="How severe?",
                          json_schema_extra={"criteria": ["cosmetic", "degraded", "outage"]})

ticket, result = decide_model(v, "I was billed twice, refund me or I cancel.", Ticket)
ticket.team                      # "billing", validated
result["answers"]["team"]        # probabilities, prediction set, abstain: still all there
```

Field mapping: `bool` is a boolean question; `Literal[str, ...]` and str Enums are choices
over the allowed values; a bounded `int` (`ge`/`le`) is a score whose levels span the range,
with criteria from `json_schema_extra` or numbered when the range speaks for itself. The field
description is the instruction the model reads, so it is required.

The instance comes through pydantic validation, so an answer outside the field's type is a
`ValidationError`, never a quiet corruption, and the second return value is the decide()
dictionary untouched: going typed loses none of the calibrated detail.
