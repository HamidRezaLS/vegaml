"""A pydantic model as the question sheet and the answer. pip install "vegaml[structured]"."""
from typing import Literal

from pydantic import BaseModel, Field

import vegaml
from vegaml.structured import decide_model


class Ticket(BaseModel):
    team: Literal["billing", "technical", "sales"] = Field(description="Which team handles this?")
    churn: bool = Field(description="Does the customer threaten to leave?")
    severity: int = Field(ge=1, le=3, description="How severe is the problem?",
                          json_schema_extra={"criteria": ["cosmetic", "degraded", "outage"]})


v = vegaml.load()
ticket, result = decide_model(v, "I was billed twice, refund me or I cancel.", Ticket)
print(ticket)                                        # validated, typed
print(result["answers"]["team"]["prediction_set"])   # the calibrated detail is still there
