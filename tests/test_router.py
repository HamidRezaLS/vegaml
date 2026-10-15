"""The cascade, run against fake checkpoints: escalation, laziness, aggregation, refusals."""
from __future__ import annotations

import pytest

from vegaml.router import Cascade, unsettled

Q2 = {"churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"},
      "team": {"type": "choice", "instructions": "Which team?",
               "criteria": {"billing": "payments", "technical": "bugs"}}}


class _FakeVega:
    """Answers from a script: {qid: answer_dict}. Records every call."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def decide(self, state, questions, mode=None, alpha="0.1", views=1):
        self.calls.append({"state": state, "questions": dict(questions), "mode": mode,
                           "alpha": alpha, "views": views})
        return {"answers": {q: dict(self.script[q]) for q in questions}, "n_tokens": 10}


def sure(p=0.95):
    return {"confidence": p, "prediction_set": ["yes"], "abstain": False, "unbound": False}


def unsure(p=0.4):
    return {"confidence": p, "prediction_set": ["yes", "no"], "abstain": False, "unbound": False}


# ---------------------------------------------------------------------- the settled rule

def test_unsettled_reads_the_engines_own_flags():
    assert not unsettled(sure())
    assert unsettled(unsure())                                  # wide set and low confidence
    assert unsettled({"confidence": 0.9, "abstain": True})
    assert unsettled({"confidence": 0.9, "unbound": True})
    assert unsettled({"confidence": 0.9, "prediction_set": ["a", "b"]})
    assert unsettled({"confidence": 0.5}, floor=0.75)
    assert not unsettled({"confidence": 0.5}, floor=0.5)


def test_a_head_answer_without_the_machinery_counts_as_settled():
    assert not unsettled({"confidence": 0.9, "readout": "ttt"})


# ---------------------------------------------------------------------- escalation

def test_settled_questions_never_wake_the_big_tier():
    loads = []
    small = _FakeVega({q: sure() for q in Q2})
    c = Cascade(tiers=("0.8b", "4b"),
                load_fn=lambda name: loads.append(name) or small)
    out = c.decide("the state", Q2)
    assert loads == ["0.8b"]
    assert c.loaded() == ["0.8b"]
    assert out["escalated"] == [] and out["unsettled"] == []
    assert {a["tier"] for a in out["answers"].values()} == {"0.8b"}


def test_only_the_unsettled_questions_escalate():
    small = _FakeVega({"churn": sure(), "team": unsure()})
    big = _FakeVega({"team": sure()})
    models = {"0.8b": small, "4b": big}
    c = Cascade(load_fn=lambda name: models[name])
    out = c.decide("the state", Q2)
    assert big.calls[0]["questions"] == {"team": Q2["team"]}      # churn stayed small
    assert out["answers"]["churn"]["tier"] == "0.8b"
    assert out["answers"]["team"]["tier"] == "4b"
    assert out["escalated"] == ["team"] and out["unsettled"] == []


def test_what_the_last_tier_cannot_settle_is_said_out_loud():
    small = _FakeVega({"churn": unsure(), "team": unsure()})
    big = _FakeVega({"churn": sure(), "team": {"confidence": 0.3, "abstain": True}})
    models = {"0.8b": small, "4b": big}
    out = Cascade(load_fn=lambda name: models[name]).decide("s", Q2)
    assert out["unsettled"] == ["team"]
    assert out["answers"]["team"]["tier"] == "4b"                 # the answer is still there
    assert sorted(out["escalated"]) == ["churn", "team"]


def test_a_single_tier_cascade_is_just_that_model_with_flags():
    one = _FakeVega({"churn": unsure(), "team": sure()})
    out = Cascade(tiers=("0.8b",), load_fn=lambda name: one).decide("s", Q2)
    assert out["unsettled"] == ["churn"] and out["escalated"] == []


def test_preloaded_models_are_used_without_loading():
    small = _FakeVega({q: sure() for q in Q2})
    c = Cascade(tiers=(small, "4b"), load_fn=lambda name: pytest.fail("should not load"))
    out = c.decide("s", Q2)
    assert out["answers"]["churn"]["tier"] == "tier0"             # unnamed model, positional name


# ---------------------------------------------------------------------- aggregation

def test_tokens_add_up_and_the_mode_is_the_calibrated_one():
    small = _FakeVega({"churn": sure(), "team": unsure()})
    big = _FakeVega({"team": sure()})
    models = {"0.8b": small, "4b": big}
    out = Cascade(load_fn=lambda name: models[name]).decide("s", Q2, views=3)
    assert out["n_tokens"] == 20
    assert out["mode"] == "cascade" and out["views"] == 3
    assert out["tiers"] == ["0.8b", "4b"]
    assert small.calls[0]["mode"] == "engine" and small.calls[0]["views"] == 3
    assert out["latency_ms"] >= 0


def test_nothing_loads_at_construction():
    Cascade(load_fn=lambda name: pytest.fail("construction must stay free"))


# ---------------------------------------------------------------------- refusals

def test_the_refusals_name_the_problem():
    with pytest.raises(ValueError, match="at least one tier"):
        Cascade(tiers=())
    with pytest.raises(ValueError, match="probability"):
        Cascade(floor=1.5)
    with pytest.raises(ValueError, match="max_set"):
        Cascade(max_set=0)
    with pytest.raises(ValueError, match="no questions"):
        Cascade(load_fn=lambda n: _FakeVega({})).decide("s", {})
