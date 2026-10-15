"""Calibration on labels, against a fake engine: temperature, coverage, persistence, refusals."""
from __future__ import annotations

import random

import pytest

from vegaml.calibrate import Calibration, calibrate, ece

QB = {"churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"}}
QC = {"team": {"type": "choice", "instructions": "Which team?",
               "criteria": {"billing": "payments", "technical": "bugs", "sales": "plans"}}}
QS = {"sev": {"type": "score", "instructions": "How severe?",
              "criteria": ["cosmetic", "degraded", "outage"]}}


class _Overconfident:
    """Says 0.97 for the label hidden in the state, but is right only when the state says so."""

    def __init__(self, labels=("false", "true")):
        self.labels = list(labels)

    def decide(self, state, questions, mode=None, alpha="0.1", views=1):
        claimed = state["claim"]
        probs = {k: (0.97 if k == claimed else 0.03 / (len(self.labels) - 1))
                 for k in self.labels}
        return {"answers": {q: {"probs": dict(probs)} for q in questions}}


def _boolean_examples(n=40, accuracy=0.7, seed=5):
    """The engine claims 0.97 but is right `accuracy` of the time: textbook overconfidence."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        claim = rng.choice(["false", "true"])
        truth = claim if rng.random() < accuracy else ("true" if claim == "false" else "false")
        out.append(({"claim": claim}, {"churn": truth}))
    return out


# ---------------------------------------------------------------------- fitting

def test_an_overconfident_engine_gets_a_temperature_above_one():
    cal = calibrate(_Overconfident(), _boolean_examples(), QB)
    r = cal.report["churn"]
    assert r["calibrated"] and r["temperature"] > 1.0
    assert r["ece_after"] < r["ece_before"]


def test_apply_moves_probability_but_never_the_answer():
    cal = calibrate(_Overconfident(), _boolean_examples(), QB)
    raw = _Overconfident().decide({"claim": "true"}, QB, mode="engine")
    out = cal.apply(raw)
    a = out["answers"]["churn"]
    assert a["calibrated"] and a["confidence"] < 0.97
    assert max(a["probs"], key=a["probs"].get) == "true"          # argmax invariant
    assert a["boolean"] == a["probs"]["true"]
    assert abs(sum(a["probs"].values()) - 1.0) < 1e-9
    assert raw["answers"]["churn"]["probs"]["true"] == 0.97        # the input is untouched


def test_the_conformal_set_widens_where_the_engine_was_wrong_often():
    cal = calibrate(_Overconfident(), _boolean_examples(accuracy=0.55), QB, alpha=0.1)
    out = cal.apply(_Overconfident().decide({"claim": "true"}, QB))
    # To cover 90% while being right barely over half the time, the set must hold both options.
    assert set(out["answers"]["churn"]["prediction_set"]) == {"false", "true"}


def test_score_and_choice_fields_are_recomputed():
    labels = ["0", "1", "2"]
    eng = _Overconfident(labels)
    rng = random.Random(7)
    ex = []
    for _ in range(30):
        claim = rng.choice(labels)
        truth = claim if rng.random() < 0.8 else rng.choice(labels)
        ex.append(({"claim": claim}, {"sev": truth}))
    cal = calibrate(eng, ex, QS)
    a = cal.apply(eng.decide({"claim": "2"}, QS))["answers"]["sev"]
    assert a["score"] == 2 and 0.0 <= a["expected_score"] <= 2.0

    eng2 = _Overconfident(["billing", "technical", "sales"])
    ex2 = [({"claim": c}, {"team": c}) for c in ["billing", "technical", "sales"] * 5]
    cal2 = calibrate(eng2, ex2, QC)
    assert cal2.apply(eng2.decide({"claim": "sales"}, QC))["answers"]["team"]["choice"] == "sales"


# ---------------------------------------------------------------------- the edges

def test_too_few_examples_leaves_the_question_alone_and_says_so():
    cal = calibrate(_Overconfident(), _boolean_examples(n=4), QB)
    assert cal.report["churn"]["calibrated"] is False
    assert "at least" in cal.report["churn"]["reason"]
    raw = _Overconfident().decide({"claim": "true"}, QB)
    assert cal.apply(raw)["answers"]["churn"] == raw["answers"]["churn"]


def test_a_label_outside_the_options_is_named():
    ex = _boolean_examples(n=12)
    ex[3] = (ex[3][0], {"churn": "maybe"})
    cal = calibrate(_Overconfident(), ex, QB)
    assert cal.report["churn"]["calibrated"] is False
    assert "'maybe'" in cal.report["churn"]["reason"]


def test_ece_is_zero_for_a_perfectly_honest_oracle():
    assert ece([(1.0, True)] * 10) == 0.0
    assert ece([(0.5, True), (0.5, False)] * 10) == 0.0
    with pytest.raises(ValueError):
        ece([])


# ---------------------------------------------------------------------- persistence

def test_save_load_round_trip(tmp_path):
    cal = calibrate(_Overconfident(), _boolean_examples(), QB)
    p = tmp_path / "calibration.json"
    cal.save(p)
    loaded = Calibration.load(p)
    raw = _Overconfident().decide({"claim": "false"}, QB)
    assert loaded.apply(raw) == cal.apply(raw)
    assert loaded.alpha == cal.alpha


# ---------------------------------------------------------------------- refusals

def test_the_refusals_name_the_problem():
    with pytest.raises(ValueError, match="no examples"):
        calibrate(_Overconfident(), [], QB)
    with pytest.raises(ValueError, match="miscoverage"):
        calibrate(_Overconfident(), _boolean_examples(), QB, alpha=1.5)
