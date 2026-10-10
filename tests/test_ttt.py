"""The TTT readout, on synthetic features: no checkpoint, no network."""
from __future__ import annotations

import numpy as np
import pytest

from vegaml import label_ids
from vegaml.ttt import TTTHead


def _separable(n=40, d=48, classes=("a", "b", "c"), seed=0):
    """Features that are genuinely separable, so a working head must get them right."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(classes), d)) * 3.0
    X, y = [], []
    for i, c in enumerate(classes):
        X.append(centres[i] + rng.normal(scale=0.35, size=(n, d)))
        y += [c] * n
    return np.vstack(X), y


def test_fit_then_predict_recovers_separable_classes():
    X, y = _separable()
    head = TTTHead(["a", "b", "c"]).fit(X, y)
    for x, label in zip(X[::7], y[::7], strict=True):
        p = head.predict(x)
        assert max(p, key=p.get) == label
        assert pytest.approx(sum(p.values()), rel=1e-5) == 1.0


def test_probabilities_are_a_distribution_over_the_declared_labels():
    X, y = _separable()
    p = TTTHead(["a", "b", "c"]).fit(X, y).predict(X[0])
    assert set(p) == {"a", "b", "c"}
    assert all(0.0 <= v <= 1.0 for v in p.values())


def test_allowed_restricts_and_renormalises():
    X, y = _separable()
    head = TTTHead(["a", "b", "c"]).fit(X, y)
    p = head.predict(X[0], allowed=["b", "c"])
    assert set(p) == {"b", "c"}
    assert pytest.approx(sum(p.values()), rel=1e-5) == 1.0


def test_cross_val_is_high_on_separable_data_and_chance_on_noise():
    X, y = _separable()
    good = TTTHead(["a", "b", "c"]).cross_val_accuracy(X, y)
    assert good is not None and good > 0.9

    rng = np.random.default_rng(1)
    Xn = rng.normal(size=(60, 48))
    yn = [["a", "b"][i % 2] for i in range(60)]
    noise = TTTHead(["a", "b"]).cross_val_accuracy(Xn, yn)
    # The number that tells a caller the head learned nothing. It must not flatter itself.
    assert noise is not None and noise < 0.75


def test_cross_val_returns_none_when_there_is_too_little_data():
    X, y = _separable(n=2, classes=("a", "b"))
    assert TTTHead(["a", "b"]).cross_val_accuracy(X, y, folds=5) is None


def test_label_ids_covers_all_three_answer_types():
    assert label_ids({"type": "boolean", "criteria": {"true": "t", "false": "f"}}) == ["false", "true"]
    assert label_ids({"type": "score", "criteria": ["low", "mid", "high"]}) == ["0", "1", "2"]
    assert label_ids({"type": "choice", "criteria": {"billing": "x", "tech": "y"}}) == ["billing", "tech"]


# --------------------------------------------------------------- fitting contracts
class _Engine:
    cfg = {"max_len": 73728}
    model_dir = "/nowhere"

    def decide(self, state, questions, alpha="0.1", views=1):
        return {"answers": {q: {"probs": {"a": 1.0}, "choice": "a"} for q in questions}}


def _four_option_question():
    return {"type": "choice", "instructions": "which?",
            "criteria": {k: "" for k in ("a", "b", "c", "d")}}


def test_fit_reads_each_example_through_its_own_question_when_given_one():
    """The failure this prevents: when what varies sits in the instructions rather than the state,
    one shared question gives every example the same prompt and the head is fitted on noise."""
    import vegaml

    seen = []
    v = vegaml.Vega(_Engine())
    v.features = lambda state, q: seen.append(q["instructions"]) or [float(len(seen))]
    q = _four_option_question()
    examples = [({}, {"t": "a"}, {"t": dict(q, instructions="utterance %d" % i)})
                for i in range(8)]
    v.fit(examples, {"t": q}, min_examples=4)
    assert len(set(seen)) == 8, "every example must be read through its own question"


def test_fit_still_accepts_two_tuples():
    import vegaml

    v = vegaml.Vega(_Engine())
    v.features = lambda state, q: [float(state)]
    report = v.fit([(i, {"t": "a" if i % 2 else "b"}) for i in range(8)],
                   {"t": _four_option_question()}, min_examples=4)
    assert report["t"]["examples"] == 8


def test_a_head_at_or_below_chance_is_not_kept():
    import numpy as np

    import vegaml

    rng = np.random.default_rng(0)
    v = vegaml.Vega(_Engine())
    noise = rng.normal(size=(40, 8))
    v.features = lambda state, q: noise[state]
    # Labels independent of the features: nothing to learn, so cross-validation cannot beat chance.
    examples = [(i, {"t": "abcd"[i % 4]}) for i in range(40)]
    report = v.fit(examples, {"t": _four_option_question()})
    assert report["t"]["at_or_below_chance"] is True
    assert report["t"]["fitted"] is False
    assert report["t"]["used_by_auto"] is False
    assert "t" not in v.heads, "a head that learned nothing must not be served"
    assert "reason" in report["t"]
