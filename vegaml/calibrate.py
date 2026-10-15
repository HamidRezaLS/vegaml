"""Refit the probabilities on your own labels: temperature per question, conformal on top.

    from vegaml.calibrate import calibrate, Calibration

    cal = calibrate(v, examples, questions)        # examples as in Vega.fit()
    out = cal.apply(v.decide(state, questions, mode="engine"))
    cal.save("calibration.json")                   # and Calibration.load() later

The engine's published calibration is measured on its own test split. Your traffic is not that
split, and a probability that is honest there can be over- or under-confident here. This module
fits, per question, the one-parameter correction with your labels as the ground truth: a
temperature on the engine's probabilities, then a conformal threshold *on the corrected
probabilities*, so the prediction sets cover at the rate you asked for on data like yours.

What it does not do: change the answer. The argmax is invariant under temperature, so
:meth:`Calibration.apply` moves probabilities, confidence, prediction sets and the abstain
flag, never which option wins. Sharper answers need labels too, and that is :meth:`Vega.fit`.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Mapping, Sequence, Tuple

from . import label_ids

__all__ = ["calibrate", "Calibration", "ece", "canon_probs"]

_CLAMP = 1e-6
_GOLD = (math.sqrt(5.0) - 1.0) / 2.0


def canon_probs(probs: Mapping, options: Sequence[str]) -> Dict[str, float]:
    """The probabilities re-keyed onto the question's own option ids.

    The engine's boolean answers key their probabilities "False"/"True" and put Python booleans
    in the prediction set, while :func:`vegaml.label_ids` speaks "false"/"true"; scores can come
    back keyed by int. This maps each key onto the option it names, case-insensitively, so label
    arithmetic never depends on which spelling produced the answer. A key naming no option is a
    :class:`ValueError`, since dropping it would silently renormalise.
    """
    by_fold = {o.lower(): o for o in options}
    out: Dict[str, float] = {}
    for k, p in probs.items():
        o = by_fold.get(str(k).lower())
        if o is None:
            raise ValueError("probability key %r names no option of %s" % (k, list(options)))
        out[o] = float(p)
    return out


def ece(rows: Sequence[Tuple[float, bool]], bins: int = 10) -> float:
    """Expected calibration error of (confidence, was_correct) rows, equal-width bins."""
    if not rows:
        raise ValueError("no rows")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(c, ok) for c, ok in rows if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if inside:
            conf = sum(c for c, _ in inside) / len(inside)
            acc = sum(1.0 for _, ok in inside if ok) / len(inside)
            total += len(inside) / len(rows) * abs(acc - conf)
    return total


def _rescale(probs: Mapping[str, float], temperature: float) -> Dict[str, float]:
    logits = {k: math.log(min(max(p, _CLAMP), 1.0 - _CLAMP)) / temperature
              for k, p in probs.items()}
    peak = max(logits.values())
    weights = {k: math.exp(z - peak) for k, z in logits.items()}
    norm = sum(weights.values())
    return {k: w / norm for k, w in weights.items()}


def _nll(rows: Sequence[Tuple[Mapping[str, float], str]], temperature: float) -> float:
    total = 0.0
    for probs, label in rows:
        total -= math.log(max(_rescale(probs, temperature)[label], _CLAMP))
    return total / len(rows)


def _fit_temperature(rows: Sequence[Tuple[Mapping[str, float], str]],
                     lo: float = 0.05, hi: float = 20.0, steps: int = 60) -> float:
    """The temperature minimising mean NLL, by golden-section search on [lo, hi]."""
    a, b = math.log(lo), math.log(hi)
    c, d = b - _GOLD * (b - a), a + _GOLD * (b - a)
    fc, fd = _nll(rows, math.exp(c)), _nll(rows, math.exp(d))
    for _ in range(steps):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - _GOLD * (b - a)
            fc = _nll(rows, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + _GOLD * (b - a)
            fd = _nll(rows, math.exp(d))
    return round(math.exp((a + b) / 2.0), 4)


def _conformal_qhat(scores: List[float], alpha: float) -> float:
    """The split-conformal quantile: coverage holds at 1 - alpha on exchangeable data."""
    n = len(scores)
    rank = min(math.ceil((n + 1) * (1.0 - alpha)), n)
    return sorted(scores)[rank - 1]


class Calibration:
    """Per-question temperature and conformal threshold, fitted by :func:`calibrate`.

    ``report`` says, per question, what was fitted and what it changed, including the in-sample
    ECE before and after. Questions with too few examples are left exactly as the engine
    answered them, and the report names them.
    """

    def __init__(self, alpha: float, questions: Dict[str, Dict], report: Dict[str, Dict]):
        self.alpha = alpha
        self.questions = questions      # qid -> {type, labels, temperature, qhat, examples}
        self.report = report

    # ------------------------------------------------------------------ applying
    def apply(self, result: Mapping) -> Dict:
        """A copy of a decide() result with every calibrated question's answer corrected."""
        out = dict(result)
        out["answers"] = {}
        for qid, ans in result["answers"].items():
            fitted = self.questions.get(qid)
            if fitted is None or "probs" not in ans:
                out["answers"][qid] = ans
                continue
            ans = dict(ans)
            probs = _rescale(canon_probs(ans["probs"], fitted["labels"]), fitted["temperature"])
            best = max(probs, key=probs.get)
            pset = [k for k in fitted["labels"] if probs[k] >= 1.0 - fitted["qhat"]]
            ans["probs"] = {k: round(p, 6) for k, p in probs.items()}
            ans["confidence"] = round(probs[best], 6)
            ans["abstain"] = not pset
            ans["prediction_set"] = pset or [best]
            if fitted["type"] == "boolean":
                ans["boolean"] = ans["probs"]["true"]
            elif fitted["type"] == "score":
                ans["score"] = int(best)
                ans["expected_score"] = round(sum(int(k) * p for k, p in probs.items()), 6)
            else:
                ans["choice"] = best
            ans["calibrated"] = True
            out["answers"][qid] = ans
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        payload = {"alpha": self.alpha, "questions": self.questions, "report": self.report}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path) -> "Calibration":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls(payload["alpha"], payload["questions"], payload["report"])


def calibrate(model, examples: Sequence[Tuple], questions: Mapping,
              alpha: float = 0.1, min_examples: int = 10,
              views: int = 1) -> Calibration:
    """Fit a :class:`Calibration` from labelled examples, one engine read per example.

    ``examples`` has :meth:`Vega.fit`'s shape: ``(state, {question_id: label})``, with an
    optional third element carrying per-example questions. ``alpha`` is the miscoverage the
    conformal sets should hold on data like these examples.
    """
    if not examples:
        raise ValueError("no examples")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha is a miscoverage rate in (0, 1), got %r" % (alpha,))
    fitted: Dict[str, Dict] = {}
    report: Dict[str, Dict] = {}
    for qid, q in questions.items():
        labels = label_ids(q)
        rows: List[Tuple[Dict[str, float], str]] = []
        bad: List[str] = []
        for example in examples:
            state, marks = example[0], example[1]
            if qid not in marks:
                continue
            label = str(marks[qid])
            if label not in set(labels):
                bad.append(label)
                continue
            asked = (example[2] if len(example) > 2 else {}).get(qid, q)
            answer = model.decide(state, {qid: asked}, mode="engine",
                                  views=views)["answers"][qid]
            rows.append((canon_probs(answer["probs"], labels), label))
        if bad:
            report[qid] = {"calibrated": False, "examples": len(rows),
                           "reason": ("label(s) %s in the examples are not options of this "
                                      "question, whose options are %s"
                                      % (", ".join(repr(b) for b in sorted(set(bad))[:5]), labels))}
            continue
        if len(rows) < min_examples:
            report[qid] = {"calibrated": False, "examples": len(rows),
                           "reason": "needs at least %d labelled examples" % min_examples}
            continue
        temperature = _fit_temperature(rows)
        scaled = [(_rescale(p, temperature), y) for p, y in rows]
        qhat = _conformal_qhat([1.0 - p[y] for p, y in scaled], alpha)
        before = ece([(max(p.values()), max(p, key=p.get) == y) for p, y in rows])
        after = ece([(max(p.values()), max(p, key=p.get) == y) for p, y in scaled])
        fitted[qid] = {"type": q["type"], "labels": labels,
                       "temperature": temperature, "qhat": round(qhat, 6),
                       "examples": len(rows)}
        report[qid] = {"calibrated": True, "examples": len(rows),
                       "temperature": temperature, "qhat": round(qhat, 6),
                       "ece_before": round(before, 4), "ece_after": round(after, 4),
                       "note": "ECE measured on the fitting examples themselves"}
    return Calibration(alpha, fitted, report)
