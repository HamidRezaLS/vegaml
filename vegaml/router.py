"""A confidence cascade over checkpoints: the small one answers, the big one gets the rest.

    from vegaml.router import Cascade

    c = Cascade()                      # "0.8b" first, "4b" for what it cannot settle
    out = c.decide("I was billed twice, refund me or I cancel.",
                   {"churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"}})

Most decisions are easy, and the point of a cascade is to pay the small model's latency for
those while the hard ones still get the big model. Whether a question is "settled" is read off
the engine's own calibrated machinery rather than guessed: an answer escalates when the engine
abstained, flagged the state as out of distribution, returned a conformal set wider than
``max_set``, or put its confidence under ``floor``. Everything else is final at the tier that
produced it.

Each answer carries ``tier`` naming the checkpoint that produced it. A question the last tier
still cannot settle keeps that tier's calibrated answer and its id is listed in
``out["unsettled"]``: that list is the cascade's honest output, the decisions to hand to a
human or a bigger system instead of pretending the model knew.

Checkpoints load lazily, so a stream of easy traffic never pays for the big model at all.
"""
from __future__ import annotations

import time
from typing import Dict, List, Mapping, Optional, Sequence

__all__ = ["Cascade", "unsettled"]


def unsettled(answer: Mapping, floor: float = 0.75, max_set: int = 1) -> bool:
    """Whether one answer is too uncertain to be final, by the engine's own flags.

    Works on a single entry of ``result["answers"]``. Missing fields count as settled, so a
    TTT-head answer, which carries no conformal set and no abstain flag, is never escalated on
    machinery it does not have.
    """
    if answer.get("abstain") or answer.get("unbound"):
        return True
    pset = answer.get("prediction_set")
    if pset is not None and len(pset) > max_set:
        return True
    conf = answer.get("confidence")
    return conf is not None and conf < floor


class Cascade:
    """Checkpoints in cost order, each answering only what the ones before could not settle.

    ``tiers`` names the checkpoints smallest first, by the same names :func:`vegaml.load`
    takes; already-loaded models can be passed in place of names. ``floor`` and ``max_set``
    set what "settled" means (see :func:`unsettled`). ``mode`` defaults to ``"engine"``, the
    calibrated readout the escalation rule is built on.
    """

    def __init__(self, tiers: Sequence = ("0.8b", "4b"), floor: float = 0.75, max_set: int = 1,
                 mode: str = "engine", alpha: str = "0.1", device: Optional[str] = None,
                 load_fn=None):
        if not tiers:
            raise ValueError("a cascade needs at least one tier")
        if not 0.0 <= floor <= 1.0:
            raise ValueError("floor is a probability, got %r" % (floor,))
        if max_set < 1:
            raise ValueError("max_set must be at least 1, got %r" % (max_set,))
        self.tiers = list(tiers)
        self.floor = floor
        self.max_set = max_set
        self.mode = mode
        self.alpha = alpha
        self.device = device
        self._load_fn = load_fn
        self._models: Dict[int, object] = {i: t for i, t in enumerate(self.tiers)
                                           if not isinstance(t, str)}

    # ------------------------------------------------------------------ tiers
    def tier_name(self, i: int) -> str:
        t = self.tiers[i]
        return t if isinstance(t, str) else getattr(t, "name", "tier%d" % i)

    def model(self, i: int):
        """The tier's model, loading it on first use."""
        if i not in self._models:
            if self._load_fn is not None:
                self._models[i] = self._load_fn(self.tiers[i])
            else:
                import vegaml
                self._models[i] = vegaml.load(self.tiers[i], device=self.device)
        return self._models[i]

    # ------------------------------------------------------------------ deciding
    def decide(self, state, questions: Mapping, views: int = 1) -> Dict:
        """Answer every question at the cheapest tier that settles it.

        The result has :meth:`vegaml.Vega.decide`'s shape, with three additions: each answer
        carries ``tier``, ``escalated`` lists the question ids the first tier could not settle,
        and ``unsettled`` lists the ids the *last* tier could not settle either.
        """
        if not questions:
            raise ValueError("no questions")
        started = time.perf_counter()
        answers: Dict[str, Dict] = {}
        escalated: List[str] = []
        unsettled_ids: List[str] = []
        n_tokens = 0
        remaining: Dict[str, Mapping] = dict(questions)
        for i in range(len(self.tiers)):
            last = i == len(self.tiers) - 1
            out = self.model(i).decide(state, remaining, mode=self.mode,
                                       alpha=self.alpha, views=views)
            n_tokens += out.get("n_tokens", 0)
            name = self.tier_name(i)
            still: Dict[str, Mapping] = {}
            for qid, ans in out["answers"].items():
                open_q = unsettled(ans, floor=self.floor, max_set=self.max_set)
                if open_q and not last:
                    still[qid] = remaining[qid]
                    continue
                ans = dict(ans)
                ans["tier"] = name
                answers[qid] = ans
                if i > 0:
                    escalated.append(qid)
                if open_q:
                    unsettled_ids.append(qid)
            remaining = still
            if not remaining:
                break
        return {"answers": answers, "mode": "cascade",
                "tiers": [self.tier_name(i) for i in range(len(self.tiers))],
                "escalated": sorted(escalated), "unsettled": sorted(unsettled_ids),
                "latency_ms": round(1000 * (time.perf_counter() - started), 1),
                "n_tokens": n_tokens, "views": views}

    # ------------------------------------------------------------------ introspection
    def loaded(self) -> List[str]:
        """The tiers that have actually been paid for, by name."""
        return [self.tier_name(i) for i in sorted(self._models)]
