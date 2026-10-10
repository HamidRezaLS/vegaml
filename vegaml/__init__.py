"""vegaml: typed decisions from a frozen language model and a small physics engine.

    import vegaml

    v = vegaml.load()              # the 0.8B, the default
    v = vegaml.load("4b")          # the 4B, same repository
    v.decide("I was billed twice, refund me or I cancel.",
             {"team": {"type": "choice", "instructions": "Which team handles this?",
                       "criteria": {"billing": "payments and refunds", "technical": "bugs"}}})

Two readouts over the same frozen features, chosen with `mode`:

* ``"engine"`` (the default) is the physics engine and its adapters. Zero-shot: it answers a question
  it has never seen, and every answer carries a calibrated probability, a conformal set and an
  abstain flag. Every benchmark figure published for this model is measured in this mode.
* ``"ttt"`` is a small head fitted here, on your labelled examples, over the same features. Sharper
  where your label space matches the examples, and worthless outside them. It carries no calibration
  and no abstain flag, so it will answer confidently when it should not.
* ``"both"`` returns both, side by side, which is the only honest way to pick between them.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .hub import default_token, fetch
from .ttt import TTTHead

__all__ = ["load", "Vega", "TTTHead", "fetch", "default_token", "__version__"]
__version__ = "0.6.0"

MODES = ("engine", "ttt", "both")


class Vega:
    """A loaded checkpoint, with both readouts over one set of frozen features.

    `mode` sets the default readout for :meth:`decide`; "engine" unless you change it.
    """

    def __init__(self, engine, mode: str = "engine"):
        if mode not in MODES:
            raise ValueError("mode must be one of %s, got %r" % (", ".join(MODES), mode))
        self.engine = engine
        self.mode = mode
        self.heads: Dict[str, Tuple[TTTHead, Optional[float]]] = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, repo_or_path: Optional[str] = None, backbone: Optional[str] = None,
                        device: Optional[str] = None, adapters: str = "auto",
                        mode: str = "engine", revision: Optional[str] = None,
                        token: Union[str, bool, None] = None,
                        max_len: Optional[int] = None) -> "Vega":
        """Load a checkpoint from the Hub or from a local directory.

        `token` is optional. Left as None it picks up `HF_TOKEN` from the environment if one is
        set, which is what makes a private or gated repository load; with nothing set, the public
        checkpoints load anonymously as before. Pass ``token=False`` to ignore the environment.
        """
        # Imported here, not at module import: torch and transformers are only needed once a
        # checkpoint is actually being loaded.
        from ._api import Vega as _Engine

        path = fetch(repo_or_path, revision=revision, token=token)
        # backbone=None lets the engine read backbone_name out of the checkpoint's own config,
        # which is what makes load("4b") pull Qwen3.5-4B rather than the 0.8B one.
        return cls(_Engine(path, backbone=backbone, device=device, adapters=adapters,
                           max_len=max_len), mode=mode)

    # ------------------------------------------------------------------ reading
    def features(self, state, question: Mapping):
        """Pooled frozen features for one (state, question): what both readouts see."""
        import numpy as np
        import torch

        from ._common import build_prompt, option_texts, serialize_state

        v = self.engine
        qi = v._internal(question)
        k = len(option_texts(qi))
        text = state if isinstance(state, str) else serialize_state(state)
        ids, spans = build_prompt(v.perc.tok, text, qi, v.cfg["max_len"],
                                  option_order=list(range(k)),
                                  opt_max_tokens=v.cfg["opt_max_tokens"],
                                  query_first_max_state=v.cfg.get("query_first_max_state", 0))
        with torch.no_grad():
            f, _ = v.perc.pooled([{"ids": np.asarray(ids, dtype=np.int32), "spans": spans}],
                                 prefix_min=v.cfg.get("prefix_min_tokens", 4096))
        return f[0].float().cpu().numpy().ravel()

    # ------------------------------------------------------------------ fitting
    def fit(self, examples: Sequence[Tuple], questions: Mapping, folds: int = 5,
            min_examples: int = 6) -> Dict[str, Dict]:
        """Fit one TTT head per question from labelled examples.

        `examples` is a sequence of ``(state, {question_id: label})``. One forward pass is run per
        example per question, so the cost is ``len(examples) × len(questions)`` reads of the frozen
        backbone. That is seconds on a GPU for a few dozen examples, and it is paid once.

        Returns, per question id, the number of examples used and a cross-validated accuracy. **Read
        that number.** A head at or below chance has learned nothing, and its answers are noise
        dressed up as probabilities.
        """
        report: Dict[str, Dict] = {}
        for qid, q in questions.items():
            X, y = [], []
            for state, labels in examples:
                if qid in labels:
                    X.append(self.features(state, q))
                    y.append(str(labels[qid]))
            if len(X) < min_examples:
                report[qid] = {"fitted": False, "examples": len(X),
                               "reason": "needs at least %d labelled examples" % min_examples}
                continue
            labels_for_q = label_ids(q)
            head = TTTHead(labels_for_q).fit(X, y)
            cv = head.cross_val_accuracy(X, y, folds=folds)
            chance = 1.0 / max(len(labels_for_q), 2)
            self.heads[qid] = (head, cv)
            # cv is None when there are too few examples to cross-validate. That is "unknown", not
            # "fine": reporting False there would read as a pass the data cannot support.
            report[qid] = {"fitted": True, "examples": len(X), "cv_accuracy": cv,
                           "chance": round(chance, 3),
                           "at_or_below_chance": None if cv is None else cv <= chance}
            if cv is None:
                report[qid]["cv_note"] = ("too few examples for %d-fold cross-validation, so the "
                                          "head's accuracy is unknown; %d or more would measure it"
                                          % (folds, 2 * folds))
        return report

    # ------------------------------------------------------------------ deciding
    def decide(self, state, questions: Mapping, mode: Optional[str] = None, alpha: str = "0.1",
               views: int = 1) -> Dict:
        """Answer every question from one read of the state.

        With ``mode="engine"`` the result is the engine's own, unchanged. With ``"ttt"`` each answer
        comes from the fitted head for that question. With ``"both"`` the result carries
        ``answers`` (engine) and ``ttt`` side by side.
        """
        mode = mode or self.mode
        if mode not in MODES:
            raise ValueError("mode must be one of %s, got %r" % (", ".join(MODES), mode))
        if mode in ("ttt", "both"):
            missing = [q for q in questions if q not in self.heads]
            if missing and mode == "ttt":
                raise RuntimeError(
                    "no TTT head for %s. Call fit(examples, questions) first, or use "
                    "mode='engine'." % ", ".join(sorted(missing)))
        if mode == "ttt":
            return {"mode": "ttt", "answers": self._ttt_answers(state, questions)}
        out = self.engine.decide(state, questions, alpha=alpha, views=views)
        if mode == "engine":
            out["mode"] = "engine"
            return out
        out["mode"] = "both"
        out["ttt"] = self._ttt_answers(state, questions, skip_missing=True)
        return out

    def _ttt_answers(self, state, questions: Mapping, skip_missing: bool = False) -> Dict:
        answers: Dict[str, Dict] = {}
        for qid, q in questions.items():
            hv = self.heads.get(qid)
            if hv is None:
                if skip_missing:
                    continue
                raise RuntimeError("no TTT head for %r" % qid)
            head, cv = hv
            probs = head.predict(self.features(state, q), allowed=label_ids(q))
            best = max(probs, key=probs.get)
            answers[qid] = {"probs": probs, "choice": best, "confidence": probs[best],
                            "cv_accuracy": cv}
        return answers

    # ------------------------------------------------------------------ images
    def decide_image(self, image, questions: Mapping, **kw) -> Dict:
        """The same typed interface over a picture, through the backbone's frozen vision encoder."""
        if getattr(self, "_vision", None) is None:
            from ._vision import VegaVision

            self._vision = VegaVision(self.engine.model_dir,
                                      backbone=self.engine.cfg.get("backbone_name"),
                                      device=str(getattr(self.engine, "device", "cpu")), **kw)
        return self._vision.decide_image(image, questions)

    def __repr__(self) -> str:
        return "Vega(mode=%r, heads=%d, max_len=%d)" % (
            self.mode, len(self.heads), self.engine.cfg.get("max_len", 0))


def label_ids(question: Mapping) -> List[str]:
    """The option ids a readout must span, for any of the three answer types."""
    t = question["type"]
    if t == "boolean":
        return ["false", "true"]
    if t == "score":
        return [str(i) for i in range(len(question.get("criteria") or []))]
    return [str(k) for k in question["criteria"]]


def load(repo_or_path: Optional[str] = None, **kw) -> Vega:
    """Load a checkpoint.

    With no argument you get the 0.8B, which is the default and the one every published figure
    measures. ``load("4b")`` gets the 4B from the same repository. Anything else is treated as a
    Hub repository id or a local path. See :meth:`Vega.from_pretrained` for the other arguments.
    """
    return Vega.from_pretrained(repo_or_path, **kw)
