"""Decisions as LangChain runnables and LangGraph nodes: route, gate, annotate.

    from vegaml.integrations.langchain import VegaDecision, VegaGate

    decide = VegaDecision(model=v, questions={...})            # or url="http://host:8000"
    chain = prompt | decide                                    # LCEL: state in, decide() out

    gate = VegaGate(model=v, questions={"injection": {...}}, deny={"injection": 0.5})
    safe = gate | llm                                          # raises GateViolation instead
                                                               # of letting the input through

    graph.add_conditional_edges("triage", decide.route("team"))  # LangGraph: the choice is
                                                                 # the edge name

Everything here degrades gracefully: with ``langchain_core`` installed these are real
runnables (``invoke``/``batch``/``stream`` and async via the base class); without it they are
plain callables with the same ``invoke`` and ``batch``, so the module imports, tests and works
anywhere. The model can be in-process (``model=``) or a ``vegaml-serve`` deployment (``url=``),
and edge clients using ``url`` never import torch.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

try:
    from langchain_core.runnables import RunnableConfig, RunnableSerializable  # noqa: F401
    _BASE: type = RunnableSerializable
    _HAVE_LANGCHAIN = True
except ImportError:                                            # pragma: no cover - CI has it
    _BASE = object
    _HAVE_LANGCHAIN = False

__all__ = ["VegaDecision", "VegaGate", "GateViolation", "state_text"]


class GateViolation(ValueError):
    """An input a :class:`VegaGate` refused to pass, with the decision that refused it."""

    def __init__(self, message: str, violations: Dict[str, float], result: Dict):
        super().__init__(message)
        self.violations = violations
        self.result = result


def state_text(value: Any, state_key: Union[str, Callable, None] = None):
    """The state inside a chain input: a string, a dict's text field, or the last user turn.

    LangChain inputs arrive as strings, prompt dicts, message lists, or messages whose
    ``content`` is a list of typed blocks. This digs the decidable state out of each, and hands
    dicts and lists it cannot interpret to ``decide()`` unchanged, which accepts them.
    """
    if callable(state_key):
        return state_key(value)
    if isinstance(state_key, str) and isinstance(value, Mapping) and state_key in value:
        return state_text(value[state_key])
    if hasattr(value, "content"):
        return _block_text(value.content)
    if isinstance(value, str) or value is None:
        return value or ""
    if isinstance(value, Mapping):
        for key in ("input", "text", "query", "prompt", "message", "content"):
            if key in value:
                return state_text(value[key])
        if isinstance(value.get("messages"), list):
            return state_text(value["messages"])
        return dict(value)
    if isinstance(value, Sequence):
        text = _block_text(value)
        if text is not value:
            return text
        for m in reversed(value):
            role = getattr(m, "type", None) or (m.get("role") if isinstance(m, Mapping) else None)
            if role in ("human", "user"):
                return state_text(m)
        return list(value)
    return str(value)


def _block_text(content: Any) -> Any:
    """The text of a message content value; a block list with no text comes back unchanged."""
    if not isinstance(content, list):
        return content
    parts = [b if isinstance(b, str) else b.get("text")
             for b in content if isinstance(b, (str, Mapping))]
    parts = [p for p in parts if isinstance(p, str)]
    return "\n".join(parts) if parts else content


class _Deciding(_BASE):
    """The shared engine access: one loaded model or one vegaml-serve URL."""

    model: Any = None
    url: Optional[str] = None
    questions: Dict[str, Dict] = {}
    mode: str = "engine"
    alpha: str = "0.1"
    state_key: Any = None
    timeout: float = 30.0

    def __init__(self, model=None, url: Optional[str] = None, questions: Optional[Mapping] = None,
                 mode: str = "engine", alpha: str = "0.1",
                 state_key: Union[str, Callable, None] = None, timeout: float = 30.0, **kw):
        if (model is None) == (url is None):
            raise ValueError("pass exactly one of model= (in process) or url= (vegaml-serve)")
        if not questions:
            raise ValueError("no questions")
        kw.update(model=model, url=url, questions=dict(questions), mode=mode, alpha=alpha,
                  state_key=state_key, timeout=timeout)
        if _HAVE_LANGCHAIN:
            super().__init__(**kw)
        else:
            for k, v in kw.items():
                setattr(self, k, v)

    model_config = {"arbitrary_types_allowed": True}           # langchain_core is pydantic v2

    def _decide(self, value) -> Dict:
        state = state_text(value, self.state_key)
        if self.model is not None:
            return self.model.decide(state, self.questions, mode=self.mode, alpha=self.alpha)
        body = json.dumps({"state": state, "questions": self.questions,
                           "mode": self.mode, "alpha": self.alpha}).encode()
        req = urllib.request.Request(self.url.rstrip("/") + "/v1/decide", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    # The plain-callable face; with langchain_core these are overridden semantics-compatibly.
    def invoke(self, input, config=None, **kw):
        return self._transform(input)

    def batch(self, inputs: List, config=None, **kw) -> List:
        return [self._transform(i) for i in inputs]

    def __call__(self, input):
        return self._transform(input)

    def _transform(self, value):
        raise NotImplementedError


class VegaDecision(_Deciding):
    """State in, the full ``decide()`` result out; compose it like any other runnable."""

    def _transform(self, value) -> Dict:
        return self._decide(value)

    def route(self, question_id: str) -> Callable[[Any], str]:
        """A callable for LangGraph's conditional edges: the state in, the option id out."""
        if question_id not in self.questions:
            raise ValueError("no question %r in this decision" % question_id)

        def _route(value) -> str:
            ans = self._decide(value)["answers"][question_id]
            if "choice" in ans:
                return str(ans["choice"])
            if "boolean" in ans:
                return "true" if ans["boolean"] >= 0.5 else "false"
            return str(ans.get("score"))
        return _route


class VegaGate(_Deciding):
    """Pass the input through untouched, or raise :class:`GateViolation`, never both.

    ``deny`` maps boolean question ids to thresholds: the gate raises when the probability of
    yes reaches the threshold. The raised error carries the offending probabilities and the
    full decision, so the caller can log exactly why the input was stopped.
    """

    deny: Dict[str, float] = {}

    def __init__(self, deny: Optional[Mapping[str, float]] = None, **kw):
        questions = kw.get("questions") or {}
        deny = dict(deny or {})
        if not deny:
            raise ValueError("deny= names no question; a gate that cannot refuse gates nothing")
        unknown = sorted(set(deny) - set(questions))
        if unknown:
            raise ValueError("deny names %s, not in the questions" % ", ".join(unknown))
        if _HAVE_LANGCHAIN:
            super().__init__(deny=deny, **kw)
        else:
            super().__init__(**kw)
            self.deny = deny

    def _transform(self, value):
        result = self._decide(value)
        violations = {}
        for qid, threshold in self.deny.items():
            p = result["answers"][qid].get("boolean")
            if p is None:
                probs = result["answers"][qid].get("probs") or {}
                p = float(probs.get("true", probs.get("True", 0.0)))
            if p >= threshold:
                violations[qid] = round(float(p), 4)
        if violations:
            raise GateViolation("gate refused the input: %s" % ", ".join(
                "%s at %.2f" % pair for pair in sorted(violations.items())), violations, result)
        return value
