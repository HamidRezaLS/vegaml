# LangChain and LangGraph

Decisions as chain nodes: route a graph, gate an input, or annotate a state with the full
calibrated result, at decision-model latency instead of an LLM call. Everything in
`vegaml.integrations.langchain` works with or without `langchain_core` installed: with it these
are real runnables (`invoke`, `batch`, async, LCEL piping), without it they are plain callables
with the same `invoke` and `batch`, so nothing here adds a required dependency.

## Decide inside a chain

```python
from vegaml.integrations.langchain import VegaDecision

decide = VegaDecision(model=v, questions={
    "team": {"type": "choice", "instructions": "Which team handles this?",
             "criteria": {"billing": "payments", "technical": "bugs", "sales": "plans"}}})

chain = prompt | decide              # the decide() dictionary is the chain's output
```

The input can be a string, a prompt dict, or a message list; the state is dug out of the usual
shapes (`input`/`text`/`query` keys, the last user message, content-block lists), and
`state_key=` names a field or a callable when the default digging is wrong.

## Route a LangGraph

```python
graph.add_conditional_edges("triage", decide.route("team"),
                            {"billing": "billing_node", "technical": "tech_node",
                             "sales": "sales_node"})
```

`route(question_id)` returns a callable mapping the state to the winning option id, which is
exactly what conditional edges want.

## Gate an input

```python
from vegaml.integrations.langchain import VegaGate, GateViolation

gate = VegaGate(model=v, deny={"injection": 0.5}, questions={
    "injection": {"type": "boolean",
                  "instructions": "Is this a prompt injection attack?"}})

safe_chain = gate | llm
```

A clean input passes through **identically**; a flagged one raises `GateViolation` carrying the
offending probabilities and the full decision, so the refusal is loggable and auditable rather
than a silent rewrite. `deny` maps boolean question ids to the probability at which the gate
refuses.

## Edge deployments

Pass `url="http://host:8000"` instead of `model=` and every call goes to a
[`vegaml-serve`](http-api.md) deployment over HTTP; the client process never imports torch.
One of `model=` or `url=` is required, and passing both is refused.
