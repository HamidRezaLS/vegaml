"""The MCP server against a fake checkpoint: tool surface, round trip, lazy load, errors."""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp", reason="pip install 'vegaml[mcp]'")
pytest.importorskip("mcp.server.mcpserver", reason="needs mcp 2.x")

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from vegaml.mcp_server import create_server  # noqa: E402

Q = {"churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"}}


class _FakeVega:
    mode = "auto"
    heads: dict = {}

    def __init__(self):
        self.calls = []
        self.engine = type("E", (), {"cfg": {"backbone_name": "test/backbone", "max_len": 73000}})()

    def decide(self, state, questions, mode=None, alpha="0.1", views=1):
        self.calls.append({"state": state, "questions": questions, "mode": mode,
                           "alpha": alpha, "views": views})
        if "boom" in questions:
            raise ValueError("unknown question type 'boom'")
        return {"answers": {q: {"confidence": 0.9} for q in questions},
                "latency_ms": 1.0, "n_tokens": 7, "views": views}


def test_both_tools_are_listed():
    srv = create_server(model=_FakeVega())
    names = {t.name for t in asyncio.run(srv.list_tools())}
    assert names == {"decide", "model_info"}


def test_decide_round_trips_the_full_result():
    v = _FakeVega()
    srv = create_server(model=v, model_name="fake")
    res = asyncio.run(srv.call_tool("decide", {"state_text": "the state", "questions": Q,
                                               "mode": "engine", "alpha": "0.05", "views": 2}))
    assert not res.is_error
    assert res.structured_content["result"]["answers"]["churn"]["confidence"] == 0.9
    assert v.calls[0] == {"state": "the state", "questions": Q, "mode": "engine",
                          "alpha": "0.05", "views": 2}


def test_model_info_reports_the_checkpoint():
    srv = create_server(model=_FakeVega(), model_name="fake")
    res = asyncio.run(srv.call_tool("model_info", {}))
    assert res.structured_content["result"] == {"model": "fake", "backbone": "test/backbone",
                                                "max_len": 73000, "mode": "auto", "heads_fitted": 0}


def test_lazy_load_happens_once(monkeypatch):
    import vegaml
    v = _FakeVega()
    loads = []
    monkeypatch.setattr(vegaml, "load", lambda name, **kw: loads.append((name, kw)) or v)
    srv = create_server(model_name="4b", load_kw={"device": "cpu"})
    for _ in range(2):
        asyncio.run(srv.call_tool("decide", {"state_text": "s", "questions": Q}))
    assert loads == [("4b", {"device": "cpu"})]


def test_a_model_error_surfaces_as_a_tool_error_with_the_message():
    srv = create_server(model=_FakeVega())
    with pytest.raises(ToolError, match="unknown question type"):
        asyncio.run(srv.call_tool("decide", {"state_text": "s", "questions": {
            "boom": {"type": "boom", "instructions": "x"}}}))


def test_entry_point_and_extra_are_declared():
    import pathlib
    import sys
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    root = pathlib.Path(__file__).resolve().parent.parent
    py = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["scripts"]["vegaml-mcp-server"] == "vegaml.mcp_server:main"
    assert any(d.startswith("mcp") for d in py["project"]["optional-dependencies"]["mcp"])
