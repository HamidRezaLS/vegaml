"""vegaml-mcp-server: decide() as an MCP tool over stdio, for agents that speak the protocol.

    pip install "vegaml[mcp]"
    vegaml-mcp-server --model 4b

Two tools. `decide` takes a state and a questions object, exactly as the Python API does, and
returns the full decide() dictionary: calibrated probabilities, prediction sets, abstain
flags, physics readouts. `model_info` says what checkpoint is loaded without running anything.

An agent wired to this server gets a decision instrument rather than another text channel:
the answer to "which team handles this?" arrives typed, with a probability the agent can gate
on, from a model running on the machine it is already on. The checkpoint loads on first use
and stays resident; calls serialize around it, since a decision is one short forward pass.
"""
from __future__ import annotations

import argparse
import threading
from typing import Any, Dict, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


def create_server(model=None, model_name: Optional[str] = None, load_kw: Optional[Dict] = None) -> MCPServer:
    """The MCP server. Pass a loaded model (tests do), or a name and the first call loads it."""
    srv = MCPServer("vegaml")
    state = {"model": model, "lock": threading.Lock()}
    load_kw = load_kw or {}

    def _model():
        with state["lock"]:
            if state["model"] is None:
                import vegaml
                state["model"] = vegaml.load(model_name, **load_kw)
        return state["model"]

    @srv.tool()
    def decide(state_text: str, questions: Dict[str, Dict[str, Any]], mode: Optional[str] = None,
               alpha: str = "0.1", views: int = 1) -> Dict:
        """Answer typed questions about a state in one forward pass.

        questions maps an id to {"type": "choice"|"boolean"|"score", "instructions": str,
        "criteria": ...}. Every answer carries calibrated probabilities, a confidence, a
        conformal prediction set and an abstain flag; gate on abstain before acting.
        """
        v = _model()
        with state["lock"]:
            try:
                return v.decide(state_text, questions, mode=mode, alpha=alpha, views=views)
            except (ValueError, RuntimeError) as e:
                # A bad question reaches the agent as a tool error with the model's own words,
                # not as a protocol-level crash it cannot read.
                raise ToolError(str(e)) from e

    @srv.tool()
    def model_info() -> Dict:
        """The loaded checkpoint: backbone, context limit, default mode, fitted heads."""
        v = _model()
        cfg = getattr(getattr(v, "engine", None), "cfg", {}) or {}
        return {"model": model_name or "default", "backbone": cfg.get("backbone_name"),
                "max_len": cfg.get("max_len"), "mode": getattr(v, "mode", None),
                "heads_fitted": len(getattr(v, "heads", {}))}

    return srv


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="vegaml-mcp-server", description="decide() as an MCP tool over stdio.")
    ap.add_argument("--model", default=None,
                    help="checkpoint: nothing for the 0.8B, '4b', a Hub id or a path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    args = ap.parse_args(argv)
    load_kw = {}
    if args.device:
        load_kw["device"] = args.device
    if args.max_len:
        load_kw["max_len"] = args.max_len
    create_server(model_name=args.model, load_kw=load_kw).run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
