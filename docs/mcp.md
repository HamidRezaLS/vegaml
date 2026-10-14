# MCP

```bash
pip install "vegaml[mcp]"
vegaml-mcp-server --model 4b
```

A stdio MCP server with two tools, for agents that speak the protocol:

- **decide** `(state_text, questions, mode?, alpha?, views?)` returns the full decide()
  dictionary. The agent gets a decision instrument, not another text channel: typed answers
  with probabilities it can gate on, from a model on the machine it is already on.
- **model_info** reports the loaded checkpoint without running anything.

A client entry looks like:

```json
{"mcpServers": {"vegaml": {"command": "vegaml-mcp-server", "args": ["--model", "4b"]}}}
```

The checkpoint loads on first use and stays resident. A bad question comes back as a tool
error carrying the model's own message. Tell the agent to gate on `abstain`: an answer flagged
abstain is the model saying it is not confident enough at the configured risk, and acting on
it anyway discards the calibration.
