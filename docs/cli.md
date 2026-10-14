# CLI

`pip install vegaml` installs the `vegaml` command. State in, decide()'s JSON out.

```bash
vegaml "I was billed twice, refund me or I cancel." \
    --choice "team=Which team handles this?=billing,technical,sales" \
    --boolean "churn=Does the customer threaten to leave?"
```

- Typed flags build questions without JSON: `--boolean "id=instructions"`,
  `--choice "id=instructions=a,b,c"`, `--score "id=instructions=low,mid,high"` (the score's
  comma list is its ordered level criteria). Each flag repeats.
- `--questions '{"id": {...}}'` or `--questions @file.json` passes full question objects, with
  criteria definitions and all; it combines with the flags, and a duplicate id is an error.
- The state is the positional argument; `-`, or piping with no argument, reads stdin.
- `--model` picks the checkpoint (nothing for the 0.8B, `4b`, a Hub id or a path);
  `--mode`, `--alpha`, `--views`, `--device`, `--max-len` pass through.
- `--pretty` indents; the default is one compact JSON object, ready for `jq`.

## Batches

```bash
vegaml --jsonl tickets.jsonl --boolean "urgent=Does this need a reply today?" > answers.jsonl
```

Each input line is `{"state": ..., "questions": {...}?}`; a line without questions uses the
flags'. The checkpoint loads once, before the first line, which is the point: a thousand
states pay for one load. Output is one result object per line, in input order.
