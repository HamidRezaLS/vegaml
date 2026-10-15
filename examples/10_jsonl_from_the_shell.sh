#!/bin/sh
# A file of states through one checkpoint load, from the shell. Output is JSONL for jq.
cat > /tmp/tickets.jsonl <<'JSON'
{"state": "I was billed twice this month, refund me or I cancel."}
{"state": "How do I rotate my API key?"}
{"state": "The export has been running for six hours."}
JSON

vegaml --jsonl /tmp/tickets.jsonl \
    --boolean "urgent=Does this need a reply today?" \
    --choice "team=Which team handles this?=billing,technical,sales" \
  | jq -c '{team: .answers.team.choice, urgent: .answers.urgent.boolean}'
