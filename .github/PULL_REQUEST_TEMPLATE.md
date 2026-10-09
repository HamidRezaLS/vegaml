## What this changes

<!-- One or two sentences. What behaviour is different after this merges? -->

## Why

<!-- The problem, not the patch. If it fixes an issue, link it. -->

## How it was verified

<!-- Commands run and what came back. "CI is green" is not verification on its own: say what you
     measured, on what, and what the numbers were. -->

## Checklist

- [ ] Tests cover the change, and they fail without it
- [ ] No secret, token or private path in the diff (the Security workflow scans the whole tree, not
      just this diff)
- [ ] No new way for a downloaded checkpoint to reach code execution: nothing imports or evaluates a
      file fetched at runtime, and weights still load through safetensors
- [ ] Any published number in the README or docstrings is one that was measured, and the diff says
      how
