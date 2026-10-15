# Docker

The image wraps `vegaml-serve`; the checkpoint downloads on first request and persists in a
volume, so restarts do not re-download.

```bash
docker compose up                              # http://localhost:8000
# or
docker build -t vegaml .
docker run -p 8000:8000 -v vegaml-hub:/root/.cache/huggingface vegaml
```

`VEGAML_MODEL` picks the checkpoint (empty for the 0.8B, `4b` for the 4B). The image is
CPU-only by design: it exists so the HTTP API runs anywhere a container does, and a GPU host
is better served by `pip install "vegaml[serve]"` against its own torch build.
