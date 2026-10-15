# MLX

```bash
pip install "vegaml[mlx]"        # Apple silicon
```

```python
from vegaml.mlx_engine import MLXEngine

eng = MLXEngine.from_dir("path/to/checkpoint")       # vega_config.json + engine.safetensors
out = eng(feats, opts, opt_mask, qtype)              # numpy in, numpy out
```

The decision engine's forward, reproduced in MLX from the same `engine.safetensors`, read
through safetensors' numpy loader: a process that only runs the settling computation needs
neither torch nor a CUDA runtime. Scope is the base engine at eval semantics, score rail and
long-state memory reader included; gated adapters stay with the torch path.

Precision, stated plainly: `device="gpu"` (the default) runs on Metal, whose matmul
accumulates below full float32. Readouts land within about 1e-2 of the torch numbers while
decisions and settling steps are identical, and the test suite pins exactly that claim.
`device="cpu"` is strict float32 and matches torch to 1e-4; use it when comparing numbers
rather than serving them.
