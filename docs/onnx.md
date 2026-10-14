# ONNX

```bash
pip install "vegaml[onnx]"
```

```python
from vegaml.onnx_export import export_engine, ONNXEngine

export_engine("path/to/checkpoint", "engine.onnx")
eng = ONNXEngine("engine.onnx")                      # onnxruntime, CPU provider by default
out = eng(feats, opts, opt_mask, qtype)              # numpy in, numpy out
```

What exports is the settling computation: pooled perception features in, well energies and the
physics readouts out (`logits`, `zT`, `kinetic`, `grad_norm`, `nearest`, `steps_used`). The
integrator unrolls into the graph at the checkpoint's own step count; batch size and option
count stay dynamic axes, so one file serves any request shape. A checkpoint with the
long-state memory reader exports with `chunks`/`chunk_mask` inputs, and an all-false mask
reproduces the chunkless path exactly.

Parity with the torch engine is held to 1e-4 by the test suite, including the score rail: the
export's example batch always contains a score question, because the tracer freezes python
branches and would otherwise drop the rail from the graph.

The perception backbone is a standard transformers model with its own mature export paths;
this module ships the part that is unique to vega.
