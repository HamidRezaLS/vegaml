"""Export the engine and run it through onnxruntime. pip install "vegaml[onnx]"."""
from vegaml.hub import fetch
from vegaml.onnx_export import ONNXEngine, export_engine

path = fetch(None)                                   # the 0.8B checkpoint directory
export_engine(path, "engine.onnx")
eng = ONNXEngine("engine.onnx")
print("inputs:", eng.inputs)                         # feats, opts, opt_mask, qtype [, chunks...]
# Feed it pooled perception features; the README's architecture section says what each is.
