"""The engine on Apple silicon without torch. pip install "vegaml[mlx]"."""
from vegaml.hub import fetch
from vegaml.mlx_engine import MLXEngine

eng = MLXEngine.from_dir(fetch(None))                # gpu stream; device="cpu" for strict parity
print("engine loaded:", eng.D, "latent dims,", eng.max_steps, "settling steps")
# Feed it pooled perception features; outputs carry the same names as the torch engine.
