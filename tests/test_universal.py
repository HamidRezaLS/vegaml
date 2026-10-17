"""vega-U: the universal checkpoint path.

A universal per-backbone dir wraps the shared engine in a port: raw features are standardized
with the backbone's extraction statistics, linearly mapped into the shared space, and the
engine runs there with prestandardized=True. These tests pin that the wrapper computes exactly
that, loads from the files a real checkpoint ships, and stays interchangeable with a bare
VegaEngine for everything _api reads off it.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from vegaml._common import QTYPES, VegaEngine
from vegaml._universal import Port, UniversalEngine

CFG = dict(latent_dim=8, world_dim=8, proj_hidden=16, n_steps=3, dt_max=0.1, settle_eps=1e-3)
D_IN, D_SHARED, LF, LO = 24, 16, 2, 1


def _checkpoint(tmp_path):
    """A universal dir the way write_backbone_dir lays it out: port + stats + config keys."""
    from safetensors.torch import save_file

    torch.manual_seed(0)
    port = Port(D_IN, D_SHARED, LF)
    save_file(port.state_dict(), os.path.join(tmp_path, "port.safetensors"))
    rng = np.random.default_rng(0)
    np.savez(
        os.path.join(tmp_path, "stats.npz"),
        mu=rng.normal(size=(LF, 3, D_IN)).astype(np.float32),
        sd=(0.5 + rng.random((LF, 3, D_IN))).astype(np.float32),
        omu=rng.normal(size=(LO, D_IN)).astype(np.float32),
        osd=(0.5 + rng.random((LO, D_IN))).astype(np.float32),
    )
    cfg = {"d_shared": D_SHARED, "port": "port.safetensors", "raw_stats": "stats.npz"}
    with open(os.path.join(tmp_path, "vega_config.json"), "w") as f:
        json.dump(cfg, f)
    return cfg, port


def test_wrapper_matches_manual_standardize_and_port(tmp_path):
    cfg, port = _checkpoint(tmp_path)
    torch.manual_seed(1)
    engine = VegaEngine(D_SHARED, LF, LO, CFG).eval()
    uni = UniversalEngine(str(tmp_path), cfg, engine).eval()

    B, K = 4, 3
    torch.manual_seed(2)
    feats = torch.randn(B, LF, 3, D_IN)
    opts = torch.randn(B, K, LO, D_IN)
    mask = torch.ones(B, K, dtype=torch.bool)
    qtype = torch.full((B,), QTYPES["choice"])

    with torch.no_grad():
        got = uni(feats, opts, mask, qtype, return_traj=False)
        f = port.items((feats - uni.mu) / uni.sd)
        o = port.opts((opts - uni.omu) / uni.osd)
        want = engine(f, o, mask, qtype, prestandardized=True, return_traj=False)
    assert torch.allclose(got["logits"], want["logits"], atol=1e-5)


def test_engine_attributes_fall_through(tmp_path):
    cfg, _ = _checkpoint(tmp_path)
    engine = VegaEngine(D_SHARED, LF, LO, CFG).eval()
    uni = UniversalEngine(str(tmp_path), cfg, engine)
    assert uni.max_steps == engine.max_steps
    assert uni.memory is engine.memory
    assert uni.adapter_mode == engine.adapter_mode
    with pytest.raises(AttributeError):
        uni.not_a_real_attribute


def test_api_wraps_on_port_key(tmp_path, monkeypatch):
    """Vega.__init__ swaps the engine for the wrapper exactly when the config names a port."""
    from safetensors.torch import save_file

    cfg, _ = _checkpoint(tmp_path)
    cfg.update(
        {"d_model": D_SHARED, "feature_layers": [1, 3], "option_layers": [3],
         "max_len": 64, "opt_max_tokens": 16, "backbone": "unused", **CFG}
    )
    with open(os.path.join(tmp_path, "vega_config.json"), "w") as f:
        json.dump(cfg, f)
    torch.manual_seed(1)
    engine = VegaEngine(D_SHARED, LF, LO, CFG)
    save_file(engine.state_dict(), os.path.join(tmp_path, "engine.safetensors"))

    from vegaml import _api

    class FakePerception:
        feature_layers = [1, 3]

    monkeypatch.setattr(_api, "Perception", lambda *a, **k: FakePerception())
    v = _api.Vega(str(tmp_path), device="cpu")
    assert isinstance(v.engine, UniversalEngine)
    v.set_adapters("off")
    assert v._core() is v.engine.engine
    assert v.engine.engine.adapter_mode == "off"
