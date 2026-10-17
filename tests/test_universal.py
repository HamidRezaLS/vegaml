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


def _universal_repo(tmp_path):
    """A downloaded universal repo: root = one reader, a sibling reader as a subfolder."""
    root_cfg = {"port": "port.safetensors", "backbone_name": "Qwen3.5-0.8B", "d_shared": 16}
    (tmp_path / "vega_config.json").write_text(json.dumps(root_cfg))
    sub = tmp_path / "gemma-3-1b-it"
    sub.mkdir()
    (sub / "vega_config.json").write_text(json.dumps(dict(root_cfg, backbone_name="gemma-3-1b-it")))
    return tmp_path


def test_reader_dir_picks_by_backbone_name(tmp_path):
    from vegaml.hub import reader_dir

    repo = _universal_repo(tmp_path)
    assert reader_dir(str(repo), "Qwen/Qwen3.5-0.8B") == str(repo)          # the root reader
    assert reader_dir(str(repo), "google/gemma-3-1b-it") == str(repo / "gemma-3-1b-it")
    assert reader_dir(str(repo), "/models/gemma-3-1b-it/") == str(repo / "gemma-3-1b-it")
    with pytest.raises(ValueError) as e:
        reader_dir(str(repo), "mistralai/Mistral-7B")
    assert "gemma-3-1b-it" in str(e.value)       # the error names what the repo does ship


def test_reader_dir_leaves_classic_checkpoints_alone(tmp_path):
    from vegaml.hub import reader_dir

    (tmp_path / "vega_config.json").write_text(json.dumps({"backbone_name": "Qwen3.5-0.8B"}))
    assert reader_dir(str(tmp_path), "/local/weights/of/the/same/backbone") == str(tmp_path)
    assert reader_dir(str(tmp_path / "no-config-here"), "x") == str(tmp_path / "no-config-here")


def test_fetch_requests_the_reader_subfolder(tmp_path, monkeypatch):
    import sys
    import types

    from vegaml import hub

    seen = {}

    def fake_snapshot_download(repo_id, **kw):
        seen.update(kw, repo_id=repo_id)
        return str(tmp_path)

    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    hub.fetch("someone/vega-U", reader="gemma-3-1b-it")
    pats = seen["allow_patterns"]
    assert "gemma-3-1b-it/vega_config.json" in pats
    assert "gemma-3-1b-it/port.safetensors" in pats
    # the data-only rule survives the widening: still nothing executable, no wildcard suffix
    for p in pats:
        assert p.endswith((".json", ".safetensors", ".npz")), p


def test_from_pretrained_routes_backbone_to_its_reader(tmp_path, monkeypatch):
    import vegaml
    from vegaml import _api

    repo = _universal_repo(tmp_path)
    monkeypatch.setattr(vegaml, "fetch", lambda *a, **k: str(repo))
    got = {}
    monkeypatch.setattr(_api, "Vega", lambda path, **kw: got.update(path=path, **kw) or object())
    vegaml.Vega.from_pretrained("someone/vega-U", backbone="google/gemma-3-1b-it")
    assert got["path"] == str(repo / "gemma-3-1b-it")
    assert got["backbone"] == "google/gemma-3-1b-it"
