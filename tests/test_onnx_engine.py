"""The exported graph against the torch engine: same readouts, dynamic shapes, memory inputs."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime", reason="pip install 'vegaml[onnx]'")

from vegaml._common import QTYPES, VegaEngine  # noqa: E402
from vegaml.onnx_export import ONNXEngine, export_engine  # noqa: E402

CFG = {"latent_dim": 8, "world_dim": 12, "proj_hidden": 16, "n_steps": 5, "dt_max": 0.5,
       "settle_eps": 1e-3, "dropout": 0.0, "feature_dropout": 0.0, "ssm_state": 4}


def _engine(cfg, seed=0):
    torch.manual_seed(seed)
    eng = VegaEngine(d_in=10, n_feat_layers=2, n_opt_layers=1, cfg=cfg).eval()
    eng.feat_mu.normal_()
    eng.feat_sd.uniform_(0.5, 2.0)
    return eng


def _inputs(B=4, K=5, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((B, 2, 3, 10), dtype=np.float32),
            rng.standard_normal((B, K, 1, 10), dtype=np.float32),
            np.ones((B, K), dtype=bool),
            np.array([QTYPES["choice"], QTYPES["boolean"], QTYPES["score"], QTYPES["choice"]])[:B])


def _torch_out(eng, feats, opts, mask, qtype, **kw):
    with torch.no_grad():
        return eng(torch.tensor(feats), torch.tensor(opts), torch.tensor(mask),
                   torch.tensor(qtype), halt=True, return_traj=False, **kw)


def test_exported_graph_matches_torch(tmp_path):
    eng = _engine(dict(CFG))
    path = export_engine(eng, str(tmp_path / "engine.onnx"))
    ort_eng = ONNXEngine(path)
    feats, opts, mask, qtype = _inputs()
    mask[0, -2:] = False
    ref = _torch_out(eng, feats, opts, mask, qtype)
    got = ort_eng(feats, opts, mask, qtype)
    np.testing.assert_array_equal(got["steps_used"], ref["steps_used"].numpy())
    for key in ("logits", "zT", "kinetic", "grad_norm", "nearest"):
        np.testing.assert_allclose(got[key], ref[key].numpy(), rtol=1e-4, atol=1e-5, err_msg=key)


def test_batch_and_option_axes_are_dynamic(tmp_path):
    eng = _engine(dict(CFG))
    path = export_engine(eng, str(tmp_path / "engine.onnx"))     # traced on B=2, K=3
    ort_eng = ONNXEngine(path)
    feats, opts, mask, qtype = _inputs(B=3, K=7, seed=2)
    ref = _torch_out(eng, feats, opts, mask, qtype)
    got = ort_eng(feats, opts, mask, qtype)
    np.testing.assert_allclose(got["logits"], ref["logits"].numpy(), rtol=1e-4, atol=1e-5)


def test_memory_checkpoint_exports_with_chunk_inputs(tmp_path):
    eng = _engine(dict(CFG, memory_chunks=4), seed=3)
    path = export_engine(eng, str(tmp_path / "engine.onnx"))
    ort_eng = ONNXEngine(path)
    feats, opts, mask, qtype = _inputs(seed=4)
    rng = np.random.default_rng(5)
    chunks = rng.standard_normal((4, 4, 10), dtype=np.float32)
    cmask = np.zeros((4, 4), dtype=bool)
    cmask[0] = True
    cmask[2, :2] = True
    ref = _torch_out(eng, feats, opts, mask, qtype,
                     chunks=torch.tensor(chunks), chunk_mask=torch.tensor(cmask))
    got = ort_eng(feats, opts, mask, qtype, chunks=chunks, chunk_mask=cmask)
    np.testing.assert_allclose(got["logits"], ref["logits"].numpy(), rtol=1e-4, atol=1e-5)
    # a batch with no chunks at all feeds zeros and must equal the chunkless torch path
    ref0 = _torch_out(eng, feats, opts, mask, qtype)
    got0 = ort_eng(feats, opts, mask, qtype)
    np.testing.assert_allclose(got0["logits"], ref0["logits"].numpy(), rtol=1e-4, atol=1e-5)


def test_export_from_a_checkpoint_directory(tmp_path):
    import json

    from safetensors.torch import save_file
    eng = _engine(dict(CFG))
    save_file(eng.state_dict(), str(tmp_path / "engine.safetensors"))
    cfg = dict(CFG, d_model=10, feature_layers=[0, 1], option_layers=[0])
    (tmp_path / "vega_config.json").write_text(json.dumps(cfg))
    path = export_engine(str(tmp_path), str(tmp_path / "engine.onnx"))
    feats, opts, mask, qtype = _inputs()
    ref = _torch_out(eng, feats, opts, mask, qtype)
    got = ONNXEngine(path)(feats, opts, mask, qtype)
    np.testing.assert_allclose(got["logits"], ref["logits"].numpy(), rtol=1e-4, atol=1e-5)
