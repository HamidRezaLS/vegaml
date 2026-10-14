"""MLX engine parity: same engine.safetensors, same inputs, the torch inference numbers to 1e-4."""
from __future__ import annotations

import json

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core", reason="pip install 'vegaml[mlx]' (Apple silicon)")
torch = pytest.importorskip("torch")

from safetensors.torch import save_file  # noqa: E402

from vegaml._common import QTYPES, VegaEngine  # noqa: E402
from vegaml.mlx_engine import MLXEngine  # noqa: E402

CFG = {"latent_dim": 8, "world_dim": 12, "proj_hidden": 16, "n_steps": 6, "dt_max": 0.5,
       "settle_eps": 1e-3, "dropout": 0.0, "feature_dropout": 0.0, "ssm_state": 4}


def _pair(cfg, seed=0):
    torch.manual_seed(seed)
    eng = VegaEngine(d_in=10, n_feat_layers=2, n_opt_layers=1, cfg=cfg).eval()
    # standardization buffers are fitted in training; random ones make the test exercise them
    eng.feat_mu.normal_()
    eng.feat_sd.uniform_(0.5, 2.0)
    eng.opt_mu.normal_()
    eng.opt_sd.uniform_(0.5, 2.0)
    state = {k: v.clone() for k, v in eng.state_dict().items()}
    mx_eng = MLXEngine(cfg, {k: v.numpy() for k, v in state.items()}, device="cpu")
    return eng, mx_eng


def _inputs(seed=1, B=4, K=5):
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((B, 2, 3, 10), dtype=np.float32)
    opts = rng.standard_normal((B, K, 1, 10), dtype=np.float32)
    mask = np.ones((B, K), dtype=bool)
    mask[0, -2:] = False                      # a padded question
    qtype = np.array([QTYPES["choice"], QTYPES["boolean"], QTYPES["score"], QTYPES["choice"]])
    return feats, opts, mask, qtype


@pytest.mark.parametrize("halt", [True, False])
def test_mlx_matches_torch_on_every_readout(halt):
    eng, mx_eng = _pair(dict(CFG))
    feats, opts, mask, qtype = _inputs()
    with torch.no_grad():
        ref = eng(torch.tensor(feats), torch.tensor(opts), torch.tensor(mask), torch.tensor(qtype),
                  halt=halt, return_traj=False)
    got = mx_eng(feats, opts, mask, qtype, halt=halt)
    np.testing.assert_array_equal(got["steps_used"], ref["steps_used"].numpy())
    for key in ("logits", "zT", "kinetic", "grad_norm", "nearest"):
        np.testing.assert_allclose(got[key], ref[key].numpy(), rtol=1e-4, atol=1e-4, err_msg=key)


def test_mlx_matches_torch_with_the_memory_reader():
    cfg = dict(CFG, memory_chunks=4)
    eng, mx_eng = _pair(cfg, seed=2)
    feats, opts, mask, qtype = _inputs(seed=3)
    rng = np.random.default_rng(4)
    chunks = rng.standard_normal((4, 4, 10), dtype=np.float32)
    cmask = np.zeros((4, 4), dtype=bool)
    cmask[0] = True
    cmask[1, :2] = True                       # records cut into different chunk counts
    with torch.no_grad():
        ref = eng(torch.tensor(feats), torch.tensor(opts), torch.tensor(mask), torch.tensor(qtype),
                  halt=True, return_traj=False, chunks=torch.tensor(chunks), chunk_mask=torch.tensor(cmask))
    got = mx_eng(feats, opts, mask, qtype, halt=True, chunks=chunks, chunk_mask=cmask)
    for key in ("logits", "zT", "nearest"):
        np.testing.assert_allclose(got[key], ref[key].numpy(), rtol=1e-4, atol=1e-4, err_msg=key)


def test_gpu_stream_reaches_the_same_decisions():
    """Metal matmul accumulates below float32, so values drift ~1e-2; the decision must not."""
    eng, cpu_eng = _pair(dict(CFG))
    state = {k: v.numpy() for k, v in eng.state_dict().items()}
    gpu_eng = MLXEngine(dict(CFG), state, device="gpu")
    feats, opts, mask, qtype = _inputs()
    ref = cpu_eng(feats, opts, mask, qtype)
    got = gpu_eng(feats, opts, mask, qtype)
    masked_ref = np.where(mask, ref["logits"], -np.inf)
    masked_got = np.where(mask, got["logits"], -np.inf)
    np.testing.assert_array_equal(masked_got.argmax(-1), masked_ref.argmax(-1))
    np.testing.assert_array_equal(got["steps_used"], ref["steps_used"])
    np.testing.assert_allclose(got["logits"][mask], ref["logits"][mask], rtol=5e-2, atol=5e-2)


def test_a_wrong_device_name_is_refused():
    eng, _ = _pair(dict(CFG))
    with pytest.raises(ValueError, match="gpu"):
        MLXEngine(dict(CFG), {k: v.numpy() for k, v in eng.state_dict().items()}, device="cuda")


def test_from_dir_reads_a_checkpoint_without_torch_tensors(tmp_path):
    eng, _ = _pair(dict(CFG))
    save_file(eng.state_dict(), str(tmp_path / "engine.safetensors"))
    cfg = dict(CFG, d_model=10, feature_layers=[0, 1], option_layers=[0])
    (tmp_path / "vega_config.json").write_text(json.dumps(cfg))
    mx_eng = MLXEngine.from_dir(str(tmp_path), device="cpu")
    feats, opts, mask, qtype = _inputs()
    with torch.no_grad():
        ref = eng(torch.tensor(feats), torch.tensor(opts), torch.tensor(mask), torch.tensor(qtype),
                  halt=True, return_traj=False)
    got = mx_eng(feats, opts, mask, qtype)
    np.testing.assert_allclose(got["logits"], ref["logits"].numpy(), rtol=1e-4, atol=1e-4)


def test_adapter_weights_in_the_file_are_ignored_not_fatal():
    eng, _ = _pair(dict(CFG))
    state = {k: v.numpy() for k, v in eng.state_dict().items()}
    state["gate_nets.0.weight"] = np.zeros((1, 24), dtype=np.float32)
    MLXEngine(dict(CFG), state)               # must construct; the base engine is the MLX scope
