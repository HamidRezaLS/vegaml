"""vega-U (universal adapter) inference: one shared engine behind a per-backbone port.

A universal checkpoint directory is a normal vega model dir plus two files: this backbone's
port (``port.safetensors``) and the raw feature statistics of its extraction (``stats.npz``).
Its ``vega_config.json`` names both and carries ``d_shared``, the width of the shared space
the engine lives in. Raw perception features are standardized with those statistics, mapped
through the port into the shared space, and the engine runs there, byte-identical across
every backbone. Switching base model is switching which per-backbone dir you load.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


class Port(nn.Module):
    """One linear map per tapped feature layer, backbone width into shared width. Option
    features and state chunk summaries are pooled at the deepest feature layer, so they go
    through the last layer's map."""

    def __init__(self, d_in: int, d_out: int, n_feat_layers: int):
        super().__init__()
        self.lin = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n_feat_layers)])

    def items(self, f):           # [B, Lf, 3, d_in] -> [B, Lf, 3, d_out]
        return torch.stack([self.lin[i](f[:, i]) for i in range(len(self.lin))], 1)

    def opts(self, o):            # [..., Lo, d_in] -> [..., Lo, d_out]
        return self.lin[-1](o)

    def chunks(self, c):          # [..., d_in] -> [..., d_out]
        return self.lin[-1](c)


class UniversalEngine(nn.Module):
    """The shared engine behind one backbone's port, on RAW perception features: standardize
    with the backbone's extraction statistics, port into the shared space, then run the engine
    with ``prestandardized=True``. Reads of anything the engine owns (``max_steps``,
    ``memory``, ``adapter_mode``, ...) fall through to it, so the caller treats this exactly
    like a :class:`~vegaml._common.VegaEngine`."""

    def __init__(self, model_dir: str, cfg: dict, engine: nn.Module):
        super().__init__()
        from safetensors.torch import load_file

        # allow_pickle=False is numpy's default, spelled out because the file comes from a
        # downloaded checkpoint: plain arrays only, never unpickled objects.
        z = np.load(os.path.join(model_dir, cfg.get("raw_stats", "stats.npz")), allow_pickle=False)
        self.register_buffer("mu", torch.tensor(z["mu"], dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(z["sd"], dtype=torch.float32))
        self.register_buffer("omu", torch.tensor(z["omu"], dtype=torch.float32))
        self.register_buffer("osd", torch.tensor(z["osd"], dtype=torch.float32))
        self.port = Port(int(self.mu.shape[-1]), int(cfg["d_shared"]), int(self.mu.shape[0]))
        self.port.load_state_dict(load_file(os.path.join(model_dir, cfg["port"])))
        self.engine = engine

    def forward(self, feats, opts, opt_mask, qtype, prestandardized=False, chunks=None,
                chunk_mask=None, **kw):
        f = self.port.items((feats.float() - self.mu) / self.sd)
        o = self.port.opts((opts.float() - self.omu) / self.osd)
        c = None if chunks is None else self.port.chunks(
            (chunks.float() - self.mu[-1, 0]) / self.sd[-1, 0])
        return self.engine(f, o, opt_mask, qtype, prestandardized=True, chunks=c,
                           chunk_mask=chunk_mask, **kw)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("engine"), name)
