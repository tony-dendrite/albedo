from __future__ import annotations

import numpy as np
import torch
from safetensors import safe_open

from model_validation.dedup.layout import COL_HIDDEN, RES_ROWS, ROW_HIDDEN, tensor_names


class Provider:
    def __init__(self, model_dir: str, device: torch.device):
        self.dir = model_dir
        self.device = device
        self.names = tensor_names(model_dir)
        self._files: dict[str, object] = {}

    def keys(self) -> list[str]:
        return sorted(self.names)

    def shape(self, c: str) -> tuple[int, ...]:
        key, fname = self.names[c]
        return tuple(self._file(fname).get_slice(key).get_shape())

    def get(self, c: str) -> torch.Tensor:
        key, fname = self.names[c]
        w = self._file(fname).get_tensor(key)
        if w.dim() < 2:
            return w
        return w.to(self.device, torch.float32)

    def _file(self, fname: str):
        if fname not in self._files:
            self._files[fname] = safe_open(f"{self.dir}/{fname}", framework="pt", device="cpu")
        return self._files[fname]


def match(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    an = torch.nn.functional.normalize(a, dim=1)
    bn = torch.nn.functional.normalize(b, dim=1)
    cs = bn @ an.T
    best, idx = cs.max(dim=1)
    if len(set(idx.tolist())) != idx.numel():
        idx = idx.clone()
        used = torch.zeros(a.shape[0], dtype=torch.bool, device=a.device)
        for j in torch.argsort(best, descending=True).tolist():
            i = int(idx[j])
            if used[i]:
                row = cs[j].clone()
                row[used] = -2
                i = int(row.argmax())
                idx[j] = i
            used[i] = True
    ident = float((idx == torch.arange(idx.numel(), device=idx.device)).float().mean())
    return idx, float(best.mean()), ident


def residual_perm(embed: torch.Tensor, ref_embed: torch.Tensor) -> tuple[torch.Tensor, float]:
    idx, _, ident = match(embed.T, ref_embed.T)
    return idx, ident


def canonicalize(
    c: str, w: torch.Tensor, wb: torch.Tensor, perm_res: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    info: dict = {}
    if w.dim() == 3:
        down = c.endswith("experts.down_proj")
        w = w[:, perm_res, :] if down else w[..., perm_res]
        idents, coss = [], []
        for e in range(w.shape[0]):
            if down:
                idx, cos, ident = match(w[e].T, wb[e].T)
                w[e] = w[e][:, idx]
            else:
                idx, cos, ident = match(w[e], wb[e])
                w[e] = w[e][idx]
            idents.append(ident)
            coss.append(cos)
        return w, dict(ident=float(np.mean(idents)), cos=float(np.mean(coss)))
    w = w[perm_res, :] if any(t in c for t in RES_ROWS) else w[:, perm_res]
    if any(t in c for t in ROW_HIDDEN):
        idx, cos, ident = match(w, wb)
        w = w[idx]
        info = dict(ident=ident, cos=cos)
    elif any(t in c for t in COL_HIDDEN):
        idx, cos, ident = match(w.T, wb.T)
        w = w[:, idx]
        info = dict(ident=ident, cos=cos)
    return w, info
