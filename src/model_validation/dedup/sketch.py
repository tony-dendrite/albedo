from __future__ import annotations

import base64
import hashlib
import math
import mmap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from model_validation.dedup.canon import Provider, canonicalize, residual_perm
from model_validation.dedup.layout import EMBED, WS_VERSION, arch_key, read_header, ttype
from model_validation.dedup.secret import key_id, seed_for

K = 64
NSAMP = 2048
VEC_DIM = 1024
MIN_DIM = 8


def _gen(secret: bytes, name: str, which: str, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed_for(secret, name, which))


def sketch(w: torch.Tensor, name: str, secret: bytes) -> tuple[np.ndarray, float, np.ndarray, int]:
    dev = w.device
    if w.dim() == 3:
        w = w.reshape(-1, w.shape[-1])
    m, n = w.shape
    k = min(K, m, n)
    psi = torch.randn(k, m, generator=_gen(secret, name, "psi", dev), device=dev) / math.sqrt(m)
    omega = torch.randn(n, k, generator=_gen(secret, name, "omega", dev), device=dev) / math.sqrt(n)
    s = psi @ (w @ omega)
    idx = torch.randint(
        0, w.numel(), (NSAMP,), generator=_gen(secret, name, "samp", dev), device=dev
    )
    x = w.reshape(-1)[idx]
    return (
        s.cpu().numpy().astype(np.float32),
        float(w.norm()),
        x.cpu().numpy().astype(np.float32),
        k,
    )


def vec_project(s: torch.Tensor, name: str, secret: bytes) -> torch.Tensor:
    flat = s.reshape(-1)
    proj = torch.randn(
        VEC_DIM, flat.numel(), generator=_gen(secret, name, "vec", s.device), device=s.device
    )
    return proj @ flat


def tensors_hash(model_dir: str) -> str:
    digests: list[str] = []
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        header = read_header(shard)
        with shard.open("rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                start = 8 + int.from_bytes(mm[:8], "little")
                for name, info in header.items():
                    a, b = info["data_offsets"]
                    h = hashlib.sha256(f"{name}|{info['dtype']}|{info['shape']}|".encode())
                    h.update(mm[start + a : start + b])
                    digests.append(h.hexdigest())
            finally:
                mm.close()
    return hashlib.sha256("\n".join(sorted(digests)).encode()).hexdigest()


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def fingerprint(
    model_dir: str,
    ref_dir: str,
    secret: bytes,
    device: torch.device,
    *,
    model_uri: str = "",
    text_set_sha: str = "",
) -> dict:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        hash_job = pool.submit(tensors_hash, model_dir)
        prov = Provider(model_dir, device)
        ref = Provider(ref_dir, device)
        perm_res, ident_res = residual_perm(prov.get(EMBED), ref.get(EMBED))
        tensors, idents, shapes = [], {}, {}
        vec = torch.zeros(VEC_DIM, dtype=torch.float32, device=device)
        total = 0
        for c in prov.keys():
            shape = prov.shape(c)
            shapes[c] = shape
            if len(shape) < 2 or min(shape[-2], shape[-1]) < MIN_DIM:
                continue
            w = prov.get(c)
            wb = ref.get(c)
            w, info = canonicalize(c, w, wb, perm_res)
            del wb
            s, wn, x, k = sketch(w, c, secret)
            del w
            st = torch.from_numpy(s).to(device)
            vec += vec_project(st, c, secret)
            total += st.numel()
            tensors.append(
                dict(name=c, shape=list(shape), k=k, wnorm=wn, s=_b64(s), x=_b64(x), **info)
            )
            if info:
                idents.setdefault(ttype(c), []).append(info["ident"])
        identity = {k: round(float(np.mean(v)), 3) for k, v in idents.items()}
        identity["residual"] = round(ident_res, 3)
        thash = hash_job.result()
    return dict(
        model_uri=model_uri,
        ws_version=WS_VERSION,
        key_id=key_id(secret, text_set_sha),
        arch_key=arch_key(shapes),
        tensors_hash=thash,
        n_tensors=len(tensors),
        identity_frac=identity,
        sketch_vec=(vec / math.sqrt(max(total, 1))).cpu().numpy().astype(np.float32).tolist(),
        secs=round(time.time() - t0, 1),
        tensors=tensors,
    )
