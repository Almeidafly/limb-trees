"""Time compute on one Grok-1 layer (xai-org/grok-1, Apache-2.0).

Route: hybrid.
  * REAL weights: one small tensor from the published checkpoint
    (huggingface.co/xai-org/grok-1, ckpt-0/tensor00001_000, a float32 (6144,)
    vector, ~24 KB). It scales the input like a norm. The full 314B model is
    never downloaded.
  * EXACT SHAPES: the layer's 8 weight groups use Grok-1's published config
    (run.py: emb 48*128=6144, 48 q / 8 kv heads x 128, 8 experts,
    widening_factor 8 -> ffn 32768). Values are random because those shards
    are GBs.
Each group is one data bit of the byte tree. It is timed as a row-tiled
matvec with a deliberate sleep per tile, so the run is slow and observable.
Timings are from this box's CPU, not Grok's production hardware.
If the download fails, it falls back to the stand-in MLP (weight_time.py).
"""
import os, pickle, time, urllib.request
import numpy as np

URL = "https://huggingface.co/xai-org/grok-1/resolve/main/ckpt-0/tensor00001_000"
CACHE = os.path.join(os.path.dirname(__file__), "..", ".cache", "grok1_tensor00001_000")
D, KV, FFN, EXPERTS = 6144, 8 * 128, 32768, 8
GROUPS = [("attn_q", D, D), ("attn_k", D, KV), ("attn_v", D, KV), ("attn_o", D, D),
          ("router", D, EXPERTS), ("expert0_w_in", D, FFN), ("expert0_w_v", D, FFN), ("expert0_w_out", FFN, D)]


def load_real_tensor():
    path = os.path.abspath(CACHE)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL, path)

    class _NumpyOnly(pickle.Unpickler):
        def find_class(self, m, n):
            if m.startswith("numpy"):
                return super().find_class(m, n)
            raise pickle.UnpicklingError(f"refusing {m}.{n}")
    with open(path, "rb") as f:
        return np.asarray(_NumpyOnly(f).load(), dtype=np.float32)


def time_group(rows, cols, x, tile=1024, slow=0.0, seed=0):
    rng = np.random.default_rng(seed)
    total, xin = 0.0, x[:rows] if rows <= len(x) else np.resize(x, rows)
    for r0 in range(0, rows, tile):
        w = rng.standard_normal((min(tile, rows - r0), cols), dtype=np.float32) * 0.02
        t0 = time.perf_counter()
        _ = xin[r0:r0 + len(w)] @ w
        if slow:
            time.sleep(slow)
        total += time.perf_counter() - t0
    return total


def measure_grok1(slow=0.01, groups=GROUPS, seed=0):
    """Return (route, [(name, seconds, load_Mparams)]) for the 8 weight groups."""
    try:
        scale = load_real_tensor()
        assert scale.shape == (D,)
        route = "grok1-real-tensor+exact-shapes"
    except Exception:
        from .weight_time import TinyMLP, measure
        return "stand-in-fallback", [(f"block{i}", t, l) for i, (t, l) in enumerate(measure(TinyMLP(), slow=slow))]
    x = np.random.default_rng(seed).standard_normal(D).astype(np.float32) * scale
    return route, [(n, time_group(r, c, x, slow=slow, seed=seed + i), r * c / 1e6)
                   for i, (n, r, c) in enumerate(groups)]
