"""Stand-in weight-time measurement.

We cannot see Grok's weights. This times a tiny local numpy MLP, one weight
block at a time, deliberately slowed (per-weight Python loop + sleep) so the
time is observable. These are STAND-IN timings, not Grok internals.
"""
import time
import numpy as np


class TinyMLP:
    def __init__(self, sizes=(8, 8, 8), seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, 0.5, (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]

    def blocks(self, n_blocks=8):
        """Split all weights into n_blocks flat blocks."""
        flat = np.concatenate([w.ravel() for w in self.W])
        return np.array_split(flat, n_blocks)


def time_block(block, x, slow=0.0005):
    """Multiply-accumulate each weight one at a time, sleeping per weight."""
    t0 = time.perf_counter()
    acc = 0.0
    for i, w in enumerate(block):
        acc += float(w) * float(x[i % len(x)])
        if slow:
            time.sleep(slow)
    return time.perf_counter() - t0, float(np.abs(block).sum())


def measure(model, n_blocks=8, slow=0.0005, seed=1):
    """Return list of (seconds, load) per weight block."""
    x = np.random.default_rng(seed).normal(size=8)
    return [time_block(b, x, slow) for b in model.blocks(n_blocks)]


def to_moves(measures, joints=("shoulder", "elbow", "wrist")):
    """Turn each block's time into a joint move: duration = time, effort = load."""
    tot = sum(m[1] for m in measures) or 1.0
    return [{"joint": joints[i % len(joints)], "duration_s": round(t, 4),
             "effort": round(load / tot, 4)} for i, (t, load) in enumerate(measures)]
