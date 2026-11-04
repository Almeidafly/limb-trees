"""Humanoid with four limbs; each button press gets its own ternary byte tree.

Byte tree (from byte-tree): 1 root -> 3 (1 control + 2 data) -> 9 (1 control + 8 data bits = 1 byte).
Here the 8 data bits are 8 weight blocks of a stand-in model; the control block votes.
Signals: lit = pass, dark = fail.
"""
import copy, time
import numpy as np
from .weight_time import TinyMLP, measure

LIMBS = ("left_arm", "right_arm", "left_leg", "right_leg")
JOINTS = {"left_arm": ("shoulder", "elbow", "wrist"), "right_arm": ("shoulder", "elbow", "wrist"),
          "left_leg": ("hip", "knee", "ankle"), "right_leg": ("hip", "knee", "ankle")}


class ByteTree:
    """One tree per button press: control block + 8 data bits (one per weight block)."""
    def __init__(self, limb, press, measures, max_time, max_load):
        self.limb, self.press = limb, press
        self.bits = [{"time": t, "load": l, "lit": t <= max_time and l <= max_load} for t, l in measures]

    @property
    def byte(self):  # active-low like byte-tree: lit = 0
        return sum((0 if b["lit"] else 1) << i for i, b in enumerate(self.bits))

    @property
    def lit(self):  # control block: majority of 8 bits
        return sum(b["lit"] for b in self.bits) > 4


class Robot:
    def __init__(self, max_time=0.08, max_load=26.0):
        self.pose = {l: {j: 0.0 for j in JOINTS[l]} for l in LIMBS}
        self.trees = {l: [] for l in LIMBS}
        self.active = {l: 0 for l in LIMBS}
        self.switch_log = []  # timer on every tree switch
        self.max_time, self.max_load = max_time, max_load
        self._pressing = False

    # snapshot / rollback
    def snapshot(self):
        return copy.deepcopy(self.pose)

    def rollback(self, snap):
        self.pose = snap

    def apply(self, limb, move):
        for j, d in zip(JOINTS[limb], move["deltas"]):
            self.pose[limb][j] += d
        return all(abs(v) <= 90 for v in self.pose[limb].values())  # joint limit

    def rotate(self, limb):
        if self._pressing:
            raise RuntimeError("rotate only between presses")
        t0 = time.perf_counter()
        n = max(1, len(self.trees[limb]))
        self.active[limb] = (self.active[limb] + 1) % n
        self.switch_log.append({"limb": limb, "to": self.active[limb], "seconds": time.perf_counter() - t0})


def candidate(limb, seed, slow):
    """A candidate movement from one stand-in model's weight-time measurement."""
    ms = measure(TinyMLP(seed=seed), n_blocks=8, slow=slow, seed=seed)
    rng = np.random.default_rng(seed)
    deltas = list(rng.uniform(-60, 60, 3))
    return {"source": f"model-seed{seed}", "measures": ms, "deltas": deltas,
            "time": sum(t for t, _ in ms), "load": sum(l for _, l in ms)}


class TakeoverHook:
    """Screen-takeover hook: a human demo (recorded joint deltas + times) becomes a candidate."""
    def __init__(self):
        self.recordings = {}

    def record(self, limb, deltas, seconds, load=1.0):
        ms = [(seconds / 8, load / 8)] * 8
        self.recordings.setdefault(limb, []).append(
            {"source": "human-demo", "measures": ms, "deltas": list(deltas), "time": seconds, "load": load})

    def pop(self, limb):
        return self.recordings.get(limb, []).pop() if self.recordings.get(limb) else None


def press(robot, limb, button, n_candidates=3, slow=0.0005, hook=None):
    """Explore candidates for one press; test each in a snapshot; keep the best lit one."""
    robot._pressing = True
    try:
        cands = [candidate(limb, button * 100 + k, slow) for k in range(n_candidates)]
        if hook and (h := hook.pop(limb)):
            cands.append(h)
        results = []
        for c in cands:
            tree = ByteTree(limb, button, c["measures"], robot.max_time / 8 * 2, robot.max_load / 8 * 2)
            snap = robot.snapshot()
            ok = robot.apply(limb, c) and tree.lit
            results.append({**c, "lit": ok, "byte": tree.byte, "tree": tree})
            robot.rollback(snap)  # every test is rolled back
        lit = [r for r in results if r["lit"]]
        best = min(lit, key=lambda r: (r["time"] * r["load"])) if lit else \
            min(results, key=lambda r: r["time"] * r["load"])
        if best["lit"]:
            robot.apply(limb, best)
        robot.trees[limb].append(best["tree"])
        return best, results
    finally:
        robot._pressing = False
