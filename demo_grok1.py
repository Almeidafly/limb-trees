"""Grok-1 layer weight time -> one press per limb. Slow on purpose."""
from limb_trees.grok1 import measure_grok1
from limb_trees.robot import ByteTree, JOINTS, LIMBS
route, ms = measure_grok1(slow=0.01)
print("route:", route)
print(f"{'group':<14} {'Mparams':>8} {'time_s':>8}")
for n, t, l in ms: print(f"{n:<14} {l:>8.2f} {t:>8.4f}")
tmed = sorted(t for _, t, _ in ms)[4]; lmed = sorted(l for *_, l in ms)[4]
print(f"\n{'button':>6} {'limb':<10} {'joint move (duration_s = group time)':<52} {'byte':>8} {'state':>5}")
for b, limb in enumerate(LIMBS, 1):
    tree = ByteTree(limb, b, [(t, l) for _, t, l in ms], tmed * (0.8 + 0.2 * b), lmed * (0.5 + 0.5 * b))
    j = JOINTS[limb]; moves = ", ".join(f"{j[i % 3]}:{ms[i][1]:.3f}" for i in range(3))
    print(f"{b:>6} {limb:<10} {moves:<52} {tree.byte:>08b} {'lit' if tree.lit else 'dark':>5}")
