from limb_trees.weight_time import TinyMLP, measure, to_moves
ms = measure(TinyMLP(), slow=0.002)
print(f"{'block':>5} {'joint':<9} {'duration_s':>10} {'effort':>7}")
for i, m in enumerate(to_moves(ms)):
    print(f"{i:>5} {m['joint']:<9} {m['duration_s']:>10} {m['effort']:>7}")
