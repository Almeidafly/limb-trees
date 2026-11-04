from limb_trees.weight_time import TinyMLP, measure, to_moves
def test_moves():
    ms = measure(TinyMLP(), slow=0.0002)
    mv = to_moves(ms)
    assert len(mv) == 8 and all(m["duration_s"] > 0 for m in mv)
    assert abs(sum(m["effort"] for m in mv) - 1) < 1e-2
def test_slow_is_slower():
    a = sum(t for t, _ in measure(TinyMLP(), slow=0)); b = sum(t for t, _ in measure(TinyMLP(), slow=0.001))
    assert b > a
