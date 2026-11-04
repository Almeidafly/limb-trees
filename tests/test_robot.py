import pytest
from limb_trees.robot import Robot, press, LIMBS, TakeoverHook, ByteTree
def test_each_press_own_tree_and_rotation():
    r = Robot()
    for b in (1, 2):
        for l in LIMBS: press(r, l, b, slow=0)
        for l in LIMBS: r.rotate(l)
    assert all(len(r.trees[l]) == 2 for l in LIMBS) and len(r.switch_log) == 8
    assert all("seconds" in s for s in r.switch_log)
def test_rollback_on_dark():
    r = Robot(); r.pose["left_leg"]["hip"] = 89.0; before = r.snapshot()
    hook = TakeoverHook(); hook.record("left_leg", (50, 0, 0), 0.0001)
    best, res = press(r, "left_leg", 1, n_candidates=0, slow=0, hook=hook)
    assert not best["lit"] and r.pose == before
def test_takeover_candidate_wins():
    r = Robot(); hook = TakeoverHook(); hook.record("right_arm", (1, 1, 1), 0.0001, 0.01)
    best, _ = press(r, "right_arm", 1, slow=0, hook=hook)
    assert best["source"] == "human-demo" and best["lit"]
def test_no_rotate_during_press():
    r = Robot(); r._pressing = True
    with pytest.raises(RuntimeError): r.rotate("left_arm")
def test_byte_tree_majority():
    t = ByteTree("left_arm", 1, [(0.0, 0.0)] * 5 + [(9, 9)] * 3, 1, 1)
    assert t.lit and t.byte == 0b11100000
