from limb_trees import grok1
SMALL = [("a", 64, 32), ("b", 128, 16)]
def test_measure_small_shapes():
    route, ms = grok1.measure_grok1(slow=0, groups=SMALL)
    assert route in ("grok1-real-tensor+exact-shapes", "stand-in-fallback")
    assert all(t > 0 for _, t, _ in ms)
def test_config_shapes():
    assert grok1.D == 48 * 128 and grok1.KV == 8 * 128 and len(grok1.GROUPS) == 8
def test_fallback(monkeypatch):
    monkeypatch.setattr(grok1, "load_real_tensor", lambda: (_ for _ in ()).throw(OSError()))
    route, ms = grok1.measure_grok1(slow=0)
    assert route == "stand-in-fallback" and len(ms) == 8
