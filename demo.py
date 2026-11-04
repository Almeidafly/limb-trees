"""All four limbs, one tree per button. Runs slowly on purpose (stand-in timings, not Grok internals)."""
from limb_trees.robot import Robot, press, LIMBS, TakeoverHook
r = Robot(); hook = TakeoverHook(); hook.record("right_arm", (10, -5, 3), seconds=0.02, load=1.0)
print(f"{'button':>6} {'limb':<10} {'source':<14} {'weight(load)':>12} {'time_s':>8} {'byte':>9} {'state':>5}")
for button in range(1, 5):
    for limb in LIMBS:
        b, _ = press(r, limb, button, slow=0.001, hook=hook)
        print(f"{button:>6} {limb:<10} {b['source']:<14} {b['load']:>12.3f} {b['time']:>8.4f} {b['byte']:>08b} {'lit' if b['lit'] else 'dark':>5}")
    for limb in LIMBS:
        r.rotate(limb)
print(f"tree switches: {len(r.switch_log)}, max switch time {max(s['seconds'] for s in r.switch_log)*1e6:.1f} us")
