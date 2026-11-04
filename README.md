# limb-trees

**Weight time → robot.** Weight time is the time a model (Grok) spends on each weight. We can't see Grok's real weights, so this repo uses **stand-in timings** from a tiny local numpy MLP, **not Grok's internals**. Each weight block is computed one weight at a time with a deliberate sleep, so it runs very slowly and the time is easy to observe. Each block's time becomes a robot joint move: duration = measured time, effort = the block's share of the weight load.

v0.1.0: one limb (shoulder/elbow/wrist).

    python3 demo_v01.py
    python3 -m pytest -q

## v0.2.0: humanoid, four limbs
- Limbs: `left_arm`, `right_arm`, `left_leg`, `right_leg`.
- **Every button press gets its own ternary byte tree** (1 → 3 → 9: one control block plus 8 data bits = 1 byte; the design comes from byte-tree). Each data bit is one weight block's (time, load). A bit is lit if both stay under the per-block budget. The control block is lit when a majority (more than 4 of 8) of the bits are lit. Bytes print active-low (lit = 0).
- **Candidates per press:** 3 stand-in models (different seeds) plus any human demo. Each candidate is scored by weight (load) and time and tested inside a **snapshot**: lit = pass, dark = fail. Every test is **rolled back**. The best lit candidate (lowest time × load) is then applied. A dark result leaves the pose unchanged.
- **Rotation:** `Robot.rotate(limb)` moves to the next active tree. It is only allowed between presses and raises an error during one. **Every switch is timed** (`robot.switch_log`).
- **Screen-takeover hook:** `TakeoverHook.record(limb, deltas, seconds, load)` turns a recorded human demo into a candidate for that limb's next press.
- Note: candidate seeds depend on the button, so all four limbs see the same model candidates for a press. They only differ when joint limits or a human demo come into play.

    python3 demo.py
