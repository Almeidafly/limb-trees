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

## v0.3.0: timing one Grok-1 layer
Route used: **hybrid**. The code downloads one real Grok-1 tensor from the published checkpoint (`ckpt-0/tensor00001_000`, a float32 vector of 6144 values, about 24 KB, unpickled with a numpy-only loader) and uses it to scale the input. Grok-1 is about 314B parameters, so the model is **never downloaded in full**. The 8 weight groups of one layer (q, k, v, o, router, expert w_in, w_v, w_out) are **exact Grok-1 shapes** from the published config (d=6144, 48 q heads and 8 kv heads × 128, 8 experts, FFN 32768). Their values are random, because those shards are several GB each. Each group is one data bit of a limb's byte tree, timed as a slow, row-tiled matvec on this CPU. The group times become joint move durations. If the download fails, the code falls back to the stand-in MLP.

These timings come from this box, not from Grok's production hardware, and reflect neither Grok's inference stack nor its trained values for the large matrices.

    python3 demo_grok1.py

### Credit
Grok-1 by xAI, https://github.com/xai-org/grok-1, weights at https://huggingface.co/xai-org/grok-1, released under the Apache License 2.0. This repo has no license of its own and does not redistribute any weights. The tensor is fetched at runtime into `.cache/`, which is gitignored.
