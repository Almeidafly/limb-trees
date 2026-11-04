# limb-trees

**Weight time → robot.** Weight time is the time a model (Grok) spends on each weight. We can't see Grok's real weights, so this repo uses **stand-in timings** from a tiny local numpy MLP, **not Grok's internals**. Each weight block is computed one weight at a time with a deliberate sleep, so it runs very slowly and the time is easy to observe. Each block's time becomes a robot joint move: duration = measured time, effort = the block's share of the weight load.

v0.1.0: one limb (shoulder/elbow/wrist).

    python3 demo_v01.py
    python3 -m pytest -q
