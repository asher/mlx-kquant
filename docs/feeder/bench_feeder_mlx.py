"""Feeder-loop component benchmarks through the kq primitives + MLX graphs.

A: per-token cost of 62 encoded signal/wait handoff pairs (feeder thread
   answering instantly) vs the same graph without them.
B: fused kq gather rate from a resident synthetic q5_k arena at MiniMax
   decode geometry (62 layers x top-8, d_model 3072, d_ff 1536).
"""
import threading
import time

import mlx.core as mx
from mlx_kquant import _ext as kq

NLAYERS = 62
TOKENS = 30
D = 3072

# ---------- A: handoff overhead ----------
r_done = kq.shared_event_create()
w_ready = kq.shared_event_create()
stop = False


def feeder():
    n = 0
    while not stop:
        n += 1
        if not kq.shared_event_wait(r_done, n, 2000):
            break
        kq.shared_event_set(w_ready, n)


def token_graph(x, w, gated=True, base=0):
    with mx.stream(mx.gpu):
        for li in range(NLAYERS):
            x = x @ w  # stand-in spine+router work per layer
            if gated:
                n = base + li + 1
                x = kq.event_signal(x, r_done, n)
                x = kq.event_wait(x, w_ready, n)
    return x


w = mx.eye(D)
x0 = mx.ones((1, D))
mx.eval(w, x0)

# ungated baseline
t0 = time.perf_counter()
for _ in range(TOKENS):
    mx.eval(token_graph(x0, w, gated=False))
base_s = (time.perf_counter() - t0) / TOKENS

# gated
t = threading.Thread(target=feeder)
t.start()
t0 = time.perf_counter()
for tok in range(TOKENS):
    mx.eval(token_graph(x0, w, gated=True, base=tok * NLAYERS))
gated_s = (time.perf_counter() - t0) / TOKENS
stop = True
kq.shared_event_set(r_done, 2**64 - 2)  # let the feeder loop exit
t.join()

print(f"A: ungated {base_s * 1e3:.2f} ms/token, gated {gated_s * 1e3:.2f} "
      f"ms/token -> {(gated_s - base_s) * 1e3 / NLAYERS * 1000:.0f} us per "
      f"handoff pair ({NLAYERS} pairs/token)")

# ---------- B: resident-arena fused gather rate ----------
import numpy as np

N_EXP = 64          # arena slots per layer (memory-bound sample; rate/expert is what matters)
D_FF = 1536
Q5K_BPB = 176       # q5_k bytes per 256-weight block
Q6K_BPB = 210
gate_up_bytes = D * Q5K_BPB // 256   # per row
rng = np.random.default_rng(0)
gate_w = mx.array(rng.integers(0, 255, (N_EXP, D_FF, gate_up_bytes), dtype=np.uint8))
up_w = mx.array(rng.integers(0, 255, (N_EXP, D_FF, gate_up_bytes), dtype=np.uint8))
xb = mx.array(rng.standard_normal((1, D)).astype(np.float16))
ids = mx.array(rng.integers(0, N_EXP, (1, 8), dtype=np.uint32))
mx.eval(gate_w, up_w, xb, ids)

with mx.stream(mx.gpu):
    out = kq.moe_glu_gather_kq(xb, gate_w, up_w, "q5_k", ids)
mx.eval(out)  # warm
REPS = 200
t0 = time.perf_counter()
with mx.stream(mx.gpu):
    outs = out
    for _ in range(REPS):
        outs = kq.moe_glu_gather_kq(xb, gate_w, up_w, "q5_k", ids)
        mx.eval(outs)
dt = (time.perf_counter() - t0) / REPS
bytes_per_call = 8 * 2 * D_FF * gate_up_bytes
print(f"B: fused q5_k gather (top-8 gate+up) {dt * 1e6:.0f} us/call, "
      f"{bytes_per_call / dt / 1e9:.1f} GB/s effective; "
      f"x{NLAYERS} layers = {dt * NLAYERS * 1e3:.1f} ms/token gate+up share")
