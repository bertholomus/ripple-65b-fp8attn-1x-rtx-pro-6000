"""Calibration hook for Ripple attention projections (BertholomusAI 2026-11-06).

Import it in the server process before launch (with RIPPLE_CALIB_OUT set) to record calibration maxima. It wraps RicursiveAttention.forward and keeps
a running max |x| of the input to each projection (q/k/g share hidden_states; v gets value_input; o gets the
gated attention output) in GPU buffers. Every RIPPLE_CALIB_EVERY layer-0 calls it writes the JSON
{"model.layers.N.self_attn.P_proj": amax, ...} plus a call counter to RIPPLE_CALIB_OUT.
Run with RIPPLE_SPEC=0 and --disable-cuda-graph so every forward runs eagerly.
"""
import json
import os

import torch

OUT = os.environ["RIPPLE_CALIB_OUT"]
EVERY = int(os.environ.get("RIPPLE_CALIB_EVERY", "50"))


def install():
    import model as M

    cls = M.RicursiveAttention
    orig = cls.forward
    state = {"bufs": {}, "calls": 0, "tokens": 0}

    def upd(name, x):
        b = state["bufs"].get(name)
        v = x.detach().abs().amax().float()
        if b is None:
            state["bufs"][name] = v.clone()
        else:
            torch.maximum(b, v, out=b)

    def dump():
        d = {k: float(v.item()) for k, v in sorted(state["bufs"].items())}
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"calls": state["calls"], "tokens": state["tokens"], "amax": d}, f, indent=1)
        os.replace(tmp, OUT)

    def forward(self, positions, hidden_states, forward_batch, value_residual=None):
        pre = f"model.layers.{self.attn.layer_id}.self_attn."
        value_input = hidden_states
        if value_residual is not None and self.use_valres:
            blend = torch.sigmoid(self.value_res_lambda[0]).to(dtype=hidden_states.dtype)
            value_input = torch.lerp(value_residual.to(dtype=hidden_states.dtype), hidden_states, blend)
        upd(pre + "q_proj", hidden_states)
        upd(pre + "k_proj", hidden_states)
        upd(pre + "g_proj", hidden_states)
        upd(pre + "v_proj", value_input)
        o_in = {}
        o_proj = self.o_proj
        real_o = o_proj.forward

        def o_hook(x, *a, **k):
            upd(pre + "o_proj", x)
            return real_o(x, *a, **k)

        o_proj.forward = o_hook
        try:
            out = orig(self, positions, hidden_states, forward_batch, value_residual)
        finally:
            o_proj.forward = real_o
        if self.attn.layer_id == 0:
            state["calls"] += 1
            state["tokens"] += int(hidden_states.shape[0])
            if state["calls"] % EVERY == 0:
                dump()
        return out

    cls.forward = forward
    print(f"[calib] RicursiveAttention.forward wrapped; writing {OUT} every {EVERY} layer-0 calls", flush=True)
