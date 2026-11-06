#!/usr/bin/env python3
"""Build Ripple-65B-Preview NVFP4 + FP8-attention (ModelOpt MIXED_PRECISION).

MLP tensors, norms, embeddings and lm_head are copied byte-for-byte from the official
ricursiveAI/Ripple-65B-Preview-NVFP4 checkpoint. The 400 BF16 attention projections
(q, k, v, o, g x 80 layers) become FP8 E4M3 with one static per-tensor weight scale and a
calibrated static per-tensor input scale.

BertholomusAI, 2026-11-06.
"""
import argparse, hashlib, json, os, re, shutil, struct, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = os.environ.get("RIPPLE_NVFP4_DIR", "Ripple-65B-Preview-NVFP4")  # local snapshot of ricursiveAI/Ripple-65B-Preview-NVFP4@99eea4477a001a41b86914e41feb4b2ed86105e5
ATTN = re.compile(r"^model\.layers\.(\d+)\.self_attn\.([qkvog])_proj\.weight$")
COPY = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
        "generation_config.json", "configuration_ricursive.py", "modeling_ricursive.py", "LICENSE", "inference"]
PRODUCER = {"name": "bertholomus-fp8attn", "version": "0.1"}
FP8_MAX = 448.0
DSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1, "I32": 4, "I64": 8}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def raw_sha(path, name, hdr=None):
    n, h = hdr or header(path)
    a, b = h[name]["data_offsets"]
    m = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(8 + n + a)
        left = b - a
        while left:
            chunk = f.read(min(left, 1 << 26))
            m.update(chunk)
            left -= len(chunk)
    return m.hexdigest()


def quantized_layers(n_layers):
    q = {}
    for i in range(n_layers):
        for p in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
            q[f"model.layers.{i}.mlp.{p}"] = {"quant_algo": "NVFP4", "group_size": 16}
        for p in "qkvog":
            q[f"model.layers.{i}.self_attn.{p}_proj"] = {"quant_algo": "FP8"}
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-amax")
    ap.add_argument("--default-amax", type=float, default=64.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="keep finished output shards (recomputes their report rows)")
    a = ap.parse_args()

    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    amax = json.load(open(a.input_amax)) if a.input_amax else {}
    shards = sorted(set(idx.values()))
    attn = [t for t in idx if ATTN.match(t)]
    assert len(attn) == 5 * cfg["num_hidden_layers"], len(attn)
    print(f"src {a.src}: {len(idx)} tensors in {len(shards)} shards, {len(attn)} attention weights; "
          f"calibrated amax for {sum(1 for t in attn if t[:-7] in amax)}")
    if a.dry_run:
        return
    if os.path.exists(a.out) and os.listdir(a.out) and not a.resume:
        sys.exit(f"--out {a.out} exists and is not empty")
    for junk in [f for f in (os.listdir(a.out) if os.path.exists(a.out) else []) if f.startswith(".tmp")]:
        os.remove(os.path.join(a.out, junk))
    os.makedirs(a.out, exist_ok=True)

    report = {"tensors": {}, "bytes_in": 0, "bytes_out": 0, "attn_bytes_in": 0, "attn_bytes_out": 0}
    wmap = {}
    for shard in shards:
        if a.resume and os.path.exists(os.path.join(a.out, shard)):
            with safe_open(os.path.join(a.out, shard), "pt", device="cpu") as g, \
                    safe_open(os.path.join(a.src, shard), "pt", device="cpu") as f:
                for name in g.keys():
                    wmap[name] = shard
                    t = g.get_tensor(name)
                    report["bytes_out"] += t.numel() * t.element_size()
                    if ATTN.match(name):
                        mod = name[: -len(".weight")]
                        w = f.get_tensor(name).float()
                        ws = g.get_tensor(mod + ".weight_scale").float()
                        rel = (torch.linalg.vector_norm(w - t.float() * ws) / torch.linalg.vector_norm(w)).item()
                        report["attn_bytes_in"] += w.numel() * 2
                        report["attn_bytes_out"] += t.numel() + 8
                        report["tensors"][mod] = {"rel_err": rel, "weight_scale": ws.item(),
                                                  "input_scale": g.get_tensor(mod + ".input_scale").item(),
                                                  "amax_source": "calibrated" if mod in amax else "default"}
                        del w
                    del t
                for name in f.keys():
                    t = f.get_tensor(name); report["bytes_in"] += t.numel() * t.element_size(); del t
            print(f"kept {shard} (resume)", flush=True)
            continue
        out = {}
        with safe_open(os.path.join(a.src, shard), "pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                report["bytes_in"] += t.numel() * t.element_size()
                m = ATTN.match(name)
                if not m:
                    out[name] = t
                    continue
                mod = name[: -len(".weight")]
                w = t.to("cuda", torch.float32) if torch.cuda.is_available() else t.float()
                ws = torch.clamp(w.abs().max(), min=1e-12) / FP8_MAX
                w8 = (w / ws).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                rel = (torch.linalg.vector_norm(w - w8.float() * ws) / torch.linalg.vector_norm(w)).item()
                src_amax = amax.get(mod)
                ins = max(float(src_amax if src_amax is not None else a.default_amax), 1e-6) / FP8_MAX
                out[name] = w8.cpu()
                out[mod + ".weight_scale"] = ws.reshape(()).float().cpu()
                out[mod + ".input_scale"] = torch.tensor(ins, dtype=torch.float32)
                report["attn_bytes_in"] += t.numel() * 2
                report["attn_bytes_out"] += t.numel() + 8
                report["tensors"][mod] = {"rel_err": rel, "weight_scale": ws.item(), "input_scale": ins,
                                          "amax_source": "calibrated" if src_amax is not None else "default"}
                del w, w8
        save_file(out, os.path.join(a.out, shard), metadata={"format": "pt"})
        for k, v in out.items():
            wmap[k] = shard
            report["bytes_out"] += v.numel() * v.element_size()
        print(f"wrote {shard}: {len(out)} tensors", flush=True)
        del out

    json.dump({"metadata": {"total_size": report["bytes_out"]}, "weight_map": dict(sorted(wmap.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    ql = quantized_layers(cfg["num_hidden_layers"])
    excl = ["lm_head", "model.embed_tokens"]
    cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "producer": PRODUCER,
                                  "kv_cache_quant_algo": None, "exclude_modules": excl, "quantized_layers": ql}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    json.dump({"producer": PRODUCER, "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None,
                                                      "exclude_modules": excl, "quantized_layers": ql}},
              open(os.path.join(a.out, "hf_quant_config.json"), "w"), indent=2)
    for c in COPY:
        s = os.path.join(a.src, c)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(a.out, c))
        elif os.path.exists(s):
            shutil.copy2(s, os.path.join(a.out, c))

    # ---------------------------------------------------------------- self-check
    ok = True
    src_names = set(idx)
    out_names = set(wmap)
    want = src_names | {t[:-7] + s for t in attn for s in (".weight_scale", ".input_scale")}
    if out_names != want:
        ok = False
        print("FAIL tensor set", len(out_names ^ want))
    else:
        print(f"PASS tensor set: {len(out_names)} = {len(src_names)} + {2 * len(attn)}")
    hs = {s: header(os.path.join(a.src, s)) for s in shards}
    ho = {s: header(os.path.join(a.out, s)) for s in shards}
    bad = 0
    for name in sorted(src_names):
        if ATTN.match(name):
            continue
        if raw_sha(os.path.join(a.src, idx[name]), name, hs[idx[name]]) != raw_sha(os.path.join(a.out, wmap[name]), name, ho[wmap[name]]):
            bad += 1
    print(("PASS" if not bad else "FAIL") + f" non-attention tensors byte-identical ({len(src_names) - len(attn)} checked, {bad} differ)")
    ok &= not bad
    worst = max(v["rel_err"] for v in report["tensors"].values())
    dt_bad = [t for t in attn if ho[wmap[t]][1][t]["dtype"] != "F8_E4M3"
              or ho[wmap[t]][1][t]["shape"] != hs[idx[t]][1][t]["shape"]]
    print(("PASS" if not dt_bad and worst <= 0.08 else "FAIL") + f" attention FP8 dtype/shape ({len(dt_bad)} bad), max rel_err {worst:.4f}")
    ok &= not dt_bad and worst <= 0.08
    total = 0
    for s, (n, h) in ho.items():
        for k, v in h.items():
            if k != "__metadata__":
                a0, b0 = v["data_offsets"]
                total += b0 - a0
    print(("PASS" if total == report["bytes_out"] else "FAIL") + f" index total_size {report['bytes_out']} vs shards {total}")
    ok &= total == report["bytes_out"]
    report["max_rel_err"] = worst
    json.dump(report, open(os.path.join(a.out, "fp8attn_report.json"), "w"), indent=1)
    print(f"bytes in {report['bytes_in']/1e9:.2f} GB -> out {report['bytes_out']/1e9:.2f} GB; attention "
          f"{report['attn_bytes_in']/1e9:.2f} -> {report['attn_bytes_out']/1e9:.2f} GB")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
