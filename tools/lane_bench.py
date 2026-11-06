#!/usr/bin/env python3
"""Frozen speed + exactness rows for one OpenAI-compatible lane (T0, no model judgement).

Rows (same prompts every time, temperature 0):
  decode_prose_c1 / decode_code_c1   single stream, 1024 new tokens, tok/s after first token
  agg_c4                             4 streams of the code prompt at once, total tok/s
  prefill_8k / prefill_32k           time to first token for a cold ~8K / ~32K prompt -> prompt tok/s
  greedy_repeat                      the code prompt twice at temperature 0: identical text? (exactness)
Usage: lane_bench.py --url http://127.0.0.1:30000/v1 --model Ripple-65B-Preview-NVFP4-FP8Attn --out rows.json [--key-env VAR]
BertholomusAI 2026-11-06. Read-only: it only sends requests.
"""
import argparse, concurrent.futures as cf, json, os, random, time, urllib.request

PROSE = "Write a detailed, plain-language explanation of how a city water treatment plant works, step by step."
CODE = ("Write a complete Python module that implements an LRU cache with TTL expiry, thread safety, "
        "and a decorator form, followed by unittest tests for every behaviour. Output only code.")


def call(url, key, model, messages, max_tokens, stream=True):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": stream, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    t0 = time.time(); first = None; text = []; usage = {}; finish = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece and first is None:
                    first = time.time()
                text.append(d.get("content") or "")
                finish = ch.get("finish_reason") or finish
    t1 = time.time()
    return {"ttft": (first or t1) - t0, "total": t1 - t0, "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"), "finish": finish, "text": "".join(text)}


def decode_row(res):
    n = res["completion_tokens"] or 0
    gen = max(res["total"] - res["ttft"], 1e-6)
    return {"tok_s": round(n / gen, 1), "tokens": n, "ttft_s": round(res["ttft"], 2), "finish": res["finish"]}


def filler(n_words, seed):
    rnd = random.Random(seed)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
    return " ".join(rnd.choice(words) + str(rnd.randint(0, 999)) for _ in range(n_words))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--key-env", default=None)
    a = ap.parse_args()
    key = os.environ.get(a.key_env) if a.key_env else None
    u = lambda c: [{"role": "user", "content": c}]
    rows = {"model": a.model, "url": a.url, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    call(a.url, key, a.model, u("Say ok."), 8)  # warm-up
    rows["decode_prose_c1"] = decode_row(call(a.url, key, a.model, u(PROSE), 1024))
    code1 = call(a.url, key, a.model, u(CODE), 1024)
    rows["decode_code_c1"] = decode_row(code1)
    t0 = time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda _: call(a.url, key, a.model, u(CODE), 1024), range(4)))
    wall = time.time() - t0
    rows["agg_c4"] = {"tok_s": round(sum(r["completion_tokens"] or 0 for r in res) / wall, 1), "wall_s": round(wall, 1)}
    # Ripple tokenizer: ~2.55 tokens per filler word (measured 2026-11-06: 5200 words -> 13.3K tokens)
    for name, words, seed in (("prefill_8k", 3200, 1), ("prefill_32k", 12800, 2)):
        # unique random text per run so no prefix cache can help; max 1 token
        p = filler(words, seed + int(time.time())) + "\n\nReply with the single word: done."
        r = call(a.url, key, a.model, u(p), 1)
        rows[name] = {"prompt_tokens": r["prompt_tokens"], "ttft_s": round(r["ttft"], 2),
                      "prompt_tok_s": round((r["prompt_tokens"] or 0) / max(r["ttft"], 1e-6), 0)}
    code2 = call(a.url, key, a.model, u(CODE), 1024)
    rows["greedy_repeat"] = {"identical": code1["text"] == code2["text"], "len": len(code1["text"])}
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
