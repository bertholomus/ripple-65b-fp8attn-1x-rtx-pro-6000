#!/usr/bin/env python3
"""GSM8K accuracy check (BertholomusAI 2026-11-06). Items: GSM8K test[200:200+N] (disjoint from calibration test[0:200]
and from the KLD set, which uses train). Greedy, 4096-token budget, concurrency 4, last number in the answer."""
import concurrent.futures as cf, json, re, sys, urllib.request

URL = __import__("os").environ.get("GSM8K_URL", "http://127.0.0.1:30000/v1/chat/completions")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
OUT = sys.argv[1]
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
t = pq.read_table(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
items = t[200:200 + N]


def gold(a):
    return a.split("####")[-1].strip().replace(",", "")


def one(it):
    q = it["question"] + "\n\nGive the final answer as a number on the last line, after 'Answer:'."
    body = {"model": "x", "messages": [{"role": "user", "content": q}], "max_tokens": 4096}
    if __import__("os").environ.get("GSM8K_GREEDY", "1") == "1":
        body["temperature"] = 0  # greedy; GSM8K_GREEDY=0 uses the server's default sampling
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=1800))
    c = r["choices"][0]
    txt = c["message"].get("content") or ""
    m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", txt) or re.findall(r"-?[\d,]*\.?\d+", txt)
    pred = m[-1].replace(",", "").rstrip(".") if m else None
    g = gold(it["answer"])
    ok = pred is not None and abs(float(pred) - float(g)) < 1e-6
    return {"ok": ok, "pred": pred, "gold": g, "finish": c["finish_reason"], "tokens": r["usage"]["completion_tokens"]}


with cf.ThreadPoolExecutor(4) as ex:
    res = list(ex.map(one, items))
s = {"n": len(res), "correct": sum(r["ok"] for r in res), "finish_length": sum(r["finish"] == "length" for r in res),
     "mean_tokens": sum(r["tokens"] for r in res) / len(res), "items": res}
json.dump(s, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in s.items() if k != "items"}))
