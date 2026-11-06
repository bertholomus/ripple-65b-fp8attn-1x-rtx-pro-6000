#!/usr/bin/env python3
"""KLD gate for the FP8-attention build (BertholomusAI 2026-11-06).

Step `ref`  (official NVFP4 lane up): for each held-out prompt, generate a 384-token reference continuation
            (temperature 0.6, top_p 0.95, seed 7), then teacher-force prompt+continuation and save, for every
            continuation position, the reference top-20 token ids/logprobs. -> fp8attn/kld_ref.jsonl
Step `cand` (candidate lane up): teacher-force the same token ids, top-20 logprobs. -> fp8attn/kld_<tag>.jsonl
Step `score`: per position, KL(ref || cand) over the reference top-20 support (renormalised on both sides;
            a token missing from the candidate's top-20 gets the candidate's 20th logprob as an upper bound),
            plus top-1 agreement. Reports mean KLD and top-1 agreement with a bootstrap 95% interval.

Uses SGLang's native /generate with input_ids, return_logprob, logprob_start_len, top_logprobs_num.
Spec decode should be OFF for teacher forcing (RIPPLE_SPEC=0) so logprobs come from plain forward passes.
"""
import concurrent.futures as cf, json, math, os, random, sys, urllib.request

BASE = os.environ.get("KLD_BASE", "http://127.0.0.1:30000")
D = os.environ.get("KLD_DIR", "kld")
TOP = 20


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1200))


def tok_chat(prompt):
    r = post("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": prompt}],
                                      "max_tokens": 1, "temperature": 0})
    return r


def ref_one(it):
    i, rec = it
    # chat-template the prompt through the server's tokenizer path: /generate with text + chat template is not
    # exposed, so use the OpenAI endpoint to generate, then re-tokenise prompt+completion via /tokenize.
    r = post("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": rec["p"]}],
                                      "max_tokens": 384, "temperature": 0.6, "top_p": 0.95, "seed": 7,
                                      "separate_reasoning": False, "skip_special_tokens": False})
    comp = r["choices"][0]["message"].get("content") or ""
    reasoning = r["choices"][0]["message"].get("reasoning_content") or ""
    pt = post("/v1/tokenize", {"model": "x", "messages": [{"role": "user", "content": rec["p"]}],
                               "add_generation_prompt": True})
    p_ids = pt["tokens"]
    c_ids = post("/v1/tokenize", {"model": "x", "prompt": (reasoning + comp), "add_special_tokens": False})["tokens"]
    ids = p_ids + c_ids
    lp = forced(ids, len(p_ids))
    return {"i": i, "src": rec["src"], "n_prompt": len(p_ids), "ids": ids, "top": lp}


def forced(ids, start):
    r = post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                           "return_logprob": True, "logprob_start_len": start - 1, "top_logprobs_num": TOP})
    tops = r["meta_info"]["input_top_logprobs"]
    out = []
    for t in tops[1:] if len(tops) == len(ids) - start + 1 else tops:
        if not t:
            continue
        out.append([[x[1], x[0]] for x in t])  # [token_id, logprob]
    return out


def cand_one(rec):
    return {"i": rec["i"], "top": forced(rec["ids"], rec["n_prompt"])}


def score(ref_path, cand_path):
    R = {r["i"]: r for r in map(json.loads, open(ref_path))}
    C = {r["i"]: r for r in map(json.loads, open(cand_path))}
    per_seq = []
    for i, r in R.items():
        c = C[i]
        n = min(len(r["top"]), len(c["top"]))
        kl_sum = agree = 0
        for a, b in zip(r["top"][:n], c["top"][:n]):
            ra = {t: l for t, l in a}
            cb = {t: l for t, l in b}
            floor = min(cb.values())
            zr = math.log(sum(math.exp(l) for l in ra.values()))
            ids = list(ra)
            cl = [cb.get(t, floor) for t in ids]
            zc = math.log(sum(math.exp(l) for l in cl))
            kl_sum += sum(math.exp(ra[t] - zr) * ((ra[t] - zr) - (l - zc)) for t, l in zip(ids, cl))
            agree += max(ra, key=ra.get) == max(cb, key=cb.get)
        per_seq.append((kl_sum, agree, n))
    tot_n = sum(x[2] for x in per_seq)
    kld = sum(x[0] for x in per_seq) / tot_n
    top1 = sum(x[1] for x in per_seq) / tot_n
    rnd = random.Random(0)
    boots = []
    for _ in range(1000):
        s = [rnd.choice(per_seq) for _ in per_seq]
        boots.append(sum(x[0] for x in s) / sum(x[2] for x in s))
    boots.sort()
    return {"positions": tot_n, "sequences": len(per_seq), "kld_mean": kld, "kld_ci95": [boots[25], boots[974]],
            "top1_agreement": top1}


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "ref":
        P = json.load(open(f"{D}/kld_prompts.json"))
        with open(f"{D}/kld_ref.jsonl", "w") as f, cf.ThreadPoolExecutor(4) as ex:
            for n, rec in enumerate(ex.map(ref_one, enumerate(P)), 1):
                f.write(json.dumps(rec) + "\n")
                if n % 20 == 0:
                    print(f"ref {n}/{len(P)}", flush=True)
        print("REF DONE", flush=True)
    elif step == "cand":
        tag = sys.argv[2]
        R = [json.loads(l) for l in open(f"{D}/kld_ref.jsonl")]
        conc = int(__import__("os").environ.get("KLD_CONC", "4"))
        with open(f"{D}/kld_{tag}.jsonl", "w") as f, cf.ThreadPoolExecutor(conc) as ex:
            for rec in ex.map(cand_one, R):
                f.write(json.dumps(rec) + "\n")
        print("CAND DONE", tag, flush=True)
    elif step == "score":
        ref = f"{D}/kld_{sys.argv[3]}.jsonl" if len(sys.argv) > 3 else f"{D}/kld_ref.jsonl"
        print(json.dumps(score(ref, f"{D}/kld_{sys.argv[2]}.jsonl"), indent=1))
