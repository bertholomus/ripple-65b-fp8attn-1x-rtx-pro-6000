#!/usr/bin/env python3
"""Quality/exactness panel for a coding lane: 12 small Python tasks with hidden asserts, temperature 0.
Prints JSON {passed, total, per_task:{id:{ok, sha}}, finish_length}. The asserts are never shown to the model.
Usage: quality_panel.py --url URL --model MODEL [--out FILE]
BertholomusAI 2026-11-06."""
import argparse, concurrent.futures as cf, hashlib, json, re, subprocess, sys, tempfile, urllib.request

TASKS = [
 ("rle", "def rle(s: str) -> str: run-length encode, e.g. 'aaabcc' -> 'a3b1c2'; empty -> ''.",
  "assert rle('aaabcc')=='a3b1c2'\nassert rle('')==''\nassert rle('z')=='z1'\nassert rle('aaaaaaaaaaaa')=='a12'"),
 ("roman", "def to_roman(n: int) -> str: convert 1..3999 to Roman numerals.",
  "assert to_roman(1994)=='MCMXCIV'\nassert to_roman(4)=='IV'\nassert to_roman(3999)=='MMMCMXCIX'\nassert to_roman(40)=='XL'"),
 ("balanced", "def balanced(s: str) -> bool: True if (), [], {} are balanced; other characters are ignored.",
  "assert balanced('a(b[c]{d}e)f')\nassert not balanced('(]')\nassert not balanced('((')\nassert balanced('')\nassert not balanced('}{')"),
 ("merge", "def merge_intervals(xs: list[list[int]]) -> list[list[int]]: merge overlapping or touching closed intervals, return sorted.",
  "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\nassert merge_intervals([[5,6],[1,2]])==[[1,2],[5,6]]"),
 ("lcs", "def lcs_len(a: str, b: str) -> int: length of the longest common subsequence.",
  "assert lcs_len('ABCBDAB','BDCABA')==4\nassert lcs_len('','x')==0\nassert lcs_len('abc','abc')==3\nassert lcs_len('abc','def')==0"),
 ("topk", "def top_k_words(text: str, k: int) -> list[str]: the k most frequent lowercase words (split on non-letters); ties broken alphabetically.",
  "assert top_k_words('b a b c a b', 2)==['b','a']\nassert top_k_words('X x y', 1)==['x']\nassert top_k_words('c b a', 2)==['a','b']\nassert top_k_words('', 3)==[]"),
 ("flatten", "def flatten(x) -> list: flatten arbitrarily nested lists/tuples into a flat list; strings are atoms.",
  "assert flatten([1,[2,(3,[4])],'ab'])==[1,2,3,4,'ab']\nassert flatten([])==[]\nassert flatten([[[]]])==[]\nassert flatten(['a',['b']])==['a','b']"),
 ("isprime", "def primes_upto(n: int) -> list[int]: all primes <= n, ascending.",
  "assert primes_upto(1)==[]\nassert primes_upto(2)==[2]\nassert primes_upto(30)==[2,3,5,7,11,13,17,19,23,29]\nassert len(primes_upto(10000))==1229"),
 ("parse", "def parse_kv(s: str) -> dict: parse 'a=1; b = two ;c=' into {'a':'1','b':'two','c':''}; ignore empty segments; later keys win.",
  "assert parse_kv('a=1; b = two ;c=')=={'a':'1','b':'two','c':''}\nassert parse_kv('')=={}\nassert parse_kv('a=1;a=2;;')=={'a':'2'}\nassert parse_kv(' x = y=z ')=={'x':'y=z'}"),
 ("lru", "class LRU: LRU(capacity) with get(key) -> value or -1 and put(key, value); evict least recently used.",
  "c=LRU(2)\nc.put(1,1);c.put(2,2)\nassert c.get(1)==1\nc.put(3,3)\nassert c.get(2)==-1\nc.put(4,4)\nassert c.get(1)==-1\nassert c.get(3)==3 and c.get(4)==4"),
 ("bsearch", "def first_ge(xs: list[int], t: int) -> int: index of the first element >= t in sorted xs, or len(xs).",
  "assert first_ge([1,2,2,3],2)==1\nassert first_ge([],5)==0\nassert first_ge([1,3],4)==2\nassert first_ge([5,6],1)==0"),
 ("wrap", "def wrap(text: str, width: int) -> list[str]: greedy word wrap on spaces; a word longer than width gets its own line unbroken.",
  "assert wrap('the quick brown fox', 10)==['the quick','brown fox']\nassert wrap('', 5)==[]\nassert wrap('abcdefghijk xy', 5)==['abcdefghijk','xy']\nassert wrap('a b c', 1)==['a','b','c']"),
]


def ask(url, model, prompt):
    body = {"model": model, "temperature": 0, "max_tokens": 6000,
            "messages": [{"role": "user", "content": prompt + "\nReply with only one ```python code block, no tests, no prose."}]}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    c = r["choices"][0]
    return c["message"].get("content") or "", c.get("finish_reason")


def run_task(url, model, t):
    tid, prompt, asserts = t
    text, finish = ask(url, model, prompt)
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    code = m[-1] if m else text
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + asserts + "\n")
    try:
        ok = subprocess.run([sys.executable, f.name], capture_output=True, timeout=30).returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    return tid, {"ok": ok, "finish": finish, "sha": hashlib.sha256(text.encode()).hexdigest()[:16]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    with cf.ThreadPoolExecutor(2) as ex:
        res = dict(ex.map(lambda t: run_task(a.url, a.model, t), TASKS))
    out = {"passed": sum(v["ok"] for v in res.values()), "total": len(TASKS),
           "finish_length": sum(v["finish"] == "length" for v in res.values()), "per_task": res}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
