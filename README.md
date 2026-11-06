
# Ripple-65B-Preview NVFP4 + FP8 attention

**Mixed NVFP4/FP8 quant of [ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) by BertholomusAI.**
It is built from Ricursive's own [ricursiveAI/Ripple-65B-Preview-NVFP4](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4)
(revision `99eea4477a001a41b86914e41feb4b2ed86105e5`): every MLP tensor, norm, embedding and the LM head are Ricursive's bytes,
unchanged. Only the 400 attention projections, which Ricursive ships in BF16, are now FP8 with calibrated static scales.
Model and weights © Ricursive under Apache-2.0. Not affiliated with or endorsed by Ricursive.

Served with SGLang 0.5.19 and Ricursive's DFlash2 drafter ([ricursiveAI/Ripple-65B-Preview-DFlash2](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2), Apache-2.0, used as is and not redistributed here).

## Measured on one RTX PRO 6000 Blackwell Max-Q (96 GB, 300 W)

SGLang 0.5.19, DFlash2 drafter, 16 draft tokens, up to 4 streams, BF16 KV cache.

| row | result |
|---|---|
| checkpoint size | **43.79 GB** |
| one stream, code | **218 tok/s** (215–222) |
| one stream, prose | **56.9 tok/s** (56.1–58.3) |
| four streams, code, total | **653 tok/s** (623–661) |
| prefill, 8K-token prompt | **~3,010 tok/s** |
| prefill, 32K-token prompt | **~2,830 tok/s** |
| greedy output repeats exactly | yes |

Median of 3 runs (range in brackets). One stream = one request, 1,024 new tokens; four streams = 4 requests at once,
total new tokens / wall time. The drafter accepts about 6.8 tokens per step on code.
Why FP8 attention: a decode step reads all weights once. The attention projections are 13.4 GB in FP8 here, half of their
26.8 GB in BF16, and SGLang's FP8 kernels run them about twice as fast as BF16 on this card.

## Quality

| check | result |
|---|---|
| KL divergence per token vs the NVFP4 source checkpoint, 120 held-out prompts, 40,461 positions | **0.0117** (95% CI 0.0102–0.0134) |
| top-1 token agreement with the NVFP4 source checkpoint | **96.4%** |
| GSM8K, 150 test items (`test[200:350]`), greedy, 4,096-token budget | **143 / 150** |
| 12 small Python tasks with hidden asserts, greedy | **12 / 12** |
| tool call (streamed and non-streamed arguments equal), two-turn tool use | pass |

KL divergence is measured against the NVFP4 checkpoint this build is made from, not against BF16. Method: the source writes a
384-token continuation per prompt (temperature 0.6), then both are teacher-forced on the same tokens with drafting off at
concurrency 1; KL(source ‖ this) over the source's top-20 tokens per position. Held-out prompts: UltraChat-200k
`test_sft[1000:1060]`, GSM8K `train[:30]`, MBPP `train[:30]`.

Weights: **https://huggingface.co/bertholomus/Ripple-65B-Preview-NVFP4-FP8Attn**

## Run it

```bash
pip install "sglang==0.5.19" huggingface_hub
git clone https://github.com/bertholomus/ripple-65b-fp8attn-1x-rtx-pro-6000 && cd ripple-65b-fp8attn-1x-rtx-pro-6000
python recipe/serve.py        # OpenAI API on http://127.0.0.1:30000/v1, model "Ripple-65B-Preview-NVFP4-FP8Attn"
```

`recipe/serve.py` downloads this checkpoint, the DFlash2 drafter and Ricursive's model code (`inference/local/*.py` and the chat
template, from the NVFP4 repo at the pinned revision), checks the checkpoint layout, applies Ricursive's two scheduler fixes
(`spark.run_worker`), and starts SGLang's HTTP server with tool-call and reasoning parsers. Settings via env: `RIPPLE_PORT`,
`RIPPLE_HOST`, `RIPPLE_CTX` (131072), `RIPPLE_MAX_RUNNING` (4), `RIPPLE_DRAFT_TOKENS` (16), `RIPPLE_SPEC=0` to serve without the drafter.

**pip-installed CUDA:** FlashInfer JIT-compiles its FP8 GEMM module on first use and links `-lcublas -lcublasLt`. The `nvidia-*-cu13`
wheels ship only `libcublas.so.13` / `libcublasLt.so.13`, so the link fails with `cannot find -lcublas`. Fix: add `libcublas.so` and
`libcublasLt.so` symlinks to the `.so.13` files in that wheel's `lib/` folder, or use a full CUDA 13 toolkit.

## How it was made

1. **Calibration.** `recipe/calib_hook.py` wraps the attention forward pass and keeps max |input| for each of the 400 projections (q, k, g and v
   see their own inputs, o sees the gated attention output). It ran in Ricursive's NVFP4 build on 864 prompts (UltraChat-200k `test_sft[0:300]`,
   HumanEval, GSM8K `test[0:200]`, MBPP `test[0:200]`; 94,708 prompt and 82,316 generated tokens). The resulting maxima are `recipe/input_amax.json`.
   Calibration sets only these 400 scalar maxima; no weights are trained.
2. **Conversion.** `recipe/make_fp8attn.py` copies every non-attention tensor byte-for-byte and converts each BF16 attention weight to FP8 E4M3 with
   one per-tensor weight scale (amax / 448) and one static per-tensor input scale (calibrated amax / 448). Worst relative weight error is 2.7%.
   The checkpoint config is ModelOpt `MIXED_PRECISION`: MLP `NVFP4` (group size 16), attention `FP8`, `lm_head` and embeddings BF16.
   The script checks the result: 2,723 tensors (1,923 + 800 scales), all 1,523 non-attention tensors byte-identical to the source.
   Per-tensor errors and scales are in `fp8attn_report.json`.

## Not covered

- Other GPUs. Only the RTX PRO 6000 Blackwell Max-Q (sm_120) was tested. The full-power card should be faster; numbers will differ.
- Context past 32K tokens. The window is set to 131,072; prefill rows stop at 32K.
- Engines other than SGLang 0.5.19.
- A drift measurement against BF16. The reference here is Ricursive's NVFP4.

## Credits

- Ricursive: Ripple-65B-Preview, the NVFP4 checkpoint this build keeps byte-for-byte, the DFlash2 drafter, and the inference code
  (`inference/local/`, chat template), all Apache-2.0. Files from Ricursive's repo are included unchanged.
- SGLang and FlashInfer projects (Apache-2.0): ModelOpt mixed-precision loader, FP8 and NVFP4 kernels.
- Calibration prompts: UltraChat-200k (MIT), HumanEval (MIT), GSM8K (MIT), MBPP (CC BY 4.0). Only derived scalar maxima are distributed.

License: Apache-2.0, as the base model. This is a modified version of Ricursive's NVFP4 checkpoint: attention weights were re-quantized and
`config.json` / `hf_quant_config.json` carry the new `quantization_config`. Everything else is unchanged.

BertholomusAI · [huggingface.co/bertholomus](https://huggingface.co/bertholomus) · [github.com/bertholomus](https://github.com/bertholomus)

## Reproduce the numbers

With the server up on port 30000:

```bash
python tools/lane_bench.py --url http://127.0.0.1:30000/v1 --model Ripple-65B-Preview-NVFP4-FP8Attn --out rows.json
python tools/quality_panel.py --url http://127.0.0.1:30000/v1 --model Ripple-65B-Preview-NVFP4-FP8Attn --out quality.json
pip install pyarrow && python tools/gsm8k_check.py gsm8k.json 150
```

KL divergence (`tools/kld_gate.py`, prompts in `tools/kld/kld_prompts.json`): serve the NVFP4 source with `RIPPLE_SPEC=0`,
run `KLD_DIR=tools/kld python tools/kld_gate.py ref`; then serve this build with `RIPPLE_SPEC=0` and run
`KLD_DIR=tools/kld KLD_CONC=1 python tools/kld_gate.py cand fp8attn` and `... score fp8attn`.
Use concurrency 1 for both teacher-forcing passes: batching alone changes logits.
