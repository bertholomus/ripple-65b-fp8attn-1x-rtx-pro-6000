#!/usr/bin/env python3
"""Serve Ripple-65B-Preview NVFP4 + FP8 attention with SGLang and the DFlash2 drafter, OpenAI API on one GPU.

Tested: SGLang 0.5.19, one RTX PRO 6000 Blackwell Max-Q (sm_120, 96 GB, 300 W), CUDA 13, Python 3.12.

    pip install "sglang==0.5.19" huggingface_hub
    python serve.py                       # http://127.0.0.1:30000/v1, served name Ripple-65B-Preview-NVFP4-FP8Attn

The model code (model.py, ripple_format.py, spark.py) and the chat template are Ricursive's, downloaded from
ricursiveAI/Ripple-65B-Preview-NVFP4 at the pinned revision below (Apache-2.0). This script adds:
  - a checkpoint check for this layout (NVFP4 MLP, FP8 attention, BF16 lm_head/embeddings/KV);
  - Ricursive's scheduler-worker numerical fixes (spark.run_worker), unchanged;
  - SGLang's HTTP server instead of the offline engine, and 16 draft tokens instead of 32 (measured faster here).

Environment: RIPPLE_TARGET, RIPPLE_DRAFT (repo id or local dir), RIPPLE_PORT (30000), RIPPLE_HOST (127.0.0.1),
RIPPLE_CTX (131072), RIPPLE_MAX_TOTAL (262144), RIPPLE_MAX_RUNNING (4), RIPPLE_DRAFT_TOKENS (16), RIPPLE_SPEC (1).
"""
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

OFFICIAL = "ricursiveAI/Ripple-65B-Preview-NVFP4"
OFFICIAL_REV = "99eea4477a001a41b86914e41feb4b2ed86105e5"
TARGET = os.environ.get("RIPPLE_TARGET", "bertholomus/Ripple-65B-Preview-NVFP4-FP8Attn")
DRAFT = os.environ.get("RIPPLE_DRAFT", "ricursiveAI/Ripple-65B-Preview-DFlash2")
SPEC = os.environ.get("RIPPLE_SPEC", "1") == "1"


def local(repo):
    return Path(repo) if Path(repo).is_dir() else Path(snapshot_download(repo))


CODE_ROOT = Path(snapshot_download(OFFICIAL, revision=OFFICIAL_REV, allow_patterns=[
    "inference/local/model.py", "inference/local/ripple_format.py", "inference/local/spark.py",
    "chat_template.jinja", "config.json"]))
sys.path.insert(0, str(CODE_ROOT / "inference/local"))
os.environ.setdefault("SGLANG_FLASHINFER_USE_PAGED", "1")
os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")

import torch  # noqa: E402
from sglang.srt.entrypoints.http_server import launch_server  # noqa: E402
from sglang.srt.models.registry import ModelRegistry  # noqa: E402
from sglang.srt.server_args import prepare_server_args  # noqa: E402
from model import RicursiveForCausalLM  # noqa: E402
import ripple_format  # noqa: E402
import spark  # noqa: E402

ModelRegistry.models["RicursiveForCausalLM"] = RicursiveForCausalLM
ripple_format.register()


def check_target(target, draft):
    cfg = json.loads((target / "config.json").read_text())
    if cfg.get("architectures") != ["RicursiveForCausalLM"]:
        raise ValueError("target must be a RicursiveForCausalLM checkpoint")
    q = cfg.get("quantization_config") or {}
    if q.get("quant_algo") != "MIXED_PRECISION":
        return spark.check_checkpoints(target, draft)  # the official NVFP4/BF16-attention layout
    ql = q.get("quantized_layers") or {}
    for i in range(cfg["num_hidden_layers"]):
        for p in ("gate_up_proj", "down_proj"):
            if ql.get(f"model.layers.{i}.mlp.{p}", {}).get("quant_algo") != "NVFP4":
                raise ValueError(f"layer {i} {p} must be NVFP4")
        for p in "qkvog":
            if ql.get(f"model.layers.{i}.self_attn.{p}_proj", {}).get("quant_algo") != "FP8":
                raise ValueError(f"layer {i} {p}_proj must be FP8")
    if q.get("kv_cache_quant_algo") is not None:
        raise ValueError("the KV cache must stay BF16")
    if sorted(q.get("exclude_modules") or []) != ["lm_head", "model.embed_tokens"]:
        raise ValueError("lm_head and embed_tokens must stay BF16")
    if draft is not None:
        d = json.loads((draft / "config.json").read_text())
        if d.get("architectures") != ["DFlash2DraftModel"] or d.get("quantization_config"):
            raise ValueError("the draft must be the unquantized BF16 DFlash2 checkpoint")


def main():
    target = local(TARGET)
    draft = local(DRAFT) if SPEC else None
    argv = ["--model-path", str(target), "--chat-template", str(CODE_ROOT / "chat_template.jinja"),
            "--served-model-name", os.environ.get("RIPPLE_SERVED_NAME", "Ripple-65B-Preview-NVFP4-FP8Attn"),
            "--host", os.environ.get("RIPPLE_HOST", "127.0.0.1"), "--port", os.environ.get("RIPPLE_PORT", "30000"),
            "--trust-remote-code", "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16", "--tp-size", "1",
            "--attention-backend", "flashinfer", "--fp4-gemm-backend", "flashinfer_cutlass",
            "--flashinfer-autotune-skip-ops", "fp8_gemm",
            "--context-length", os.environ.get("RIPPLE_CTX", "131072"),
            "--max-total-tokens", os.environ.get("RIPPLE_MAX_TOTAL", "262144"), "--mem-fraction-static", "0.80",
            "--max-running-requests", os.environ.get("RIPPLE_MAX_RUNNING", "4"),
            "--cuda-graph-bs-decode", "1", "2", "4", "--cuda-graph-backend-prefill", "disabled",
            "--disable-radix-cache", "--model-loader-extra-config", '{"enable_multithread_load": false}',
            "--random-seed", "42", "--watchdog-timeout", "1800",
            "--tool-call-parser", "ripple", "--reasoning-parser", "ripple"]
    if SPEC:
        argv += ["--speculative-draft-model-path", str(draft), "--speculative-algorithm", "DFLASH",
                 "--speculative-draft-model-quantization", "unquant",
                 "--speculative-num-draft-tokens", os.environ.get("RIPPLE_DRAFT_TOKENS", "16"),
                 "--speculative-draft-window-size", "16384", "--speculative-draft-attention-backend", "triton"]
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    check_target(target, draft)
    launch_server(prepare_server_args(argv), run_scheduler_process_func=spark.run_worker)


if __name__ == "__main__":
    main()
