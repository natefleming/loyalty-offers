# Databricks notebook source
# MAGIC %md
# MAGIC # Gate 1 — Standalone vLLM smoke test (driver-only)
# MAGIC
# MAGIC The single most valuable diagnostic notebook in this bundle. It does **no
# MAGIC Spark, no Ray** — just loads vLLM on the cluster's driver GPU (or first
# MAGIC worker GPU via Ray if the driver is CPU-only) and runs five tiny prompts
# MAGIC against the chosen Qwen model.
# MAGIC
# MAGIC Run this **before** every full-pipeline iteration. If it fails, vLLM
# MAGIC itself is broken on the cluster — no orchestration tweak will save you.
# MAGIC If it passes, the engine is healthy and any later failure is in the
# MAGIC Spark/Ray/Pandas glue, not in vLLM.
# MAGIC
# MAGIC **Pass criterion:** 5 non-empty outputs in &lt; 5 minutes.
# MAGIC
# MAGIC ## Why driver-only (not via mapInPandas)
# MAGIC
# MAGIC Earlier runs on 2026-05-19 had us debating whether the failure was in
# MAGIC vLLM, in the Spark Python worker pipe, in Ray's actor scheduling, or
# MAGIC somewhere in the Hugging Face download path. Running vLLM on a single
# MAGIC GPU with zero distributed-systems glue isolates the engine. If it works
# MAGIC here, every later failure is somebody else's bug.

# COMMAND ----------
# MAGIC %md ## Read job parameters

# COMMAND ----------
from __future__ import annotations

import os
import time
from typing import Final

dbutils.widgets.text("catalog", "retail_consumer_goods")
dbutils.widgets.text("schema", "tractor_supply")
dbutils.widgets.text("hf_model", "Qwen/Qwen2.5-1.5B-Instruct")
dbutils.widgets.text("hf_secret_scope", "loyalty-offers")
dbutils.widgets.text("hf_secret_key", "HF_TOKEN")

CATALOG: Final[str] = dbutils.widgets.get("catalog")
SCHEMA: Final[str] = dbutils.widgets.get("schema")
HF_MODEL: Final[str] = dbutils.widgets.get("hf_model")
HF_TOKEN: Final[str] = dbutils.secrets.get(
    scope=dbutils.widgets.get("hf_secret_scope"),
    key=dbutils.widgets.get("hf_secret_key"),
)
HF_HOME: Final[str] = f"/Volumes/{CATALOG}/{SCHEMA}/hf_model_cache"
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_HOME"] = HF_HOME

print(f"[smoke_vllm] model={HF_MODEL} HF_HOME={HF_HOME}")

# COMMAND ----------
# MAGIC %md ## Confirm CUDA visibility on the driver
# MAGIC
# MAGIC If we have a GPU driver (e.g., a single-node GPU cluster), vLLM runs
# MAGIC right here. If the driver is CPU-only (our prod config: `Standard_DS4_v2`),
# MAGIC this cell will fail clearly — re-attach this notebook to a single-node GPU
# MAGIC interactive cluster for the smoke test, or run it on the worker via the
# MAGIC `00_smoke_mapinpandas` notebook instead.

# COMMAND ----------
import torch  # type: ignore[import-not-found]

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available on the driver. This smoke notebook is meant to "
        "be attached to a single-node GPU cluster (or run as a job task on a "
        "GPU node). For the worker-GPU pattern use notebooks/00_smoke_inference_engine.py."
    )

print(
    f"[smoke_vllm] CUDA OK: device={torch.cuda.get_device_name(0)} "
    f"capability={torch.cuda.get_device_capability(0)} "
    f"mem={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
)

# COMMAND ----------
# MAGIC %md ## Pre-download model weights (idempotent — uses Volume cache)

# COMMAND ----------
from huggingface_hub import snapshot_download  # noqa: E402

t0: float = time.time()
MODEL_LOCAL_PATH: Final[str] = snapshot_download(
    repo_id=HF_MODEL,
    cache_dir=HF_HOME,
    token=HF_TOKEN,
    max_workers=8,
)
print(f"[smoke_vllm] snapshot ready in {time.time() - t0:.1f}s at {MODEL_LOCAL_PATH}")

# COMMAND ----------
# MAGIC %md ## Construct vLLM and run 5 prompts
# MAGIC
# MAGIC Same `LLM(...)` constructor and `SamplingParams` we use in the main
# MAGIC inference notebook. If vLLM init or the first `chat()` call hangs here,
# MAGIC we know the engine itself is the problem — and we know it before
# MAGIC burning the rest of the cluster on a 40-minute Spark stage that would
# MAGIC have ended the same way.

# COMMAND ----------
from vllm import LLM, SamplingParams  # type: ignore[import-not-found]  # noqa: E402

t1: float = time.time()
llm: LLM = LLM(
    model=MODEL_LOCAL_PATH,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    max_model_len=2048,
    enable_prefix_caching=True,
    enforce_eager=True,
    tensor_parallel_size=1,
)
print(f"[smoke_vllm] LLM constructed in {time.time() - t1:.1f}s")

sampling: SamplingParams = SamplingParams(
    temperature=0.2,
    top_p=0.9,
    max_tokens=128,
)

# COMMAND ----------
prompts: list[list[dict[str, str]]] = [
    [{"role": "user", "content": "Give me one sentence about loyalty programs."}],
    [{"role": "user", "content": "Recommend a single offer for a rural customer."}],
    [{"role": "user", "content": "What does TSC sell?"}],
    [{"role": "user", "content": "Reply with only the word OK."}],
    [{"role": "user", "content": "Sum 17 and 25."}],
]

t2: float = time.time()
results = llm.chat(prompts, sampling, use_tqdm=False)
elapsed: float = time.time() - t2
print(f"[smoke_vllm] {len(results)} generations in {elapsed:.1f}s "
      f"({len(prompts) / elapsed:.2f} prompts/s)")

# COMMAND ----------
# MAGIC %md ## Assertions and output preview

# COMMAND ----------
assert len(results) == len(prompts), f"vLLM returned {len(results)} outputs for {len(prompts)} prompts"
for i, r in enumerate(results, start=1):
    text: str = r.outputs[0].text.strip() if r.outputs else ""
    assert text, f"prompt {i} produced empty output — vLLM is unhealthy"
    print(f"  [{i}] {text[:120]}")

print("[smoke_vllm] PASS — vLLM is healthy on this cluster")
