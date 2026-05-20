# Databricks notebook source
# MAGIC %md
# MAGIC # Smoke test — vLLM-on-Spark inference engine
# MAGIC
# MAGIC Lightweight integration test that runs ONLY when the job parameter
# MAGIC `run_smoke_tests=true`. Loads vLLM on a worker GPU via `mapInPandas`,
# MAGIC scores 100 synthetic customers, and asserts the output shape. Does NOT
# MAGIC write to UC — purely a sanity check.
# MAGIC
# MAGIC **Why it exists:** If the vLLM-on-this-cluster stack is broken (init
# MAGIC script regression, vLLM/torch ABI mismatch, CUDA context leak from a
# MAGIC prior run), this fails in ~3–5 min. Without it, the same failure would
# MAGIC be discovered ~30 min into the full inference pass on 10k–30M rows.
# MAGIC
# MAGIC **Pass criterion:** 400 output rows (100 customers × 4 ranks each) in
# MAGIC &lt; 10 minutes.
# MAGIC
# MAGIC See also `scripts/dev_smoke_vllm.py` for an interactive-only driver-GPU
# MAGIC variant (attach to a single-node GPU cluster).

# COMMAND ----------
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any, Final

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# COMMAND ----------
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

spark: SparkSession = SparkSession.getActiveSession()  # type: ignore[assignment]
assert spark is not None
print(f"[smoke_mapinpandas] model={HF_MODEL} HF_HOME={HF_HOME}")

# COMMAND ----------
# MAGIC %md ## Pre-download model on the driver (same pattern as `03_batch_inference`)

# COMMAND ----------
from huggingface_hub import snapshot_download  # noqa: E402

t0: float = time.time()
MODEL_LOCAL_PATH: Final[str] = snapshot_download(
    repo_id=HF_MODEL,
    cache_dir=HF_HOME,
    token=HF_TOKEN,
    max_workers=8,
)
print(f"[smoke_mapinpandas] snapshot ready in {time.time() - t0:.1f}s at {MODEL_LOCAL_PATH}")

# COMMAND ----------
# MAGIC %md ## Build a 100-row synthetic input DataFrame

# COMMAND ----------
SAMPLE_SIZE: Final[int] = 100
TIERS: Final[list[str]] = ["bronze", "silver", "gold", "platinum"]

sample_pdf: pd.DataFrame = pd.DataFrame(
    {
        "customer_id": [f"SMK-{i:06d}" for i in range(SAMPLE_SIZE)],
        "profile_text": [
            f"tier={TIERS[i % 4]} | years_member={i % 12} | ytd_spend=${(i * 13) % 2500} "
            f"| state=TX | top_categories=feed,fence | last_purchase_days_ago={i % 365}"
            for i in range(SAMPLE_SIZE)
        ],
    }
)
input_df: DataFrame = spark.createDataFrame(sample_pdf).repartition(1)
print(f"[smoke_mapinpandas] input rows={input_df.count()}, partitions={input_df.rdd.getNumPartitions()}")

# COMMAND ----------
# MAGIC %md ## Define the UDF and run it once
# MAGIC
# MAGIC The closure-captured `MODEL_LOCAL_PATH` is a Volume path readable from
# MAGIC every executor. The vLLM `LLM` object is cached in a module global so
# MAGIC if this task is retried, the second invocation reuses the engine.

# COMMAND ----------
OUTPUT_SCHEMA: Final[StructType] = StructType(
    [
        StructField("customer_id", StringType(), nullable=False),
        StructField("rank", IntegerType(), nullable=False),
        StructField("response", StringType(), nullable=True),
    ]
)

# MODEL_LOCAL_PATH and HF_HOME are captured directly from the closure;
# no `_LOCAL` rebinding (they're immutable module-level constants).


def score_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Score a single Spark partition by calling vLLM on the executor's GPU.

    For Gate 2 we emit a simplified output (rank + raw model response) instead
    of the full offer schema — the goal is just to confirm vLLM + Spark Python
    worker can coexist for the duration of one batch.

    Args:
        iterator: Stream of pandas DataFrames with columns ``customer_id`` and
            ``profile_text``.

    Yields:
        Pandas DataFrames matching ``OUTPUT_SCHEMA``. Four rows per input
        customer (one per rank), all with the same raw response text.
    """
    import os as _os

    from vllm import LLM, SamplingParams

    _os.environ["HF_HOME"] = HF_HOME

    cache: dict[str, LLM] = globals().setdefault("_SMOKE_VLLM_CACHE", {})
    llm: LLM | None = cache.get(MODEL_LOCAL_PATH)
    if llm is None:
        llm = LLM(
            model=MODEL_LOCAL_PATH,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=2048,
            enable_prefix_caching=True,
            enforce_eager=True,
            tensor_parallel_size=1,
        )
        cache[MODEL_LOCAL_PATH] = llm

    sampling = SamplingParams(temperature=0.2, max_tokens=64)

    for pdf in iterator:
        if pdf.empty:
            continue
        prompts: list[list[dict[str, str]]] = [
            [{"role": "user", "content": f"Reply OK for customer with profile: {p}"}]
            for p in pdf["profile_text"].tolist()
        ]
        results = llm.chat(prompts, sampling, use_tqdm=False)

        out_customer_id: list[str] = []
        out_rank: list[int] = []
        out_response: list[str] = []
        for customer_id, output in zip(pdf["customer_id"].tolist(), results, strict=True):
            text: str = output.outputs[0].text.strip() if output.outputs else ""
            for rank in (1, 2, 3, 4):
                out_customer_id.append(customer_id)
                out_rank.append(rank)
                out_response.append(text[:120])
        yield pd.DataFrame(
            {"customer_id": out_customer_id, "rank": out_rank, "response": out_response}
        )


# COMMAND ----------
t1: float = time.time()
scored_df: DataFrame = input_df.mapInPandas(score_partition, schema=OUTPUT_SCHEMA)
out_pdf: pd.DataFrame = scored_df.toPandas()
elapsed: float = time.time() - t1
print(f"[smoke_mapinpandas] received {len(out_pdf):,} rows in {elapsed:.1f}s")

# COMMAND ----------
# MAGIC %md ## Assertions

# COMMAND ----------
expected_rows: int = SAMPLE_SIZE * 4
assert len(out_pdf) == expected_rows, (
    f"expected {expected_rows} output rows, got {len(out_pdf)}"
)
assert out_pdf["customer_id"].nunique() == SAMPLE_SIZE
assert set(out_pdf["rank"].unique()) == {1, 2, 3, 4}
assert (out_pdf["response"].str.len() > 0).all(), "some customers got empty responses"

print(f"[smoke_mapinpandas] PASS — vLLM in mapInPandas is healthy "
      f"({SAMPLE_SIZE} customers, {expected_rows} output rows, {elapsed:.1f}s wall time)")
