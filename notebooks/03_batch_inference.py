# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Batch inference (vLLM + Qwen2.5 + `mapInPandas`)
# MAGIC
# MAGIC Reads unscored customers from `customers`, runs Qwen2.5-Instruct on each GPU
# MAGIC worker via Spark `mapInPandas` (one task = one worker = one vLLM engine), and
# MAGIC appends 4 ranked offers per customer to `offer_recommendations`. The pipeline
# MAGIC is resumable via an anti-join against the output table at the start of the run.
# MAGIC
# MAGIC ## Shipping config
# MAGIC
# MAGIC - **Model**: Qwen2.5-1.5B-Instruct (1.5B + `guided_json` hits the 8h SLO on 30M
# MAGIC   rows at ~$1k/run; 7B costs ~4× more for the same SLO).
# MAGIC - **Inference**: vLLM 0.8.5.post1 + xgrammar backend + `guided_json` (one-shot
# MAGIC   structured-output call per customer). The only backend that handles a 20-
# MAGIC   element offer_id enum without hanging.
# MAGIC - **Schema**: no `minItems`/`maxItems`/`minLength`/`maxLength`/`minimum`/`maximum`
# MAGIC   (xgrammar silently demotes to Outlines on any of those, re-triggering an FSM
# MAGIC   hang on the enum). Equivalent invariants enforced in Python post-processing.
# MAGIC   Guarded by `tests/test_smoke.py::test_offers_schema_has_no_xgrammar_blocking_constraints`.
# MAGIC - **Cluster**: fixed `num_workers=4` dev / `90` prod, one `mapInPandas` partition
# MAGIC   per worker, `spark.python.worker.reuse=false` to bound cgroup memory growth.
# MAGIC
# MAGIC The empirical comparison that selected these knobs (6 variants × 1.5B and 7B) is
# MAGIC in the vault memory `reference_vllm_guided_decoding_matrix_2026_05_19.md` and the
# MAGIC postmortem at `~/.claude/plans/i-need-you-to-joyful-bachman.md`. The version
# MAGIC pins are in `scripts/init_vllm.sh` (do NOT bump in isolation — vllm 0.9.0+
# MAGIC requires torch ≥ 2.7 which breaks DBR 16.4's pre-installed flash_attn).

# COMMAND ----------
from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from datetime import date, datetime, timezone
from typing import Any, Final

import pandas as pd
from huggingface_hub import snapshot_download
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# COMMAND ----------
# MAGIC %md ## Read job parameters and the upstream `run_date`

# COMMAND ----------
dbutils.widgets.text("catalog", "retail_consumer_goods")
dbutils.widgets.text("schema", "tractor_supply")
dbutils.widgets.text("hf_model", "Qwen/Qwen2.5-7B-Instruct")
dbutils.widgets.text("hf_secret_scope", "loyalty-offers")
dbutils.widgets.text("hf_secret_key", "HF_TOKEN")
# `force_rerun=true` clears every offer_recommendation for this run_date before
# scoring. The default (anti-join skip) is correct for fault-tolerant retries;
# `force_rerun=true` is the explicit "re-score today from scratch" escape hatch.
dbutils.widgets.text("force_rerun", "false")
# `inference_mode` controls how we invoke vLLM:
#   - "guided_json" (default): one-shot `llm.chat()` with `GuidedDecodingParams(json=...)`.
#     Empirical winner from the May 2026 backend matrix. ~3% fallback on 1.5B,
#     ~0% on 7B. Works on xgrammar with the cleaned schema (no min/max constraints).
#   - "none": one-shot completion with no structured-output guard; relies entirely on
#     post-hoc JSON parsing + catalog filter + fallback. Useful as a baseline
#     comparator; produces ~17–28% fallback rate.
dbutils.widgets.text("inference_mode", "guided_json")

# Backend selection for vLLM guided decoding (ignored when inference_mode="none").
# `xgrammar` is the only backend that handles our 20-element offer_id enum
# without hanging — Outlines compiles the enum into a regex FSM that explodes
# at this cardinality (outlines#680, vllm#12005).
dbutils.widgets.text("guided_decoding_backend", "xgrammar")

CATALOG: Final[str] = dbutils.widgets.get("catalog")
SCHEMA: Final[str] = dbutils.widgets.get("schema")
HF_MODEL: Final[str] = dbutils.widgets.get("hf_model")
HF_TOKEN: Final[str] = dbutils.secrets.get(
    scope=dbutils.widgets.get("hf_secret_scope"),
    key=dbutils.widgets.get("hf_secret_key"),
)
FORCE_RERUN: Final[bool] = dbutils.widgets.get("force_rerun").strip().lower() == "true"
INFERENCE_MODE: Final[str] = dbutils.widgets.get("inference_mode").strip().lower()
GUIDED_DECODING_BACKEND: Final[str] = dbutils.widgets.get("guided_decoding_backend").strip().lower()
# Kept in sync with `resources/job.yml` job parameter definition.
_VALID_INFERENCE_MODES: Final[set[str]] = {"guided_json", "none"}
_VALID_BACKENDS: Final[set[str]] = {"xgrammar", "lm-format-enforcer", "outlines"}
assert INFERENCE_MODE in _VALID_INFERENCE_MODES, (
    f"inference_mode must be one of {_VALID_INFERENCE_MODES}, got {INFERENCE_MODE!r}"
)
assert GUIDED_DECODING_BACKEND in _VALID_BACKENDS, (
    f"guided_decoding_backend must be one of {_VALID_BACKENDS}, got {GUIDED_DECODING_BACKEND!r}"
)
RUN_DATE: Final[str] = dbutils.jobs.taskValues.get(
    taskKey="provision", key="run_date", default="", debugValue="1970-01-01"
)
assert RUN_DATE, "run_date task value missing from upstream `provision` task"

HF_HOME: Final[str] = f"/Volumes/{CATALOG}/{SCHEMA}/hf_model_cache"
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_HOME"] = HF_HOME

spark: SparkSession = SparkSession.getActiveSession()  # type: ignore[assignment]
assert spark is not None
print(f"[inference] model={HF_MODEL} run_date={RUN_DATE} HF_HOME={HF_HOME}")

# COMMAND ----------
# MAGIC %md ## Pre-download model on the driver
# MAGIC
# MAGIC One download per cluster lifetime — saved into the Volume cache and
# MAGIC read from there by every executor. Avoids parallel download races that
# MAGIC produce corrupt `.incomplete` blobs.

# COMMAND ----------
MODEL_LOCAL_PATH: Final[str] = snapshot_download(
    repo_id=HF_MODEL,
    cache_dir=HF_HOME,
    token=HF_TOKEN,
    max_workers=8,
)
print(f"[inference] model snapshot at {MODEL_LOCAL_PATH}")

# COMMAND ----------
# MAGIC %md ## Find unscored customers via anti-join

# COMMAND ----------
if FORCE_RERUN:
    print(f"[inference] force_rerun=true; clearing existing rows for {RUN_DATE}")
    spark.sql(
        f"DELETE FROM {CATALOG}.{SCHEMA}.offer_recommendations "
        f"WHERE run_date = DATE'{RUN_DATE}'"
    )

customers_df: DataFrame = spark.table(f"{CATALOG}.{SCHEMA}.customers").where(
    F.col("run_date") == F.to_date(F.lit(RUN_DATE))
)
scored_df: DataFrame = (
    spark.table(f"{CATALOG}.{SCHEMA}.offer_recommendations")
    .where(F.col("run_date") == F.to_date(F.lit(RUN_DATE)))
    .select("customer_id")
    .distinct()
)
to_score_df: DataFrame = customers_df.join(
    scored_df, on="customer_id", how="left_anti"
).select("customer_id", "profile_text")

if not to_score_df.head(1):
    print("[inference] every customer already scored; exiting cleanly")
    dbutils.notebook.exit("OK: no unscored customers")

# COMMAND ----------
# MAGIC %md ## Size partitions for the GPU fleet
# MAGIC
# MAGIC **One Spark partition per worker GPU.** With `spark.python.worker.reuse=
# MAGIC false` (set in cluster spec to bound cgroup memory growth), each task
# MAGIC spawns a fresh Python worker, loads vLLM once, and processes the full
# MAGIC partition before exiting. More partitions ⇒ more vLLM cold-loads ⇒
# MAGIC strictly slower.

# COMMAND ----------
NUM_WORKERS: Final[int] = int(
    spark.conf.get("spark.databricks.clusterUsageTags.clusterTargetWorkers", "1")
)
TARGET_PARTITIONS: Final[int] = NUM_WORKERS
print(f"[inference] workers={NUM_WORKERS} target_partitions={TARGET_PARTITIONS}")

# COMMAND ----------
# MAGIC %md ## Load offer catalog and build the guided-decoding JSON schema

# COMMAND ----------
offers_pdf: pd.DataFrame = (
    spark.table(f"{CATALOG}.{SCHEMA}.offer_catalog")
    .select("offer_id", "name", "description", "category", "audience_hint")
    .toPandas()
)
OFFER_IDS: Final[list[str]] = sorted(offers_pdf["offer_id"].tolist())
OFFER_SUMMARY: Final[str] = "\n".join(
    f"- {row.offer_id}: {row.name} ({row.category}; targets {row.audience_hint}). {row.description}"
    for row in offers_pdf.itertuples()
)

# vLLM's guided decoding masks any token that would break this schema. By
# construction every model response: (a) parses as JSON, (b) has exactly 4
# offers, (c) only ever references a real `offer_id`.
# xgrammar-clean schema: no `minItems`, `maxItems`, `minLength`, `maxLength`,
# `minimum`, or `maximum`. Those keywords cause vLLM 0.8.5.post1 V0 xgrammar
# to silently fall back to Outlines (vllm#16723, vllm#16880), which then hangs
# the FSM compile on our 20-element offer_id enum.
#
# Equivalent enforcement is in Python post-processing in `score_partitions`:
#   - "exactly 4 offers" via `picked[:4]` + catalog backfill loop
#   - "reason ≤ 200 chars" via `reason[:200]`
#   - "rank ∈ {1,2,3,4}" via `enumerate(picked, start=1)`
OFFERS_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "required": ["offers"],
    "additionalProperties": False,
    "properties": {
        "offers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["rank", "offer_id", "reason"],
                "additionalProperties": False,
                "properties": {
                    "rank": {"type": "integer"},
                    "offer_id": {"type": "string", "enum": OFFER_IDS},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}
print(f"[inference] catalog has {len(OFFER_IDS)} offers")

# COMMAND ----------
# MAGIC %md ## Output schema and closure-captured constants

# COMMAND ----------
OUTPUT_SCHEMA: Final[StructType] = StructType(
    [
        StructField("customer_id", StringType(), nullable=False),
        StructField("run_date", DateType(), nullable=False),
        StructField("rank", IntegerType(), nullable=False),
        StructField("offer_id", StringType(), nullable=False),
        StructField("reason", StringType(), nullable=True),
        StructField("model_name", StringType(), nullable=True),
        StructField("model_version", StringType(), nullable=True),
        StructField("scored_at", TimestampType(), nullable=True),
    ]
)

# `OFFER_IDS_SET` and the schema/model constants are captured into the UDF's
# closure as-is; PySpark serialises module-level globals into the executor
# process. No `_LOCAL` rebindings — earlier revisions had them but they're
# defensive padding for immutables.
OFFER_IDS_SET: Final[set[str]] = set(OFFER_IDS)


# COMMAND ----------
# MAGIC %md ## The per-partition UDF
# MAGIC
# MAGIC With `spark.python.worker.reuse=false`, each task gets a fresh Python
# MAGIC process and a fresh vLLM engine. Engine init is ~30s for 1.5B / ~60s
# MAGIC for 7B in eager mode. After that the entire partition is scored in
# MAGIC one `llm.chat()` call (vLLM batches internally).

# COMMAND ----------
def score_partitions(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Score one Spark partition of customers with a per-task vLLM engine.

    Args:
        iterator: Stream of pandas DataFrames with columns
            ``customer_id`` and ``profile_text``.

    Yields:
        Pandas DataFrames matching :data:`OUTPUT_SCHEMA`. Exactly 4 rows
        per input customer.
    """
    import os as _os
    import re as _re

    from vllm import LLM, SamplingParams

    _os.environ["HF_HOME"] = HF_HOME

    # Module-global cache so that if Spark replays this task (e.g., shuffle
    # retry within the same JVM, which can happen even with reuse=false in
    # some retry paths), we don't reload vLLM.
    cache: dict[str, LLM] = globals().setdefault("_LOYALTY_VLLM", {})
    llm: LLM | None = cache.get(MODEL_LOCAL_PATH)
    if llm is None:
        print(
            f"[score_partitions] constructing vLLM for {MODEL_LOCAL_PATH} "
            f"(mode={INFERENCE_MODE}, backend={GUIDED_DECODING_BACKEND})",
            flush=True, 
        )
        llm = LLM(
            model=MODEL_LOCAL_PATH,
            dtype="bfloat16",
            gpu_memory_utilization=0.70,
            max_model_len=2048,
            enable_prefix_caching=True,
            enforce_eager=True,
            tensor_parallel_size=1,
            # Engine-level guided-decoding backend. vLLM 0.8.5 V0 requires the
            # backend to be fixed at construction (vllm#15762); xgrammar is
            # the only backend that handles our 20-element offer_id enum
            # without hanging (vllm PRs #15594, #15878).
            guided_decoding_backend=GUIDED_DECODING_BACKEND,
        )
        cache[MODEL_LOCAL_PATH] = llm
        print("[score_partitions] vLLM ready", flush=True)

    # Lazy import — only needed in guided mode, and importing it forces
    # vLLM's outlines/xgrammar symbol resolution which we want to avoid as
    # a precaution when guidance is off.
    if INFERENCE_MODE == "guided_json":
        from vllm.sampling_params import GuidedDecodingParams

    system_msg: str = (
        "You are a loyalty marketing strategist for Tractor Supply Company. "
        "Given a customer's loyalty profile and a catalog of offers, return "
        "the four offers most likely to drive incremental purchase, ranked "
        "1 (best) to 4. Each `offer_id` MUST come from the provided catalog. "
        "Respond with JSON ONLY, no commentary."
    )

    def _build_prompt(profile: str) -> list[dict[str, str]]:
        """Build the chat-message list for one customer.

        Catalog goes first (stable prefix for vLLM's prefix cache); the
        per-customer profile is last so the cache-miss suffix is minimal.
        """
        return [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": (
                    f"Available offers:\n{OFFER_SUMMARY}\n\n"
                    'Respond with JSON: '
                    '{"offers":[{"rank":1,"offer_id":"...","reason":"..."},...x4]}'
                    f"\n\nCustomer profile:\n{profile}"
                ),
            },
        ]

    # Sized to bound regex backtracking on pathological model outputs. A
    # well-formed JSON response is ~300-400 chars; 4096 gives a 10× safety
    # margin without inviting catastrophic re-scans.
    _JSON_EXTRACT_LIMIT: int = 4096

    def _extract_json(text: str) -> dict[str, Any]:
        """Pull the first JSON object out of model output, robustly.

        Tries the whole string first (cheap on well-formed output), then
        falls back to a length-bounded non-greedy regex scan. Returns `{}`
        on any failure — downstream backfills missing offers from the
        catalog so the output table always has 4 rows per customer.
        """
        stripped: str = text.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        # Length-bounded non-greedy fallback. Avoids the greedy `{.*}` trap
        # of matching first `{` to last `}` across the whole output.
        sample: str = text[:_JSON_EXTRACT_LIMIT]
        m = _re.search(r"\{.*?\}", sample, flags=_re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}

    model_version: str = HF_MODEL.split("/")[-1]
    run_date_obj: date = datetime.strptime(RUN_DATE, "%Y-%m-%d").date()
    fallback_ids: list[str] = sorted(OFFER_IDS_SET)

    try:
        for pdf in iterator:
            if pdf.empty:
                continue
            customer_ids: list[str] = pdf["customer_id"].tolist()
            profiles: list[str] = pdf["profile_text"].tolist()
            scored_at: datetime = datetime.now(timezone.utc)

            sampling_kwargs: dict[str, Any] = {
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 320,
            }
            if INFERENCE_MODE == "guided_json":
                sampling_kwargs["guided_decoding"] = GuidedDecodingParams(json=OFFERS_JSON_SCHEMA)
            sampling: SamplingParams = SamplingParams(**sampling_kwargs)
            prompts: list[list[dict[str, str]]] = [_build_prompt(p) for p in profiles]
            results = llm.chat(prompts, sampling, use_tqdm=False)

            records: list[dict[str, object]] = []
            for cid, output in zip(customer_ids, results, strict=True):
                raw_text: str = output.outputs[0].text if output.outputs else ""
                payload: dict[str, Any] = _extract_json(raw_text)
                raw_offers: list[dict[str, Any]] = list(payload.get("offers", []))
                picks_this: list[tuple[str, str]] = []
                seen: set[str] = set()
                for entry in raw_offers:
                    if not isinstance(entry, dict):
                        continue
                    oid: str = str(entry.get("offer_id", ""))
                    if oid not in OFFER_IDS_SET or oid in seen:
                        continue
                    reason_val: str = str(entry.get("reason", ""))[:200]
                    picks_this.append((oid, reason_val))
                    seen.add(oid)
                    if len(picks_this) == 4:
                        break
                # Backfill to 4. Reason gets a recognizable prefix so SQL
                # consumers can filter `WHERE reason LIKE 'fallback:%'`.
                for fid in fallback_ids:
                    if len(picks_this) == 4:
                        break
                    if fid not in seen:
                        picks_this.append((fid, "fallback: model response was incomplete"))
                        seen.add(fid)
                for position, (offer_id_val, reason_val) in enumerate(picks_this, start=1):
                    records.append(
                        {
                            "customer_id": cid,
                            "run_date": run_date_obj,
                            "rank": position,
                            "offer_id": offer_id_val,
                            "reason": reason_val,
                            "model_name": HF_MODEL,
                            "model_version": model_version,
                            "scored_at": scored_at,
                        }
                    )
            if records:
                yield pd.DataFrame(records)
    finally:
        # Best-effort cleanup so retries on the same executor don't pile up
        # CUDA contexts. vLLM 0.8.x exposes `LLM.shutdown()`; if unavailable,
        # falling back to `del llm` + `torch.cuda.empty_cache()` clears the
        # KV cache. Either way the process is about to exit (worker.reuse=
        # false), but this gets a head start on freeing VRAM.
        try:
            import gc as _gc
            import torch as _torch  # type: ignore[import-not-found]
            shutdown_fn = getattr(llm, "shutdown", None)
            if callable(shutdown_fn):
                shutdown_fn()
            cache.pop(MODEL_LOCAL_PATH, None)
            del llm
            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            print("[score_partitions] vLLM cleanup OK", flush=True)
        except Exception as cleanup_err:  # noqa: BLE001 — cleanup must never raise
            print(f"[score_partitions] cleanup warning: {cleanup_err}", flush=True)


# COMMAND ----------
# MAGIC %md ## Run the UDF and append the results

# COMMAND ----------
results_df: DataFrame = to_score_df.repartition(TARGET_PARTITIONS).mapInPandas(
    score_partitions, schema=OUTPUT_SCHEMA
)

(
    results_df.write.mode("append")
    .partitionBy("run_date")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.offer_recommendations")
)

final_count: int = (
    spark.table(f"{CATALOG}.{SCHEMA}.offer_recommendations")
    .where(F.col("run_date") == F.to_date(F.lit(RUN_DATE)))
    .count()
)
print(f"[inference] offer_recommendations rows for {RUN_DATE}: {final_count:,}")
