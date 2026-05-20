# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Verify
# MAGIC
# MAGIC Post-inference correctness checks. Fails loudly via `AssertionError` so a green
# MAGIC run means the output table is structurally sound.
# MAGIC
# MAGIC **Checks**:
# MAGIC 1. Row count = `customers_target * 4`
# MAGIC 2. Every customer has ranks {1, 2, 3, 4} (no duplicates, no gaps)
# MAGIC 3. Every `offer_id` exists in `offer_catalog` (no hallucinations)
# MAGIC 4. Logs throughput (rows / sec, derived from `scored_at` min/max)

# COMMAND ----------
from __future__ import annotations

from typing import Final

from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F

# COMMAND ----------
dbutils.widgets.text("catalog", "retail_consumer_goods")
dbutils.widgets.text("schema", "tractor_supply")
dbutils.widgets.text("customers_target", "10000")

CATALOG: Final[str] = dbutils.widgets.get("catalog")
SCHEMA: Final[str] = dbutils.widgets.get("schema")
CUSTOMERS_TARGET: Final[int] = int(dbutils.widgets.get("customers_target"))
RUN_DATE: Final[str] = dbutils.jobs.taskValues.get(
    taskKey="provision", key="run_date", default="", debugValue="1970-01-01"
)
assert RUN_DATE, "run_date task value missing from upstream `provision` task"

spark: SparkSession = SparkSession.getActiveSession()  # type: ignore[assignment]
assert spark is not None
print(f"[verify] catalog={CATALOG} schema={SCHEMA} run_date={RUN_DATE} target={CUSTOMERS_TARGET}")

# COMMAND ----------
# MAGIC %md ## 1) Row count

# COMMAND ----------
rec_df = spark.table(f"{CATALOG}.{SCHEMA}.offer_recommendations").where(
    F.col("run_date") == F.to_date(F.lit(RUN_DATE))
)
row_count: int = rec_df.count()
expected: int = CUSTOMERS_TARGET * 4
print(f"[verify] rows={row_count:,} expected={expected:,}")
assert row_count == expected, f"row count mismatch: got {row_count}, expected {expected}"

# COMMAND ----------
# MAGIC %md ## 2) Every customer has exactly 4 distinct ranks

# COMMAND ----------
bad_customers: int = (
    rec_df.groupBy("customer_id")
    .agg(F.countDistinct("rank").alias("ranks"))
    .filter("ranks != 4")
    .count()
)
print(f"[verify] customers with incorrect rank coverage: {bad_customers}")
assert bad_customers == 0, f"{bad_customers} customers do not have ranks 1..4"

# COMMAND ----------
# MAGIC %md ## 3) Every offer_id is in the catalog

# COMMAND ----------
catalog_df = spark.table(f"{CATALOG}.{SCHEMA}.offer_catalog").select("offer_id")
unknown_offers: int = (
    rec_df.join(catalog_df, on="offer_id", how="left_anti").count()
)
print(f"[verify] unknown offer_ids: {unknown_offers}")
assert unknown_offers == 0, f"{unknown_offers} hallucinated offer_ids leaked through guided decoding"

# COMMAND ----------
# MAGIC %md ## 4) Throughput stats

# COMMAND ----------
stats_row: Row = rec_df.agg(
    F.min("scored_at").alias("first_score"),
    F.max("scored_at").alias("last_score"),
).collect()[0]
first_score = stats_row["first_score"]
last_score = stats_row["last_score"]
if first_score is not None and last_score is not None and last_score > first_score:
    elapsed_seconds: float = (last_score - first_score).total_seconds()
    rows_per_sec: float = row_count / max(elapsed_seconds, 1.0)
    print(
        f"[verify] window={first_score.isoformat()} -> {last_score.isoformat()} "
        f"({elapsed_seconds:.1f}s) throughput={rows_per_sec:,.1f} output_rows/s"
    )
else:
    print("[verify] scored_at window is degenerate; skipping throughput estimate")

print("[verify] OK")
