# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Synthesize customers
# MAGIC
# MAGIC Materializes `customers_target` synthetic Tractor Supply loyalty-club customers into
# MAGIC `<catalog>.<schema>.customers`. The synthesis is **deterministic** (seeded by row id)
# MAGIC so the same `customers_target` and `run_date` always produce the same rows — that lets
# MAGIC reruns be idempotent.
# MAGIC
# MAGIC **Inputs (job parameters)**: `catalog`, `schema`, `customers_target`.
# MAGIC **Inputs (task values)**: `run_date` (from `provision`).
# MAGIC
# MAGIC **Output**: `<catalog>.<schema>.customers` with `customers_target` rows for the
# MAGIC current `run_date`.

# COMMAND ----------
from __future__ import annotations

from typing import Final

from pyspark.sql import Column, DataFrame, SparkSession
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
print(f"[synthesize] customers_target={CUSTOMERS_TARGET} run_date={RUN_DATE}")

spark: SparkSession = SparkSession.getActiveSession()  # type: ignore[assignment]
assert spark is not None

# COMMAND ----------
# MAGIC %md ## Deterministic synthesis using `spark.range` + hash-based picks

# COMMAND ----------
TIERS: Final[list[str]] = ["bronze", "silver", "gold", "platinum"]
STATES: Final[list[str]] = [
    "TX", "OH", "PA", "TN", "KY", "GA", "NC", "MO", "IN", "AL",
    "VA", "OK", "WV", "AR", "SC", "WI", "IA", "MI", "KS", "FL",
]
CATEGORIES: Final[list[str]] = [
    "feed", "fence", "apparel", "tools", "seasonal", "membership",
]


def _pick(col: Column, options: list[str]) -> Column:
    """Deterministically map a hash column to one of `options`.

    Args:
        col: Integer column to use as the picker (typically `abs(hash(...))`).
        options: List of choices.

    Returns:
        A `Column` of `StringType` containing one of the `options` per row.
    """
    n: int = len(options)
    return F.element_at(F.array([F.lit(o) for o in options]), (col % n).cast("int") + 1)


def _build_customers(target_rows: int, run_date: str) -> DataFrame:
    """Generate a deterministic synthetic loyalty-customer DataFrame.

    Args:
        target_rows: Number of distinct customers to materialize.
        run_date: ISO date string to stamp on every row.

    Returns:
        A `DataFrame` matching the `customers` table schema.
    """
    base: DataFrame = spark.range(0, target_rows).withColumnRenamed("id", "row_id")
    h_tier: Column = F.abs(F.xxhash64(F.col("row_id"), F.lit("tier"))).cast("long")
    h_state: Column = F.abs(F.xxhash64(F.col("row_id"), F.lit("state"))).cast("long")
    h_cat_a: Column = F.abs(F.xxhash64(F.col("row_id"), F.lit("cat-a"))).cast("long")
    h_cat_b: Column = F.abs(F.xxhash64(F.col("row_id"), F.lit("cat-b"))).cast("long")
    h_spend: Column = F.abs(F.xxhash64(F.col("row_id"), F.lit("spend"))).cast("long")
    h_member: Column = F.abs(F.xxhash64(F.col("row_id"), F.lit("member"))).cast("long")
    h_last: Column = F.abs(F.xxhash64(F.col("row_id"), F.lit("last"))).cast("long")

    customer_id: Column = F.concat(F.lit("TSC-"), F.lpad(F.col("row_id").cast("string"), 9, "0"))
    loyalty_tier: Column = _pick(h_tier, TIERS)
    state: Column = _pick(h_state, STATES)
    cat_a: Column = _pick(h_cat_a, CATEGORIES)
    cat_b: Column = _pick(h_cat_b, CATEGORIES)
    top_categories: Column = F.array_distinct(F.array(cat_a, cat_b))
    years_member: Column = (h_member % 12).cast("int")
    ytd_spend: Column = ((h_spend % 250_000) / F.lit(100)).cast("decimal(10,2)")
    last_purchase_days_ago: Column = (h_last % 365).cast("int")

    profile_text: Column = F.concat_ws(
        " | ",
        F.concat(F.lit("tier="), loyalty_tier),
        F.concat(F.lit("years_member="), years_member.cast("string")),
        F.concat(F.lit("ytd_spend=$"), ytd_spend.cast("string")),
        F.concat(F.lit("state="), state),
        F.concat(F.lit("top_categories="), F.concat_ws(",", top_categories)),
        F.concat(F.lit("last_purchase_days_ago="), last_purchase_days_ago.cast("string")),
    )

    return base.select(
        customer_id.alias("customer_id"),
        F.to_date(F.lit(run_date)).alias("run_date"),
        loyalty_tier.alias("loyalty_tier"),
        years_member.alias("years_member"),
        ytd_spend.alias("ytd_spend"),
        top_categories.alias("top_categories"),
        last_purchase_days_ago.alias("last_purchase_days_ago"),
        state.alias("state"),
        profile_text.alias("profile_text"),
    )


customers_df: DataFrame = _build_customers(CUSTOMERS_TARGET, RUN_DATE)
customers_df.createOrReplaceTempView("_customers_staged")

# COMMAND ----------
# MAGIC %md ## MERGE into `customers` (idempotent on customer_id + run_date)

# COMMAND ----------
spark.sql(
    f"""
    MERGE INTO {CATALOG}.{SCHEMA}.customers t
    USING _customers_staged s
    ON  t.customer_id = s.customer_id
    AND t.run_date    = s.run_date
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """
)

row_count: int = spark.table(f"{CATALOG}.{SCHEMA}.customers").where(
    F.col("run_date") == F.to_date(F.lit(RUN_DATE))
).count()
print(f"[synthesize] customers for {RUN_DATE}: {row_count:,}")
assert row_count == CUSTOMERS_TARGET, f"expected {CUSTOMERS_TARGET} rows, got {row_count}"
