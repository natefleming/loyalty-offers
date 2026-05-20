# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Provision
# MAGIC
# MAGIC Idempotently creates the Unity Catalog catalog, schema, and tables used by the
# MAGIC loyalty-offers batch inference pipeline, seeds the curated offer catalog, validates
# MAGIC the HuggingFace token, and exposes the current `run_date` as a job task value
# MAGIC consumed by the downstream tasks.
# MAGIC
# MAGIC **Inputs (job parameters)**: `catalog`, `schema`, `hf_secret_scope`, `hf_secret_key`.
# MAGIC
# MAGIC **Side effects**:
# MAGIC - `CREATE CATALOG IF NOT EXISTS <catalog>` (no-op if already managed)
# MAGIC - `CREATE SCHEMA IF NOT EXISTS <catalog>.<schema>` (DAB also creates this)
# MAGIC - `CREATE TABLE IF NOT EXISTS customers | offer_catalog | offer_recommendations`
# MAGIC - MERGE-loads `offer_catalog` from the bundle's `data/offer_catalog.json` (loaded
# MAGIC   from the workspace files alongside this notebook)
# MAGIC - Sets task value `run_date` (ISO date string) for downstream tasks

# COMMAND ----------
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Final

from huggingface_hub import HfApi
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    DateType,
    DecimalType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# COMMAND ----------
# MAGIC %md ## Read job parameters

# COMMAND ----------
dbutils.widgets.text("catalog", "retail_consumer_goods")
dbutils.widgets.text("schema", "tractor_supply")
dbutils.widgets.text("hf_secret_scope", "loyalty-offers")
dbutils.widgets.text("hf_secret_key", "HF_TOKEN")

CATALOG: Final[str] = dbutils.widgets.get("catalog")
SCHEMA: Final[str] = dbutils.widgets.get("schema")
HF_SECRET_SCOPE: Final[str] = dbutils.widgets.get("hf_secret_scope")
HF_SECRET_KEY: Final[str] = dbutils.widgets.get("hf_secret_key")

print(f"[provision] catalog={CATALOG} schema={SCHEMA}")

# COMMAND ----------
# MAGIC %md ## Validate the HuggingFace token

# COMMAND ----------
def _validate_hf_token() -> None:
    """Call HF whoami to confirm the secret-scoped token is valid before the run continues.

    Raises:
        RuntimeError: if the HF token is missing, malformed, or rejected.
    """
    token: str = dbutils.secrets.get(scope=HF_SECRET_SCOPE, key=HF_SECRET_KEY)
    if not token:
        raise RuntimeError(
            f"HF token at {HF_SECRET_SCOPE}/{HF_SECRET_KEY} is empty. "
            "Run scripts/bootstrap_secret.sh and re-deploy."
        )
    info: dict[str, object] = HfApi().whoami(token=token)
    print(f"[provision] HF whoami: name={info.get('name')!r} type={info.get('type')!r}")


_validate_hf_token()

# COMMAND ----------
# MAGIC %md ## Create catalog, schema, and tables (idempotent)

# COMMAND ----------
spark: SparkSession = SparkSession.getActiveSession()  # type: ignore[assignment]
assert spark is not None, "Spark session must be active in a Databricks notebook"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

CUSTOMERS_SCHEMA: Final[StructType] = StructType(
    [
        StructField("customer_id", StringType(), nullable=False),
        StructField("run_date", DateType(), nullable=False),
        StructField("loyalty_tier", StringType(), nullable=True),
        StructField("years_member", IntegerType(), nullable=True),
        StructField("ytd_spend", DecimalType(10, 2), nullable=True),
        StructField("top_categories", ArrayType(StringType()), nullable=True),
        StructField("last_purchase_days_ago", IntegerType(), nullable=True),
        StructField("state", StringType(), nullable=True),
        StructField("profile_text", StringType(), nullable=True),
    ]
)

OFFER_CATALOG_SCHEMA: Final[StructType] = StructType(
    [
        StructField("offer_id", StringType(), nullable=False),
        StructField("name", StringType(), nullable=True),
        StructField("description", StringType(), nullable=True),
        StructField("category", StringType(), nullable=True),
        StructField("audience_hint", StringType(), nullable=True),
    ]
)

OFFER_RECOMMENDATIONS_SCHEMA: Final[StructType] = StructType(
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


def _ensure_table(
    name: str,
    schema: StructType,
    cluster_cols: list[str] | None = None,
) -> None:
    """Create a managed Delta table with the given schema if it does not yet exist.

    We use **liquid clustering** (`CLUSTER BY`) rather than Hive-style
    partitioning. Liquid clustering treats the cluster keys as a soft hint
    and reorganises files in the background — better for the access pattern
    here, where the hot lookup is "today's `run_date` + a `customer_id`
    filter" (the anti-join in `03_batch_inference.py`).

    Args:
        name: Fully-qualified table name (``catalog.schema.table``).
        schema: PySpark ``StructType`` declaring the columns.
        cluster_cols: Optional liquid-clustering columns.
    """
    cols_sql: str = ",\n  ".join(
        f"{f.name} {f.dataType.simpleString().upper()} {'NOT NULL' if not f.nullable else ''}".strip()
        for f in schema.fields
    )
    cluster_sql: str = (
        f"CLUSTER BY ({', '.join(cluster_cols)})" if cluster_cols else ""
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {name} (
          {cols_sql}
        ) USING DELTA {cluster_sql}
        """
    )
    print(f"[provision] table ready: {name}")


_ensure_table(
    f"{CATALOG}.{SCHEMA}.customers",
    CUSTOMERS_SCHEMA,
    ["run_date", "customer_id"],
)
_ensure_table(f"{CATALOG}.{SCHEMA}.offer_catalog", OFFER_CATALOG_SCHEMA)
_ensure_table(
    f"{CATALOG}.{SCHEMA}.offer_recommendations",
    OFFER_RECOMMENDATIONS_SCHEMA,
    ["run_date", "customer_id"],
)

# COMMAND ----------
# MAGIC %md ## Seed `offer_catalog` from the bundle's JSON file

# COMMAND ----------
def _load_offer_catalog_path() -> Path:
    """Locate `data/offer_catalog.json` next to this notebook in the deployed bundle.

    Returns:
        Path to the offer catalog JSON file.

    Raises:
        FileNotFoundError: if the file cannot be located in any expected location.
    """
    notebook_path: str = (
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    )
    notebook_dir: Path = Path(notebook_path).parent
    candidates: list[Path] = [
        Path("/Workspace") / notebook_dir.relative_to("/") / ".." / "data" / "offer_catalog.json",
        Path("/Workspace") / notebook_dir.relative_to("/").parent / "data" / "offer_catalog.json",
    ]
    for candidate in candidates:
        resolved: Path = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"offer_catalog.json not found in any of: {candidates}")


offers_path: Path = _load_offer_catalog_path()
print(f"[provision] loading offer catalog from {offers_path}")
with offers_path.open() as handle:
    offers: list[dict[str, str]] = json.load(handle)

offers_df = spark.createDataFrame(offers, schema=OFFER_CATALOG_SCHEMA)
offers_df.createOrReplaceTempView("_offers_staged")
spark.sql(
    f"""
    MERGE INTO {CATALOG}.{SCHEMA}.offer_catalog t
    USING _offers_staged s
    ON t.offer_id = s.offer_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """
)
print(f"[provision] merged {len(offers)} offers into offer_catalog")

# COMMAND ----------
# MAGIC %md ## Publish `run_date` task value

# COMMAND ----------
RUN_DATE: Final[str] = date.today().isoformat()
dbutils.jobs.taskValues.set(key="run_date", value=RUN_DATE)
print(f"[provision] run_date={RUN_DATE}")
