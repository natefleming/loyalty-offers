# Loyalty Offers — GPU Batch Inference DAB

A Databricks Asset Bundle that demonstrates **large-scale batch LLM inference** with **Qwen2.5-1.5B-Instruct** (with one-flag escalation to 7B) on **Azure Databricks** GPU compute. The pipeline takes a customer's loyalty profile and ranks the top-4 offers from a curated catalog of 20 loyalty offers.

- **Target cloud**: Azure Databricks
- **Bundle engine**: `direct` (Databricks CLI ≥ 0.279.0)
- **Cluster**: DAB-provisioned top-level cluster (`resources.clusters.gpu_inference`), **fixed `num_workers`** (no autoscale — see "Why fixed workers" below), Azure spot with on-demand fallback. The cluster survives between runs (it's an all-purpose cluster bound by `existing_cluster_id`); `autotermination_minutes: 10` shuts it down ~10 minutes after the last activity.
- **VM SKU**: `Standard_NC24ads_A100_v4` (1× NVIDIA A100 80 GB per worker)
- **Inference engine**: **vLLM 0.8.5.post1** with **xgrammar** guided-decoding backend in `**guided_json`** mode, installed at cluster boot via `scripts/init_vllm.sh`. This combination is the empirical winner of a 6-variant May 2026 matrix on this hardware — see `notebooks/03_batch_inference.py` for the per-knob rationale.
- **Model**: **Qwen2.5-1.5B-Instruct** (default for both dev and prod). 1.5B + guided_json hits the 8h SLO on 30M rows at ~$1k/run on Azure spot with ~3% fallback. 7B + guided_json costs ~4× more for the same SLO at 0% fallback — switch via `--var hf_model=Qwen/Qwen2.5-7B-Instruct` if needed.
- **UC home**: `retail_consumer_goods.tractor_supply` (override via `--var catalog=... --var schema=...`)
- **Scale & SLO**: three targets ship out of the box:
  - `**dev`**: 10 k rows, 4 GPU workers, ~13 min, ~$2/run on Azure spot. Iteration loop.
  - `**staging**`: 1 M rows, 15 GPU workers, ~6 h, ~$135/run on Azure spot. Load-test the full pipeline before committing 30 M GPU spend; ship intermediate output to downstream integration testing.
  - `**prod**`: 30 M rows, 90 GPU workers. Cost ~$1,000-1,200/run on Azure spot. Double `num_workers` to ~180 for a 4 h SLO at ~$2k/run.

---

## Why this Azure SKU

`Standard_NC24ads_A100_v4` was selected because:

1. **VRAM headroom**. Qwen2.5-7B in bf16 takes ~14 GB; an A100 80 GB leaves ~60 GB for vLLM's KV cache, so we run `gpu_memory_utilization=0.90` and `max_model_len=2048` without OOM.
2. **Single-GPU workers**. Avoids `tensor_parallel_size > 1`, which has NVLink/PCIe overhead and complicates Spark scheduling.
3. **Price/perf**. ~$3.67/hr on-demand, ~$1.50/hr on Azure spot — best $/token for Qwen-7B class models.
4. **Azure spot availability**. Strong A100 capacity in East US 2, West US 2, West Europe, North Europe, and Southeast Asia.

---

## Why vLLM, and why an init script

For 30 M customers in 4–8 h on this fleet we need ~80–150 customers/s/GPU. Plain `transformers.pipeline()` tops out around 5–8 customers/s/GPU on A100. vLLM closes the gap with:

- **Continuous batching** — requests are mixed at the token level, so the GPU is never idle waiting on the longest sequence.
- **Prefix caching** — every prompt shares the same ~700-token offer catalog. vLLM caches that prefix's KV-state and reuses it.
- **Guided JSON decoding** — the model can only emit tokens that match the JSON schema, so hallucinated `offer_id`s are impossible and every output parses cleanly. No regex extractor, no malformed-JSON fallback.

vLLM is installed via `scripts/init_vllm.sh`, referenced from `resources/cluster.yml` as a cluster init script. Init scripts run **before** the Spark/Python kernel boots, which avoids the kernel-restart race that `%pip install vllm` from a notebook causes.

---

## Layout

```
loyalty-offers/
├── databricks.yml              # bundle root: variables, targets, engine: direct
├── resources/
│   ├── cluster.yml             # DAB-provisioned GPU cluster (fixed workers, init script)
│   ├── job.yml                 # job + 4 tasks, attach to the cluster above
│   ├── schema.yml              # UC schema lifecycle
│   └── volume.yml              # HF model cache volume
├── notebooks/
│   ├── 01_provision.py         # create tables (with liquid clustering), seed offer_catalog
│   ├── 00_smoke_inference_engine.py  # optional pre-flight smoke (run_smoke_tests=true)
│   ├── 02_synthesize_customers.py
│   ├── 03_batch_inference.py   # vLLM + mapInPandas + append
│   └── 04_verify.py
├── scripts/
│   ├── init_vllm.sh            # cluster init script that installs vLLM at boot
│   ├── bootstrap_secret.sh     # one-time: HF token → secret scope
│   └── dev_smoke_vllm.py       # interactive-only vLLM smoke (attach to a GPU driver cluster)
├── data/
│   └── offer_catalog.json
├── tests/
│   └── test_smoke.py           # local (non-Spark, non-GPU) tests
├── pyproject.toml              # mypy strict + ruff
├── requirements.txt            # local-dev only; cluster deps come from init_vllm.sh
└── .gitignore
```

---

## Prerequisites

1. Databricks CLI ≥ 0.279.0 (`databricks --version`)
2. A Databricks CLI **profile** for the target workspace. Create with:
  ```bash
   databricks auth login --host https://<your-workspace>.cloud.databricks.com
  ```
   The login command will prompt for a profile name (any string you like).
3. A HuggingFace token with read access to `Qwen/Qwen2.5-7B-Instruct` exported as `HF_TOKEN`
4. `jq` installed (the bootstrap script uses it)

---

## Quickstart

```bash
# 1. Stash HF token in a Databricks secret scope (one-time per workspace)
HF_TOKEN=hf_xxx PROFILE=<your-profile> bash scripts/bootstrap_secret.sh

# 2. Validate
databricks bundle validate -t dev -p <your-profile>

# 3. Deploy (creates schema, volume, cluster, job, and uploads the init script)
databricks bundle deploy -t dev -p <your-profile>

# 4. Run end-to-end (10 k customers, ~5–10 min)
databricks bundle run loyalty_offers_batch_inference -t dev -p <your-profile>
```

When the run completes the cluster autoterminates within ~10 minutes. To scale up for the real workload:

```bash
databricks bundle deploy -t prod -p <your-profile>
databricks bundle run   loyalty_offers_batch_inference -t prod -p <your-profile>
```

The `prod` target sets `customers_target=30_000_000`, `num_workers=25`, and pins `hf_model=Qwen/Qwen2.5-7B-Instruct`.

> **Note on portability**: this bundle does **not** hardcode a profile name or any user's identity. The `prod` target uses `mode: production`, which auto-resolves the deploy root to `/Workspace/Users/${workspace.current_user.userName}/.bundle/...` for whoever runs `bundle deploy`. The cluster's `owner` tag is similarly resolved at deploy time.

---

## Configurable variables


| Variable                            | dev                           | staging                      | prod                         | Purpose                                                                                                                                                    |
| ----------------------------------- | ----------------------------- | ---------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `catalog`                           | `retail_consumer_goods`       | `retail_consumer_goods`      | `retail_consumer_goods`      | UC catalog                                                                                                                                                 |
| `schema`                            | `tractor_supply`              | `tractor_supply`             | `tractor_supply`             | UC schema                                                                                                                                                  |
| `customers_target`                  | `10_000`                      | `1_000_000`                  | `30_000_000`                 | Synthetic customer count                                                                                                                                   |
| `num_workers`                       | `4`                           | `15`                         | `90`                         | Fixed GPU worker count                                                                                                                                     |
| `gpu_node_type`                     | `Standard_NC24ads_A100_v4`    | same                         | same                         | Worker VM SKU                                                                                                                                              |
| `cpu_driver_node_type`              | `Standard_DS4_v2`             | same                         | same                         | Driver VM SKU (no GPU)                                                                                                                                     |
| `hf_model`                          | `Qwen/Qwen2.5-1.5B-Instruct`  | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model ID. 1.5B is the cost-optimal default; flip to 7B if 0% fallback is required.                                                             |
| Job param `inference_mode`          | `guided_json`                 | same                         | same                         | vLLM constrained-output mode. `guided_json` is the empirical winner. `none` (post-hoc validation only, ~17-28% fallback) is kept as a baseline comparator. |
| Job param `guided_decoding_backend` | `xgrammar`                    | same                         | same                         | Only backend that handles the 20-element `offer_id` enum without hanging on this DBR/vLLM combo.                                                           |
| `hf_secret_scope` / `hf_secret_key` | `loyalty-offers` / `HF_TOKEN` | same                         | same                         | Where the HF token lives                                                                                                                                   |


Override any of them with `--var key=value` on the CLI.

---

## Pipeline (4 tasks)

1. `**provision`** — `CREATE CATALOG/SCHEMA/TABLE IF NOT EXISTS` with **liquid clustering** on `(run_date, customer_id)`, MERGE-load `offer_catalog`, set the `run_date` task value.
2. `**synthesize`** — `spark.range(N)` + deterministic transforms produce `customers_target` rows, MERGE into `customers`.
3. `**inference**` (depends on `synthesize`, `max_retries: 2`) — anti-join against `offer_recommendations` to find unscored customers, repartition to ~50 k rows per task, `mapInPandas` runs **vLLM** with guided-decoding JSON schema to emit 4 ranked offers per customer, **append** to `offer_recommendations`. The whole append is a single Delta transaction — if any task fails terminally the write rolls back and the next attempt re-scores. What retries actually preserve: the model snapshot in the Volume (saves the ~14 GB download, ~5 min per retry) and the cached vLLM engine on any executor whose process survived.
4. `**verify`** — asserts row counts, rank coverage, and that no hallucinated `offer_id` slipped through (which guided decoding makes impossible at the token level — verify is the belt to that suspenders).

---

## Failure recovery


| Failure mode                                                    | What happens                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| vLLM transient error / executor OOM                             | Spark task retry (`spark.task.maxFailures: 2`)                                                                                                                                                                                                                                                                                                                 |
| Inference task itself fails                                     | Job-level `max_retries: 2` re-runs the task. The Spark write is one atomic Delta transaction, so a failed attempt commits nothing and the retry re-scores everyone for `run_date` (the anti-join is therefore a no-op on the second try). What the retry *does* save: the ~14 GB model snapshot is already in the Volume cache, so cold-start is ~5 min faster |
| Whole job killed and you rerun a new run on the same `run_date` | The anti-join skips anyone already committed by an earlier *successful* attempt. This is the failure mode where the anti-join earns its keep                                                                                                                                                                                                                   |
| Bad output / want to redo                                       | `DELETE FROM offer_recommendations WHERE run_date = '<date>'` then re-run                                                                                                                                                                                                                                                                                      |


No Structured Streaming, no RocksDB state, no custom WAL. The pattern is: durable input + idempotent output + anti-join on restart.

---

## UC tables (in `<catalog>.<schema>`)

- `**customers*`* — synthetic loyalty profiles, `CLUSTER BY (run_date, customer_id)` (PK: `customer_id` + `run_date`)
- `**offer_catalog**` — 20 curated TSC loyalty offers (PK: `offer_id`)
- `**offer_recommendations**` — 4 rows per customer, `CLUSTER BY (run_date, customer_id)` (PK: `customer_id` + `run_date` + `rank`)
- Volume `**hf_model_cache**` — persistent HF model weights cache mounted as `HF_HOME`

---

## Verification SQL

```sql
-- Row count: 4 × customers_target
SELECT COUNT(*) FROM <catalog>.<schema>.offer_recommendations
WHERE run_date = current_date();

-- Every customer has exactly 4 distinct ranks 1..4
SELECT customer_id, COUNT(DISTINCT rank) AS ranks
FROM <catalog>.<schema>.offer_recommendations
WHERE run_date = current_date()
GROUP BY 1 HAVING ranks != 4;

-- No hallucinated offers (always zero rows when guided decoding is on)
SELECT r.offer_id
FROM <catalog>.<schema>.offer_recommendations r
LEFT JOIN <catalog>.<schema>.offer_catalog c USING(offer_id)
WHERE c.offer_id IS NULL;
```

---

## Cleanup

```bash
databricks bundle destroy -t dev  -p <your-profile> --auto-approve
databricks bundle destroy -t prod -p <your-profile> --auto-approve
```

The HF secret scope is **not** destroyed automatically — remove with `databricks secrets delete-scope loyalty-offers -p <your-profile>` if desired.
