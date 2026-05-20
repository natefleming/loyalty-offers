#!/bin/bash
#
# Cluster init script: install vLLM into the Databricks Runtime ML GPU
# Python environment.
#
# Source of the version pins: github.com/ryancicak/databricks-ray-vllm — the
# only publicly verified end-to-end working combination on DBR 16.4 LTS ML GPU
# as of May 2026. The install order is non-obvious and load-bearing.
#
# Wiring:
#   This script is referenced from `resources/cluster.yml` via
#   `init_scripts.workspace.destination: ${workspace.file_path}/scripts/init_vllm.sh`.
#   `databricks bundle deploy` uploads it to the workspace automatically.
#
# Constraints:
#   - DBR 16.4 LTS ML GPU ships torch==2.6.0+cu124 and flash_attn==2.7.4.post1
#     pre-installed. We replace the pre-built flash_attn wheel with the one
#     explicitly compiled for cu12torch2.6 so the ABI lines up after vllm pulls
#     in its own torch transitively.
#   - flash-attn MUST install before vllm — vllm's resolver otherwise picks up
#     a flash_attn build that doesn't match torch and silently SIGKILLs Python
#     workers ~30 min into a run.
#   - transformers MUST install AFTER vllm — vllm 0.8.5.post1's resolver wants
#     transformers<4.51 which doesn't know about Qwen2.5; we override that with
#     `transformers<4.54.0` after vllm settles.
#   - numpy==1.26.4 pin prevents vllm from pulling numpy 2.x, which breaks
#     other pre-installed DBR ML packages.

set -euxo pipefail

PIP="/databricks/python3/bin/pip"
PY="/databricks/python3/bin/python"

# 1) Pre-built flash_attn wheel matching DBR 16.4's torch 2.6.0+cu124.
#    --force-reinstall: overwrite the pre-installed flash_attn.
#    --no-deps: do NOT yank torch with this install (the wheel already targets
#               the correct torch). This is the critical flag.
"$PIP" install --force-reinstall --no-cache-dir --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

# 2) vllm and friends, in one resolver pass so transformers/numpy land as
#    upper-bound pinned versions (not whatever vllm prefers).
"$PIP" install \
  "vllm==0.8.5.post1" \
  "transformers<4.54.0" \
  "numpy==1.26.4" \
  "hf_transfer>=0.1.8"

# 3) Fail-fast smoke check. Cluster boot fails here if any of these can't be
#    imported — much better than discovering it ~30 min into a run.
"$PY" -c "import vllm, torch, transformers, flash_attn; \
  print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'flash_attn', flash_attn.__version__)"

echo "[init_vllm.sh] OK"
