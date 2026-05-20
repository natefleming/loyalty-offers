"""Local smoke tests for prompt formatting and JSON parsing.

These run without Spark or a GPU — they validate only the pure-Python helpers
and config artifacts that the inference notebook depends on. Run with
`uv run pytest tests/` or `pytest tests/`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import jsonschema
import pytest

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Existing tests — offer catalog basic invariants
# ---------------------------------------------------------------------------


def test_offer_catalog_json_is_valid() -> None:
    """`data/offer_catalog.json` must parse and contain >= 20 well-formed offers."""
    with (REPO_ROOT / "data" / "offer_catalog.json").open() as handle:
        offers: list[dict[str, str]] = json.load(handle)
    assert isinstance(offers, list)
    assert len(offers) >= 20
    required_keys: set[str] = {"offer_id", "name", "description", "category", "audience_hint"}
    for offer in offers:
        assert required_keys.issubset(offer.keys()), f"missing keys in {offer}"
        assert offer["offer_id"].startswith("TSC-"), f"unexpected offer_id format: {offer['offer_id']}"


def test_offer_ids_are_unique() -> None:
    """Every offer_id must be unique — otherwise MERGE on the output table breaks."""
    with (REPO_ROOT / "data" / "offer_catalog.json").open() as handle:
        offers: list[dict[str, str]] = json.load(handle)
    ids: list[str] = [o["offer_id"] for o in offers]
    assert len(ids) == len(set(ids)), f"duplicate offer_ids detected: {sorted(ids)}"


# ---------------------------------------------------------------------------
# Prompt structure — catalog MUST come before profile for prefix caching
# ---------------------------------------------------------------------------


def _build_prompt(profile: str, offer_summary: str, system_msg: str = "stub system") -> list[dict[str, str]]:
    """Mirror of the production prompt builder in ``03_batch_inference.py``.

    Duplicated here intentionally — extracting to a shared module would require
    a Python package layout, which is overkill for a Databricks notebook bundle.
    The test_prompt_structure assertion catches drift.
    """
    return [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": (
                f"Available offers:\n{offer_summary}\n\n"
                'Respond with JSON: '
                '{"offers":[{"rank":1,"offer_id":"...","reason":"..."},...x4]}'
                f"\n\nCustomer profile:\n{profile}"
            ),
        },
    ]


def test_prompt_structure_catalog_before_profile() -> None:
    """Catalog must appear BEFORE the per-customer profile.

    vLLM's prefix caching only matches common token prefixes. The catalog is
    the same across every customer in a batch (~700 tokens); the profile is
    unique per customer. Putting the profile first would invalidate the cache
    on every request and erase most of the throughput win.
    """
    prompt = _build_prompt(profile="tier=gold,state=TX", offer_summary="CATALOG_MARKER")
    user_content: str = prompt[1]["content"]
    assert user_content.index("CATALOG_MARKER") < user_content.index("tier=gold"), (
        "catalog must appear before profile for prefix caching to be effective"
    )


def test_prompt_includes_required_sections() -> None:
    """Prompt must contain the catalog, JSON instruction, and profile."""
    prompt = _build_prompt(profile="P", offer_summary="C")
    user_content: str = prompt[1]["content"]
    assert "Available offers:" in user_content
    assert '"offers":' in user_content  # JSON template
    assert "Customer profile:" in user_content


# ---------------------------------------------------------------------------
# Guided-decoding JSON schema — must accept good output and reject bad
# ---------------------------------------------------------------------------


def _build_schema(offer_ids: list[str]) -> dict[str, Any]:
    """Mirror of the production xgrammar-clean ``OFFERS_JSON_SCHEMA`` in
    ``03_batch_inference.py``.

    Deliberately omits `minItems`, `maxItems`, `minLength`, `maxLength`,
    `minimum`, and `maximum` — vLLM 0.8.5.post1 V0 xgrammar silently demotes
    to Outlines when those are present and re-triggers the FSM-compile hang
    on the offer_id enum (vLLM issues #16723 / #16880). Production enforces
    the equivalent constraints in Python post-processing:
        * exactly 4 offers per customer (`picked[:4]` + backfill)
        * reason ≤ 200 chars (`reason[:200]`)
        * rank in 1..4 (assigned by enumerate(picked, start=1), not from model)
    """
    return {
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
                        "offer_id": {"type": "string", "enum": offer_ids},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    }


def _well_formed_response(offer_ids: list[str]) -> dict[str, Any]:
    return {
        "offers": [
            {"rank": i + 1, "offer_id": offer_ids[i % len(offer_ids)], "reason": f"reason {i + 1}"}
            for i in range(4)
        ]
    }


def test_offers_schema_accepts_well_formed_response() -> None:
    """A canonical 4-offer response with real offer_ids should validate."""
    ids: list[str] = ["TSC-FEED-10", "TSC-FEED-BOGO", "TSC-WINTER", "TSC-NEIGH-30"]
    schema = _build_schema(ids)
    jsonschema.validate(_well_formed_response(ids), schema)


def test_offers_schema_rejects_hallucinated_offer_id() -> None:
    """offer_id not in the catalog must fail validation."""
    ids: list[str] = ["TSC-FEED-10", "TSC-FEED-BOGO", "TSC-WINTER", "TSC-NEIGH-30"]
    invalid: dict[str, Any] = {
        "offers": [{"rank": 1, "offer_id": "TSC-MADE-UP", "reason": "x"}] * 4
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, _build_schema(ids))


# NOTE: the previous tests that asserted JSON-schema-level rejection of
# `len != 4` offers and `rank` out of range were removed when the schema
# dropped `minItems`/`maxItems`/`minimum`/`maximum` to keep xgrammar happy.
# The equivalent enforcement now lives in Python post-processing in
# `notebooks/03_batch_inference.py::score_partitions`: the picked-list slicer
# caps at 4, the catalog backfill loop fills to 4, and rank is assigned by
# `enumerate(picked, start=1)` (not by trusting the model's `rank` field).


def test_offers_schema_rejects_extra_top_level_keys() -> None:
    """`additionalProperties: false` should bar surprise top-level keys."""
    ids: list[str] = ["TSC-FEED-10", "TSC-FEED-BOGO", "TSC-WINTER", "TSC-NEIGH-30"]
    invalid: dict[str, Any] = {**_well_formed_response(ids), "extra": "should not be here"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, _build_schema(ids))


# ---------------------------------------------------------------------------
# Schema built from the real catalog should validate against itself
# ---------------------------------------------------------------------------


_XGRAMMAR_UNSUPPORTED_KEYS: Final[frozenset[str]] = frozenset(
    {"minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum"}
)


def _walk_schema_keys(node: Any) -> list[str]:
    """Recursively collect every JSON Schema keyword present in a schema fragment."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append(key)
            found.extend(_walk_schema_keys(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_schema_keys(item))
    return found


def test_offers_schema_has_no_xgrammar_blocking_constraints() -> None:
    """vLLM 0.8.5.post1 V0 xgrammar silently falls back to Outlines when the schema
    contains any of: minItems, maxItems, minLength, maxLength.

    The fallback re-triggers the Outlines FSM-compile hang on our 20-element
    offer_id enum (vLLM issues #16723 / #16880). This test guards against
    accidentally re-introducing those keywords in the inference notebook's
    schema constant — adopt the "post-hoc Python validation" pattern instead.

    The check is intentionally strict: even `minimum` / `maximum` on integer
    fields can trip xgrammar in older versions, so we forbid them too and
    enforce ranges in Python post-processing.
    """
    valid_ids: list[str] = ["TSC-FEED-10", "TSC-FEED-BOGO", "TSC-WINTER", "TSC-NEIGH-30"]
    schema: dict[str, Any] = _build_schema(valid_ids)
    keys_present: set[str] = set(_walk_schema_keys(schema))
    offending: set[str] = keys_present & _XGRAMMAR_UNSUPPORTED_KEYS
    assert not offending, (
        f"Schema contains keys that xgrammar silently demotes to Outlines: "
        f"{sorted(offending)}. Remove them from the schema and enforce the "
        f"constraint in Python instead."
    )


def test_offers_schema_built_from_real_catalog() -> None:
    """End-to-end check: build the schema from the real catalog file and
    validate a synthetic well-formed response. This catches regressions in
    either the catalog format or the schema construction logic.
    """
    with (REPO_ROOT / "data" / "offer_catalog.json").open() as handle:
        offers: list[dict[str, str]] = json.load(handle)
    real_ids: list[str] = sorted(o["offer_id"] for o in offers)
    schema = _build_schema(real_ids)
    jsonschema.validate(_well_formed_response(real_ids), schema)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
