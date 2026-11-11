#!/usr/bin/env python3
"""Compare batch CROWN JSON results and write a wide CSV table."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


COMPARE_FIELDS = [
    "status",
    "certified",
    "plain_gap",
    "sci_gap",
    "abs_err_gap",
    "time_ms",
    "comm_bytes",
    "party1_exit",
    "party2_exit",
    "error",
]


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_").lower()
    return slug or "run"


def load_result_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"Not a batch result JSON file: {path}")
    records = payload["records"]
    if not isinstance(records, list):
        raise ValueError(f"Batch result records must be a list: {path}")
    return payload


def default_run_name(path: Path, payload: dict[str, Any]) -> str:
    config = payload.get("config", {})
    method = config.get("method") if isinstance(config, dict) else None
    if method:
        return _slug(str(method))
    return _slug(path.stem)


def build_comparison_rows(
    named_payloads: list[tuple[str, dict[str, Any]]],
) -> tuple[list[dict[str, object]], list[str]]:
    base_by_index: dict[int, dict[str, object]] = {}
    record_by_name: dict[str, dict[int, dict[str, object]]] = {}

    for name, payload in named_payloads:
        by_index: dict[int, dict[str, object]] = {}
        for raw in payload.get("records", []):
            if not isinstance(raw, dict):
                continue
            index = int(raw["index"])
            by_index[index] = raw
            base = base_by_index.setdefault(
                index,
                {
                    "index": index,
                    "label": raw.get("label", ""),
                    "pred": raw.get("pred", ""),
                },
            )
            if base.get("label", "") == "" and raw.get("label") is not None:
                base["label"] = raw.get("label", "")
            if base.get("pred", "") == "" and raw.get("pred") is not None:
                base["pred"] = raw.get("pred", "")
        record_by_name[name] = by_index

    fields = ["index", "label", "pred"]
    for name, _payload in named_payloads:
        fields.extend(f"{name}_{field}" for field in COMPARE_FIELDS)

    rows: list[dict[str, object]] = []
    for index in sorted(base_by_index):
        row = dict(base_by_index[index])
        for name, _payload in named_payloads:
            record = record_by_name[name].get(index, {})
            for field in COMPARE_FIELDS:
                row[f"{name}_{field}"] = record.get(field, "")
        rows.append(row)
    return rows, fields


def write_comparison_csv(
    out_path: Path, rows: list[dict[str, object]], fields: list[str]
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True, type=Path)
    p.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Optional column prefixes, one per input JSON.",
    )
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payloads = [load_result_json(path) for path in args.inputs]
    if args.names is not None and len(args.names) != len(args.inputs):
        raise SystemExit("--names must have the same count as --inputs")
    names = (
        [_slug(name) for name in args.names]
        if args.names is not None
        else [default_run_name(path, payload) for path, payload in zip(args.inputs, payloads)]
    )
    if len(set(names)) != len(names):
        raise SystemExit("Run names must be unique after normalization")

    rows, fields = build_comparison_rows(list(zip(names, payloads)))
    write_comparison_csv(args.out, rows, fields)
    print(f"[compare] inputs={len(args.inputs)} rows={len(rows)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
