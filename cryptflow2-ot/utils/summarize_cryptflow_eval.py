#!/usr/bin/env python3
"""Summarize Phase 7 CryptFlow2-CROWN raw evaluation records."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REL_ERROR_FLOOR = 1e-6
OK_STATUS = "ok"
TIMEOUT_STATUS = "timeout"
MEMORY_STATUS = "memory_limit"
SECURE_ERROR_STATUS = "secure_error"
PLAIN_ERROR_STATUS = "plain_error"
MISCLASSIFIED_STATUS = "plain_misclassified"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _status(record: dict[str, Any]) -> str:
    status = str(record.get("status") or "")
    if status:
        return status
    return str(record.get("resource_status") or "unknown")


def _blocked(status: str) -> bool:
    return status.startswith("blocked")


def _group_name(record: dict[str, Any]) -> str:
    group = record.get("plaintext_group")
    if group:
        return str(group)
    status = _status(record)
    if _blocked(status):
        return status
    return "unknown"


def _record_gap_abs_err(record: dict[str, Any]) -> float | None:
    plain_gap = _num(record.get("plain_gap"))
    secure_gap = _num(record.get("secure_gap"))
    if secure_gap is None:
        secure_gap = _num(record.get("sci_gap"))
    if plain_gap is None or secure_gap is None:
        return _num(record.get("gap_abs_err") if record.get("gap_abs_err") is not None else record.get("abs_err_gap"))
    return abs(secure_gap - plain_gap)


def _record_gap_rel_err(record: dict[str, Any], abs_err: float | None) -> float | None:
    if abs_err is None:
        return None
    plain_gap = _num(record.get("plain_gap"))
    if plain_gap is None:
        return _num(record.get("gap_rel_err"))
    return abs_err / max(abs(plain_gap), REL_ERROR_FLOOR)


def _field_abs_rel(
    record: dict[str, Any],
    plain_key: str,
    secure_key: str,
    abs_key: str,
    rel_key: str,
) -> tuple[float | None, float | None]:
    plain = _num(record.get(plain_key))
    secure = _num(record.get(secure_key))
    if plain is not None and secure is not None:
        abs_err = abs(secure - plain)
    else:
        abs_err = _num(record.get(abs_key))
    if abs_err is None:
        return None, None
    rel_err = _num(record.get(rel_key))
    if rel_err is None and plain is not None:
        rel_err = abs_err / max(abs(plain), REL_ERROR_FLOOR)
    return abs_err, rel_err


def _plain_certified(record: dict[str, Any]) -> bool | None:
    group = record.get("plaintext_group")
    if group == "robust":
        return True
    if group == "not-certified":
        return False
    gap = _num(record.get("plain_gap"))
    if gap is None:
        return None
    return gap >= 0.0


def _counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(_status(record) for record in records)
    blocked = sum(count for status, count in statuses.items() if _blocked(status))
    return {
        "n_attempted": len(records),
        "n_secure_ok": statuses.get(OK_STATUS, 0),
        "timeout_count": statuses.get(TIMEOUT_STATUS, 0),
        "memory_limit_count": statuses.get(MEMORY_STATUS, 0),
        "secure_error_count": statuses.get(SECURE_ERROR_STATUS, 0),
        "plain_error_count": statuses.get(PLAIN_ERROR_STATUS, 0),
        "misclassified_excluded": statuses.get(MISCLASSIFIED_STATUS, 0),
        "blocked_count": blocked,
        "statuses": dict(sorted(statuses.items())),
    }


def _reliability_key(record: dict[str, Any]) -> tuple[str, str, float]:
    return (
        str(record.get("dataset") or "unknown"),
        str(record.get("model") or "unknown"),
        float(record.get("eps") or 0.0),
    )


def _group_key(record: dict[str, Any]) -> tuple[str, str, float, str]:
    dataset, model, eps = _reliability_key(record)
    return dataset, model, eps, _group_name(record)


def _summarize_group(key: tuple[str, str, float, str], records: list[dict[str, Any]]) -> dict[str, Any]:
    dataset, model, eps, plaintext_group = key
    ok_records = [record for record in records if _status(record) == OK_STATUS]
    gap_abs: list[float] = []
    gap_rel: list[float] = []
    lower_abs: list[float] = []
    lower_rel: list[float] = []
    upper_abs: list[float] = []
    upper_rel: list[float] = []
    offline_time_s: list[float] = []
    raw_offline_time_s: list[float] = []
    online_time_s: list[float] = []
    total_time_s: list[float] = []
    offline_comm_mb: list[float] = []
    online_comm_mb: list[float] = []
    total_comm_mb: list[float] = []
    consistency_matches = 0

    for record in ok_records:
        abs_err = _record_gap_abs_err(record)
        rel_err = _record_gap_rel_err(record, abs_err)
        if abs_err is not None:
            gap_abs.append(abs_err)
        if rel_err is not None:
            gap_rel.append(rel_err)

        lb_abs, lb_rel = _field_abs_rel(record, "plain_lb", "sci_lb", "abs_err_lb", "rel_err_lb")
        ub_abs, ub_rel = _field_abs_rel(record, "plain_ub", "sci_ub", "abs_err_ub", "rel_err_ub")
        if lb_abs is not None:
            lower_abs.append(lb_abs)
        if lb_rel is not None:
            lower_rel.append(lb_rel)
        if ub_abs is not None:
            upper_abs.append(ub_abs)
        if ub_rel is not None:
            upper_rel.append(ub_rel)

        for target, keys, scale in [
            (offline_time_s, ("amortized_offline_time_ms", "offline_time_ms"), 1000.0),
            (raw_offline_time_s, ("raw_offline_time_ms", "offline_time_ms"), 1000.0),
            (online_time_s, ("online_time_ms",), 1000.0),
            (total_time_s, ("total_time_ms", "time_ms"), 1000.0),
            (offline_comm_mb, ("offline_comm_bytes",), 1_000_000.0),
            (online_comm_mb, ("online_comm_bytes",), 1_000_000.0),
            (total_comm_mb, ("total_comm_bytes", "comm_bytes"), 1_000_000.0),
        ]:
            value = next((_num(record.get(key_name)) for key_name in keys if _num(record.get(key_name)) is not None), None)
            if value is not None:
                target.append(value / scale)

        plain_cert = _plain_certified(record)
        secure_cert = record.get("certified")
        if plain_cert is not None and secure_cert is not None and bool(secure_cert) == plain_cert:
            consistency_matches += 1

    counts = _counts(records)
    n_secure_ok = counts["n_secure_ok"]
    row = {
        "dataset": dataset,
        "model": model,
        "eps": eps,
        "plaintext_group": plaintext_group,
        "n_records": len(records),
        **counts,
        "gap_mae": _mean(gap_abs),
        "gap_mre": _mean(gap_rel),
        "consistency_rate": (consistency_matches / n_secure_ok) if n_secure_ok else None,
        "offline_time_s_mean": _mean(offline_time_s),
        "raw_offline_time_s_mean": _mean(raw_offline_time_s),
        "online_time_s_mean": _mean(online_time_s),
        "total_time_s_mean": _mean(total_time_s),
        "offline_comm_mb_mean": _mean(offline_comm_mb),
        "online_comm_mb_mean": _mean(online_comm_mb),
        "total_comm_mb_mean": _mean(total_comm_mb),
        "lower_mae": _mean(lower_abs),
        "lower_mre": _mean(lower_rel),
        "upper_mae": _mean(upper_abs),
        "upper_mre": _mean(upper_rel),
        "bound_appendix": {
            "lower_mae": _mean(lower_abs),
            "lower_mre": _mean(lower_rel),
            "upper_mae": _mean(upper_abs),
            "upper_mre": _mean(upper_rel),
            "bound_fields_available_count": sum(1 for record in ok_records if record.get("bound_fields_available") is True),
        },
    }
    return row


def summarize_raw_dir(raw_dir: Path) -> dict[str, Any]:
    raw_files = sorted(path for path in raw_dir.glob("*.json") if path.is_file())
    records: list[dict[str, Any]] = []
    raw_file_summaries: list[dict[str, Any]] = []

    for path in raw_files:
        payload = load_json(path)
        file_records = payload.get("records", [])
        if not isinstance(file_records, list):
            raise ValueError(f"records must be a list in {path}")
        for record in file_records:
            if isinstance(record, dict):
                copied = dict(record)
                copied.setdefault("dataset", payload.get("config", {}).get("dataset"))
                copied.setdefault("model", payload.get("config", {}).get("model"))
                copied.setdefault("eps", payload.get("config", {}).get("eps"))
                copied["raw_path"] = str(path)
                records.append(copied)
        raw_file_summaries.append({"path": str(path), **_counts([r for r in file_records if isinstance(r, dict)])})

    grouped: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    reliability: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if _status(record) != MISCLASSIFIED_STATUS:
            grouped[_group_key(record)].append(record)
        reliability[_reliability_key(record)].append(record)

    groups = [_summarize_group(key, grouped[key]) for key in sorted(grouped)]
    reliability_rows = [
        {"dataset": key[0], "model": key[1], "eps": key[2], **_counts(reliability[key])}
        for key in sorted(reliability)
    ]
    totals = _counts(records)
    totals["small_regression_passed"] = totals["timeout_count"] == 0 and totals["secure_error_count"] == 0

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "relative_error_floor": REL_ERROR_FLOOR,
        "raw_dir": str(raw_dir),
        "totals": totals,
        "groups": groups,
        "reliability_rows": reliability_rows,
        "raw_files": raw_file_summaries,
        "records": records,
    }


def load_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return load_json(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_main_table(path: Path, groups: list[dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "model",
        "eps",
        "plaintext group",
        "ok",
        "gap MAE",
        "gap MRE",
        "consistency",
        "offline time s",
        "online time s",
        "total time s",
        "offline communication MB",
        "online communication MB",
        "total communication MB",
    ]
    rows = []
    for row in groups:
        rows.append(
            [
                str(row["dataset"]),
                str(row["model"]),
                _fmt(row["eps"]),
                str(row["plaintext_group"]),
                str(row["n_secure_ok"]),
                _fmt(row["gap_mae"]),
                _fmt(row["gap_mre"]),
                _fmt(row["consistency_rate"]),
                _fmt(row["offline_time_s_mean"]),
                _fmt(row["online_time_s_mean"]),
                _fmt(row["total_time_s_mean"]),
                _fmt(row["offline_comm_mb_mean"]),
                _fmt(row["online_comm_mb_mean"]),
                _fmt(row["total_comm_mb_mean"]),
            ]
        )
    path.write_text(_markdown_table(headers, rows), encoding="utf-8")


def write_reliability_table(path: Path, rows_in: list[dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "model",
        "eps",
        "attempted",
        "ok",
        "timeouts",
        "memory",
        "secure errors",
        "plain errors",
        "misclassified excluded",
        "blocked",
        "statuses",
    ]
    rows = []
    for row in rows_in:
        rows.append(
            [
                str(row["dataset"]),
                str(row["model"]),
                _fmt(row["eps"]),
                str(row["n_attempted"]),
                str(row["n_secure_ok"]),
                str(row["timeout_count"]),
                str(row["memory_limit_count"]),
                str(row["secure_error_count"]),
                str(row["plain_error_count"]),
                str(row["misclassified_excluded"]),
                str(row["blocked_count"]),
                ", ".join(f"{k}={v}" for k, v in row["statuses"].items()),
            ]
        )
    path.write_text(_markdown_table(headers, rows), encoding="utf-8")


def write_cifar_sanity_table(path: Path, cifar_sanity: dict[str, Any] | None) -> None:
    headers = [
        "model",
        "status",
        "shark robust",
        "shark not-certified",
        "shark correct",
        "project robust",
        "project not-certified",
        "project correct",
        "secure CIFAR allowed",
    ]
    rows = []
    for model, entry in sorted((cifar_sanity or {}).get("models", {}).items()):
        rows.append(
            [
                model,
                str(entry.get("sanity_status")),
                _fmt(entry.get("reference_robust_count")),
                _fmt(entry.get("reference_not_certified_count")),
                _fmt(entry.get("reference_correct_count")),
                _fmt(entry.get("plain_robust_count")),
                _fmt(entry.get("plain_not_certified_count")),
                _fmt(entry.get("correct_count")),
                str(entry.get("secure_cifar_allowed")),
            ]
        )
    path.write_text(_markdown_table(headers, rows), encoding="utf-8")


def write_tables(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_main_table(out_dir / "main_table.md", summary["groups"])
    write_reliability_table(out_dir / "reliability_table.md", summary["reliability_rows"])
    write_cifar_sanity_table(out_dir / "cifar_sanity_table.md", summary.get("cifar_sanity"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--cifar-sanity", required=True, type=Path)
    parser.add_argument("--nested-state", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_raw_dir(args.raw_dir)
    summary["cifar_sanity"] = load_optional(args.cifar_sanity)
    summary["nested_cryptflow_state"] = load_optional(args.nested_state)
    summary["accounting_boundary"] = {
        "model": "implementation-level per-sample accounting",
        "true_reusable_session_amortization": False,
        "offline_note": "Raw offline and amortized offline may be equal because each secure sample launches fresh two-party SCI state.",
    }
    write_json(args.out_json, summary)
    write_tables(args.out_dir, summary)
    print(f"[summary] records={summary['totals']['n_attempted']} groups={len(summary['groups'])} out={args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
