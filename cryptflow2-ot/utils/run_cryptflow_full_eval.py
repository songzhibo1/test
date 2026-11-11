#!/usr/bin/env python3
"""Phase 07 CryptFlow2-CROWN full-evaluation helpers."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cifar10_data import (  # noqa: E402
    REQUIRED_CIFAR_MODELS,
    CifarAssetStatus,
    discover_shark_cifar_assets,
)
from crown_runner import (  # noqa: E402
    CrownSampleResult,
    prepare_crown_mlp,
    run_cryptflow_crown_sample,
    run_plain_crown_sample,
)
from mnist_data import load_mnist_test_set  # noqa: E402
from model_eval import ModelEvaluator, min_logit_targets_from_logits  # noqa: E402


CIFAR_SANITY_REFERENCES: dict[str, dict[str, int]] = {
    "5x100": {"robust": 24, "not_certified": 21, "correct": 45},
    "7x100": {"robust": 32, "not_certified": 14, "correct": 46},
    "10x200": {"robust": 28, "not_certified": 24, "correct": 52},
}

CIFAR_EPSILONS: dict[str, list[float]] = {
    "5x100": [0.002, 0.0078],
    "7x100": [0.001],
    "10x200": [0.001],
}

MNIST_EPSILONS: dict[str, list[float]] = {
    "20x1": [0.015, 0.045, 0.1],
    "20x2": [0.015, 0.03, 0.1],
    "256x2": [0.015, 0.1],
    "256x4": [0.015, 0.1],
    "256x6": [0.015, 0.1],
}

RESOURCE_LIMITS: dict[tuple[str, str], tuple[int, int]] = {
    ("mnist", "20x1"): (300, 16),
    ("mnist", "20x2"): (300, 16),
    ("mnist", "256x2"): (1200, 32),
    ("mnist", "256x4"): (2400, 48),
    ("mnist", "256x6"): (3600, 64),
    ("cifar10", "5x100"): (2400, 48),
    ("cifar10", "7x100"): (3600, 64),
    ("cifar10", "10x200"): (5400, 96),
}

CORRECT_COUNT_CLOSE_TOLERANCE = 2
SPLIT_TOLERANCE = 2
RELATIVE_ERROR_FLOOR = 1e-6
DEFAULT_RAW_DIR = Path("results/cryptflow2_crown_full_eval/raw")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_ready(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def write_json(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resource_limits_for(dataset: str, model: str) -> tuple[int, int]:
    try:
        return RESOURCE_LIMITS[(dataset, model)]
    except KeyError as exc:
        raise ValueError(f"Unknown resource-limit tier: {dataset}/{model}") from exc


def build_config_matrix(generated_at: str | None = None) -> dict[str, Any]:
    combinations: list[dict[str, Any]] = []
    for model, epsilons in MNIST_EPSILONS.items():
        timeout_sec, memory_limit_gb = resource_limits_for("mnist", model)
        combinations.append(
            {
                "dataset": "mnist",
                "model": model,
                "epsilons": epsilons,
                "secure_execution": "allowed",
                "onnx": f"models/mnistfc/mnist-net_{model}.onnx",
                "resource_limits": {
                    "timeout_sec": timeout_sec,
                    "memory_limit_gb": memory_limit_gb,
                },
            }
        )
    for model, epsilons in CIFAR_EPSILONS.items():
        timeout_sec, memory_limit_gb = resource_limits_for("cifar10", model)
        combinations.append(
            {
                "dataset": "cifar10",
                "model": model,
                "epsilons": epsilons,
                "secure_execution": "requires_cifar_sanity",
                "onnx": None,
                "resource_limits": {
                    "timeout_sec": timeout_sec,
                    "memory_limit_gb": memory_limit_gb,
                },
            }
        )
    return {
        "schema_version": 1,
        "generated_at": generated_at or _now_iso(),
        "sample_policy": {
            "secure_n": 20,
            "scan_prefix": 20,
            "rebalance_plaintext_groups": False,
            "misclassified_status": "plain_misclassified",
            "relative_error_floor": RELATIVE_ERROR_FLOOR,
        },
        "status_contract": [
            "ok",
            "timeout",
            "memory_limit",
            "secure_error",
            "plain_error",
            "plain_misclassified",
            "blocked_cifar_sanity",
        "blocked_missing_assets",
        "blocked_small_regression",
        ],
        "combinations": combinations,
    }


def cifar_secure_allowed(sanity: dict[str, Any], model: str, eps: float) -> bool:
    entry = sanity.get("models", {}).get(model)
    if not isinstance(entry, dict):
        return False
    if entry.get("secure_cifar_allowed") is not True:
        return False
    return float(eps) in {float(value) for value in entry.get("epsilons", [])}


def plaintext_group_from_gap(gap: float | None) -> str | None:
    if gap is None:
        return None
    return "robust" if gap > 0.0 else "not-certified"


def _gap_errors(plain_gap: float | None, secure_gap: float | None) -> tuple[float | None, float | None]:
    if plain_gap is None or secure_gap is None:
        return None, None
    abs_err = abs(float(secure_gap) - float(plain_gap))
    return abs_err, abs_err / max(abs(float(plain_gap)), RELATIVE_ERROR_FLOOR)


def build_plaintext_prefilter_records(
    *,
    dataset: str,
    model: str,
    eps: float,
    images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    plain_results: dict[int, CrownSampleResult],
    targets: np.ndarray | None = None,
    secure_n: int = 20,
) -> tuple[list[CrownSampleResult], list[dict[str, Any]]]:
    records: list[CrownSampleResult] = []
    eligible: list[dict[str, Any]] = []
    upper = min(secure_n, int(labels.shape[0]))
    for index in range(upper):
        label = int(labels[index])
        pred = int(preds[index])
        if pred != label:
            records.append(
                CrownSampleResult(
                    index=index,
                    label=label,
                    pred=pred,
                    eps=float(eps),
                    method="cryptflow-crown",
                    status="plain_misclassified",
                    dataset=dataset,
                    model=model,
                )
            )
            continue
        plain = plain_results.get(index)
        if plain is None:
            records.append(
                CrownSampleResult(
                    index=index,
                    label=label,
                    pred=pred,
                    eps=float(eps),
                    method="plain-crown",
                    status="plain_error",
                    dataset=dataset,
                    model=model,
                    resource_status="plain_error",
                    error="plaintext CROWN result missing for correctly classified sample",
                )
            )
            continue
        group = plaintext_group_from_gap(plain.plain_gap)
        status = plain.status if plain.status == "ok" else "plain_error"
        records.append(
            CrownSampleResult(
                index=index,
                label=label,
                pred=pred,
                eps=float(eps),
                method="plain-crown",
                status=status,
                certified=plain.certified,
                plain_gap=plain.plain_gap,
                dataset=dataset,
                model=model,
                plaintext_group=group,
                resource_status=status,
                error=plain.error,
            )
        )
        if status == "ok":
            eligible.append(
                {
                    "index": index,
                    "x": np.asarray(images[index], dtype=np.float64).reshape(-1),
                    "label": label,
                    "pred": pred,
                    "target": int(targets[index]) if targets is not None else None,
                    "plaintext_group": group,
                    "plain_gap": plain.plain_gap,
                    "plain_certified": plain.certified,
                }
            )
    return records, eligible


def summarize_full_eval_records(records: list[CrownSampleResult]) -> dict[str, Any]:
    reliability = {
        "n_attempted": len(records),
        "n_secure_ok": sum(1 for r in records if r.method == "cryptflow-crown" and r.status == "ok"),
        "timeout_count": sum(1 for r in records if r.status == "timeout"),
        "memory_limit_count": sum(1 for r in records if r.status == "memory_limit"),
        "secure_error_count": sum(1 for r in records if r.status == "secure_error"),
        "plain_error_count": sum(1 for r in records if r.status == "plain_error"),
        "misclassified_excluded": sum(1 for r in records if r.status == "plain_misclassified"),
    }
    grouped: dict[tuple[str, str, float, str], list[CrownSampleResult]] = {}
    for record in records:
        if record.plaintext_group is None:
            continue
        key = (
            record.dataset or "",
            record.model or "",
            float(record.eps),
            record.plaintext_group,
        )
        grouped.setdefault(key, []).append(record)

    groups = []
    for (dataset, model, eps, plaintext_group), rows in sorted(grouped.items()):
        ok = [r for r in rows if r.status == "ok"]
        consistent = sum(
            1
            for r in ok
            if r.certified is not None
            and (
                (plaintext_group == "robust" and r.certified is True)
                or (plaintext_group == "not-certified" and r.certified is False)
            )
        )
        gap_abs_values = [r.gap_abs_err for r in ok if r.gap_abs_err is not None]
        gap_rel_values = [r.gap_rel_err for r in ok if r.gap_rel_err is not None]
        groups.append(
            {
                "dataset": dataset,
                "model": model,
                "eps": eps,
                "plaintext_group": plaintext_group,
                "n_records": len(rows),
                "n_secure_ok": len(ok),
                "consistency_rate": (consistent / len(ok)) if ok else None,
                "gap_mae": (sum(gap_abs_values) / len(gap_abs_values)) if gap_abs_values else None,
                "gap_mre": (sum(gap_rel_values) / len(gap_rel_values)) if gap_rel_values else None,
                "timeout_count": sum(1 for r in rows if r.status == "timeout"),
                "memory_limit_count": sum(1 for r in rows if r.status == "memory_limit"),
                "secure_error_count": sum(1 for r in rows if r.status == "secure_error"),
                "plain_error_count": sum(1 for r in rows if r.status == "plain_error"),
                "misclassified_excluded": sum(1 for r in rows if r.status == "plain_misclassified"),
            }
        )
    return {"reliability": reliability, "groups": groups}


def flush_combination_json(
    out_path: Path,
    *,
    config: dict[str, Any],
    records: list[CrownSampleResult],
) -> None:
    payload = {
        "config": config,
        "summary": summarize_full_eval_records(records),
        "records": [record.to_dict() for record in records],
    }
    write_json(out_path, payload)


def _copy_log_into_raw(
    log_path: str | None, *, raw_dir: Path, stem: str, role: str
) -> str | None:
    if not log_path:
        return None
    src = Path(log_path)
    if not src.is_file():
        return log_path
    try:
        src.resolve().relative_to(raw_dir.resolve())
        return str(src)
    except ValueError:
        pass
    logs_dir = raw_dir / "logs" / stem
    logs_dir.mkdir(parents=True, exist_ok=True)
    dst = logs_dir / f"{role}_{src.name}"
    shutil.copy2(src, dst)
    return str(dst)


def localize_record_logs(
    record: CrownSampleResult, *, raw_dir: Path, combination_stem: str
) -> CrownSampleResult:
    data = record.to_dict()
    data["stdout_path"] = _copy_log_into_raw(
        record.stdout_path,
        raw_dir=raw_dir,
        stem=combination_stem,
        role=f"sample{record.index}_stdout",
    )
    data["stderr_path"] = _copy_log_into_raw(
        record.stderr_path,
        raw_dir=raw_dir,
        stem=combination_stem,
        role=f"sample{record.index}_stderr",
    )
    return CrownSampleResult(**data)


def _safe_name(dataset: str, model: str, eps: float) -> str:
    eps_s = f"{eps:.17g}".replace(".", "p").replace("-", "m")
    return f"{dataset}_{model}_eps{eps_s}.json"


def classify_cifar_sanity_counts(
    *,
    model: str,
    correct_count: int,
    plain_robust_count: int,
    plain_not_certified_count: int,
    correct_close_tolerance: int = CORRECT_COUNT_CLOSE_TOLERANCE,
    split_tolerance: int = SPLIT_TOLERANCE,
) -> dict[str, Any]:
    if model not in CIFAR_SANITY_REFERENCES:
        raise ValueError(f"Unknown CIFAR sanity model: {model}")
    ref = CIFAR_SANITY_REFERENCES[model]
    ref_correct = ref["correct"]
    correct_delta = int(correct_count) - ref_correct
    if correct_delta == 0:
        correct_status = "exact"
    elif abs(correct_delta) <= correct_close_tolerance:
        correct_status = "close"
    else:
        correct_status = "mismatch"

    robust_delta = int(plain_robust_count) - ref["robust"]
    not_certified_delta = int(plain_not_certified_count) - ref["not_certified"]
    if robust_delta == 0 and not_certified_delta == 0:
        split_status = "exact"
    elif (
        abs(robust_delta) <= split_tolerance
        and abs(not_certified_delta) <= split_tolerance
    ):
        split_status = "tolerant"
    else:
        split_status = "mismatch_reported"

    sanity_status = (
        "failed_preprocessing_mismatch"
        if correct_status == "mismatch"
        else "passed"
    )
    return {
        "model": model,
        "sanity_status": sanity_status,
        "secure_cifar_allowed": sanity_status == "passed",
        "correct_count": int(correct_count),
        "plain_robust_count": int(plain_robust_count),
        "plain_not_certified_count": int(plain_not_certified_count),
        "reference_correct_count": ref_correct,
        "reference_robust_count": ref["robust"],
        "reference_not_certified_count": ref["not_certified"],
        "correct_count_status": correct_status,
        "correct_count_delta": correct_delta,
        "split_status": split_status,
        "robust_count_delta": robust_delta,
        "not_certified_count_delta": not_certified_delta,
    }


def _blocked_missing_asset_entry(
    model: str, asset_status: CifarAssetStatus
) -> dict[str, Any]:
    ref = CIFAR_SANITY_REFERENCES[model]
    return {
        "model": model,
        "epsilons": CIFAR_EPSILONS[model],
        "sanity_status": "blocked_missing_assets",
        "secure_cifar_allowed": False,
        "asset_path": asset_status.found_models.get(model),
        "missing_exact_asset": model in asset_status.missing_models,
        "correct_count": None,
        "plain_robust_count": None,
        "plain_not_certified_count": None,
        "reference_correct_count": ref["correct"],
        "reference_robust_count": ref["robust"],
        "reference_not_certified_count": ref["not_certified"],
        "correct_count_status": "not_run_missing_assets",
        "split_status": "not_run_missing_assets",
        "notes": (
            "Exact CIFAR asset was not found under source_root; no CIFAR secure "
            "CryptFlow2 job is allowed for this model."
        ),
    }


def _existing_generated_at(out_path: Path) -> str | None:
    if not out_path.is_file():
        return None
    try:
        loaded = json.loads(out_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    value = loaded.get("generated_at")
    return value if isinstance(value, str) else None


def build_blocked_missing_assets_payload(
    *, source_root: Path, asset_status: CifarAssetStatus, generated_at: str | None = None
) -> dict[str, Any]:
    models = {
        model: _blocked_missing_asset_entry(model, asset_status)
        for model in REQUIRED_CIFAR_MODELS
    }
    return {
        "schema_version": 1,
        "generated_at": generated_at or _now_iso(),
        "sanity_status": "blocked_missing_assets",
        "secure_cifar_allowed": False,
        "source_root": str(Path(source_root).expanduser()),
        "asset_status": asset_status.to_dict(),
        "preprocessing_candidates": [
            spec.to_dict() for spec in asset_status.candidate_preprocessing
        ],
        "models": models,
        "secure_results": [],
        "blocked_reason": (
            "missing_exact_assets: required CIFAR models are exactly 5x100, 7x100, "
            "and 10x200; substitutes are not permitted."
        ),
    }


def run_cifar_sanity(source_root: Path, out_path: Path) -> dict[str, Any]:
    asset_status = discover_shark_cifar_assets(source_root)
    generated_at = _existing_generated_at(out_path)
    if asset_status.status != "found":
        payload = build_blocked_missing_assets_payload(
            source_root=source_root,
            asset_status=asset_status,
            generated_at=generated_at,
        )
        write_json(out_path, payload)
        return payload

    models: dict[str, Any] = {}
    for model in REQUIRED_CIFAR_MODELS:
        ref = CIFAR_SANITY_REFERENCES[model]
        models[model] = {
            "model": model,
            "epsilons": CIFAR_EPSILONS[model],
            "sanity_status": "blocked_missing",
            "secure_cifar_allowed": False,
            "asset_path": asset_status.found_models.get(model),
            "correct_count": None,
            "plain_robust_count": None,
            "plain_not_certified_count": None,
            "reference_correct_count": ref["correct"],
            "reference_robust_count": ref["robust"],
            "reference_not_certified_count": ref["not_certified"],
            "correct_count_status": "not_run_plaintext_oracle_unavailable",
            "split_status": "not_run_plaintext_oracle_unavailable",
            "notes": (
                "Exact assets were detected, but automated plaintext CROWN migration "
                "for those external assets is not available in this plan run."
            ),
        }

    payload = {
        "schema_version": 1,
        "generated_at": generated_at or _now_iso(),
        "sanity_status": "blocked_missing",
        "secure_cifar_allowed": False,
        "source_root": str(Path(source_root).expanduser()),
        "asset_status": asset_status.to_dict(),
        "preprocessing_candidates": [
            spec.to_dict() for spec in asset_status.candidate_preprocessing
        ],
        "models": models,
        "secure_results": [],
        "blocked_reason": "plaintext_sanity_not_executed",
    }
    write_json(out_path, payload)
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _existing_records(path: Path) -> list[CrownSampleResult]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    records = payload.get("records", [])
    if not isinstance(records, list):
        return []
    return [CrownSampleResult(**record) for record in records]


def _blocked_cifar_records(
    *,
    model: str,
    eps: float,
    secure_n: int,
    status: str,
) -> list[CrownSampleResult]:
    return [
        CrownSampleResult(
            index=index,
            label=-1,
            pred=-1,
            eps=float(eps),
            method="cryptflow-crown",
            status=status,
            dataset="cifar10",
            model=model,
            resource_status=status,
        )
        for index in range(secure_n)
    ]


def _blocked_records(
    *,
    dataset: str,
    model: str,
    eps: float,
    secure_n: int,
    status: str,
) -> list[CrownSampleResult]:
    return [
        CrownSampleResult(
            index=index,
            label=-1,
            pred=-1,
            eps=float(eps),
            method="cryptflow-crown",
            status=status,
            dataset=dataset,
            model=model,
            resource_status=status,
        )
        for index in range(secure_n)
    ]


def _cifar_block_status(sanity: dict[str, Any], model: str, eps: float) -> str | None:
    if cifar_secure_allowed(sanity, model, eps):
        return None
    entry = sanity.get("models", {}).get(model)
    if isinstance(entry, dict) and (
        entry.get("missing_exact_asset") is True
        or entry.get("sanity_status") == "blocked_missing_assets"
    ):
        return "blocked_missing_assets"
    if sanity.get("sanity_status") == "blocked_missing_assets":
        return "blocked_missing_assets"
    return "blocked_cifar_sanity"


def _selected_combinations(
    matrix: dict[str, Any], only_models: set[str]
) -> list[dict[str, Any]]:
    combos = list(matrix.get("combinations", []))
    if not only_models:
        return combos
    return [combo for combo in combos if str(combo.get("model")) in only_models]


def _raw_path_for(raw_dir: Path, dataset: str, model: str, eps: float) -> Path:
    return raw_dir / _safe_name(dataset, model, eps)


def _combination_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "raw_path": str(path),
            "exists": False,
            "n_attempted": 0,
            "n_secure_ok": 0,
            "timeout_count": 0,
            "memory_limit_count": 0,
            "secure_error_count": 0,
            "plain_error_count": 0,
            "misclassified_excluded": 0,
            "statuses": {},
            "groups": [],
        }
    payload = json.loads(path.read_text())
    records = payload.get("records", [])
    statuses: dict[str, int] = {}
    for record in records:
        if isinstance(record, dict):
            status = str(record.get("status"))
            statuses[status] = statuses.get(status, 0) + 1
    reliability = payload.get("summary", {}).get("reliability", {})
    return {
        "raw_path": str(path),
        "exists": True,
        "n_attempted": int(reliability.get("n_attempted", len(records))),
        "n_secure_ok": int(reliability.get("n_secure_ok", 0)),
        "timeout_count": int(reliability.get("timeout_count", 0)),
        "memory_limit_count": int(reliability.get("memory_limit_count", 0)),
        "secure_error_count": int(reliability.get("secure_error_count", 0)),
        "plain_error_count": int(reliability.get("plain_error_count", 0)),
        "misclassified_excluded": int(reliability.get("misclassified_excluded", 0)),
        "statuses": statuses,
        "groups": payload.get("summary", {}).get("groups", []),
    }


def _small_regression_gate(
    *, raw_dir: Path, matrix: dict[str, Any]
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    required = {"20x1", "20x2"}
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    gate_path = Path("results/cryptflow2_crown_full_eval/07-01-small-regression-gate.json")
    prior_gate = _load_json(gate_path)
    if prior_gate.get("small_regression_passed") is not True:
        reasons.append(f"prior 07-01 gate is not true: {gate_path}")

    for combo in matrix.get("combinations", []):
        if combo.get("dataset") != "mnist" or str(combo.get("model")) not in required:
            continue
        for eps in combo.get("epsilons", []):
            summary = _combination_summary(
                _raw_path_for(raw_dir, "mnist", str(combo["model"]), float(eps))
            )
            row = {
                "dataset": "mnist",
                "model": str(combo["model"]),
                "eps": float(eps),
                **summary,
            }
            rows.append(row)
            if not summary["exists"]:
                reasons.append(f"missing raw JSON for {combo['model']} eps={eps}")
                continue
            if summary["n_secure_ok"] <= 0:
                reasons.append(f"no successful secure rows for {combo['model']} eps={eps}")
            for key in ("timeout_count", "memory_limit_count", "secure_error_count", "plain_error_count"):
                if summary[key] != 0:
                    reasons.append(f"{key}={summary[key]} for {combo['model']} eps={eps}")
            for group in summary["groups"]:
                if group.get("n_secure_ok", 0) > 0 and group.get("consistency_rate") != 1.0:
                    reasons.append(
                        f"consistency_rate={group.get('consistency_rate')} for "
                        f"{combo['model']} eps={eps} group={group.get('plaintext_group')}"
                    )
    seen = {row["model"] for row in rows}
    for model in sorted(required - seen):
        reasons.append(f"missing small model in matrix/raw output: {model}")
    return not reasons, rows, reasons


def _git_text(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except OSError as exc:
        return f"git unavailable: {exc}"


def write_nested_cryptflow_state(
    out_path: Path,
    *,
    repo: Path,
    matrix: dict[str, Any],
    command_lines: list[str],
    accounting_models: list[str],
) -> None:
    repo = repo.resolve()
    payload = {
        "schema_version": 1,
        "created_at": _now_iso(),
        "cryptflow2_crown": {
            "path": str(repo),
            "branch": _git_text(["rev-parse", "--abbrev-ref", "HEAD"], repo),
            "head": _git_text(["rev-parse", "--short", "HEAD"], repo),
            "status_short": _git_text(["status", "--short"], repo).splitlines(),
            "diff_stat": _git_text(["diff", "--stat"], repo).splitlines(),
        },
        "command_lines": command_lines,
        "resource_limits": {
            f"{dataset}/{model}": {
                "timeout_sec": timeout,
                "memory_limit_gb": memory,
            }
            for (dataset, model), (timeout, memory) in RESOURCE_LIMITS.items()
        },
        "accounting_models_seen": sorted(set(accounting_models)),
        "accounting_model_expected": "per_sample_setup",
        "sample_policy": matrix.get("sample_policy", {}),
    }
    write_json(out_path, payload)


def write_run_manifest(
    out_path: Path,
    *,
    matrix: dict[str, Any],
    raw_dir: Path,
    selected: list[dict[str, Any]],
    small_regression_passed: bool,
    small_gate_rows: list[dict[str, Any]],
    small_gate_reasons: list[str],
    command_lines: list[str],
) -> dict[str, Any]:
    combinations: list[dict[str, Any]] = []
    totals = {
        "n_attempted": 0,
        "n_secure_ok": 0,
        "timeout_count": 0,
        "memory_limit_count": 0,
        "secure_error_count": 0,
        "plain_error_count": 0,
        "misclassified_excluded": 0,
    }
    selected_keys = {
        (str(combo.get("dataset")), str(combo.get("model")), float(eps))
        for combo in selected
        for eps in combo.get("epsilons", [])
    }
    accounting_models: list[str] = []
    cifar_blocked = False
    for combo in matrix.get("combinations", []):
        dataset = str(combo["dataset"])
        model = str(combo["model"])
        for eps in combo["epsilons"]:
            eps_f = float(eps)
            summary = _combination_summary(_raw_path_for(raw_dir, dataset, model, eps_f))
            included_this_run = (dataset, model, eps_f) in selected_keys
            statuses = summary["statuses"]
            if any(status.startswith("blocked") for status in statuses):
                cifar_blocked = cifar_blocked or dataset == "cifar10"
            if summary["exists"]:
                try:
                    payload = json.loads(Path(summary["raw_path"]).read_text())
                except (OSError, json.JSONDecodeError):
                    payload = {}
                for record in payload.get("records", []):
                    if isinstance(record, dict) and record.get("accounting_model"):
                        accounting_models.append(str(record["accounting_model"]))
            row = {
                "dataset": dataset,
                "model": model,
                "eps": eps_f,
                "included_this_run": included_this_run,
                **summary,
            }
            combinations.append(row)
            for key in totals:
                totals[key] += int(summary.get(key, 0))
    manifest = {
        "schema_version": 1,
        "updated_at": _now_iso(),
        "status": "ok" if small_regression_passed else "blocked_small_regression",
        "small_regression_passed": small_regression_passed,
        "small_regression_gate": {
            "rows": small_gate_rows,
            "reasons": small_gate_reasons,
        },
        "sample_policy": matrix.get("sample_policy", {}),
        "command_lines": command_lines,
        "n_combinations": len(combinations),
        "n_attempted": totals["n_attempted"],
        "totals": totals,
        "cifar_blocked": cifar_blocked,
        "accounting_models_seen": sorted(set(accounting_models)),
        "combinations": combinations,
    }
    write_json(out_path, manifest)
    return manifest


def _run_mnist_combination(
    *,
    combo: dict[str, Any],
    eps: float,
    out_path: Path,
    args: argparse.Namespace,
    secure_n: int,
) -> list[CrownSampleResult]:
    if args.resume and out_path.is_file():
        return _existing_records(out_path)

    onnx_path = Path(combo["onnx"])
    images, labels = load_mnist_test_set(args.data_root)
    evaluator = ModelEvaluator(onnx_path)
    if images.shape[1] != evaluator.input_dim:
        raise ValueError(
            f"MNIST input dimension {images.shape[1]} != model input dimension {evaluator.input_dim}"
        )
    prefix_images = images[:secure_n]
    logits = evaluator.logits_batch(prefix_images)
    preds = np.argmax(logits, axis=1).astype(np.int64)
    targets = min_logit_targets_from_logits(logits, preds)
    full_preds = np.full(labels.shape[0], -1, dtype=np.int64)
    full_targets = np.full(labels.shape[0], -1, dtype=np.int64)
    full_preds[:secure_n] = preds
    full_targets[:secure_n] = targets

    plain_results: dict[int, CrownSampleResult] = {}
    for index in range(min(secure_n, labels.shape[0])):
        if int(full_preds[index]) != int(labels[index]):
            continue
        target = int(full_targets[index])
        plain_results[index] = run_plain_crown_sample(
            layers=evaluator.layers,
            x=images[index],
            eps=float(eps),
            index=index,
            label=int(labels[index]),
            pred=int(full_preds[index]),
            target=target,
        )

    records, eligible = build_plaintext_prefilter_records(
        dataset="mnist",
        model=str(combo["model"]),
        eps=float(eps),
        images=images,
        labels=labels,
        preds=full_preds,
        targets=full_targets,
        plain_results=plain_results,
        secure_n=secure_n,
    )
    flush_combination_json(
        out_path,
        config={
            **combo,
            "eps": float(eps),
            "secure_n": secure_n,
            "decision_mode": "targeted",
            "target_policy": "min_logits",
        },
        records=records,
    )

    mlp_path = args.out_dir / "prepared_models" / f"{onnx_path.stem}.crownmlp"
    mlp_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_crown_mlp(onnx_path, mlp_path)
    timeout_sec, memory_limit_gb = resource_limits_for("mnist", str(combo["model"]))
    secure_records_by_index: dict[int, CrownSampleResult] = {}
    combination_stem = out_path.stem
    for offset, item in enumerate(eligible):
        secure = run_cryptflow_crown_sample(
            onnx_path=onnx_path,
            x=item["x"],
            eps=float(eps),
            index=int(item["index"]),
            label=int(item["label"]),
            pred=int(item["pred"]),
            target=int(item["target"]),
            backend=args.backend,
            repo=args.repo,
            bitlength=args.bitlength,
            scale_bits=args.scale_bits,
            threads=args.threads,
            ip=args.ip,
            port=args.port + offset,
            timeout=timeout_sec,
            memory_limit_gb=memory_limit_gb,
            skip_build=args.skip_build,
            keep_artifacts=args.keep_artifacts,
            mlp_path=mlp_path,
        )
        secure = localize_record_logs(
            secure,
            raw_dir=args.out_dir / "raw",
            combination_stem=combination_stem,
        )
        group = str(item["plaintext_group"])
        gap_abs_err, gap_rel_err = _gap_errors(item["plain_gap"], secure.sci_gap)
        secure_records_by_index[int(item["index"])] = CrownSampleResult(
            **{
                **secure.to_dict(),
                "dataset": "mnist",
                "model": str(combo["model"]),
                "plaintext_group": group,
                "plain_gap": item["plain_gap"],
                "gap_abs_err": gap_abs_err,
                "gap_rel_err": gap_rel_err,
            }
        )
        merged = [
            secure_records_by_index.get(record.index, record)
            for record in records
            if record.method != "plain-crown" or record.index not in secure_records_by_index
        ]
        for index, secure_record in secure_records_by_index.items():
            if not any(record.index == index for record in merged):
                merged.append(secure_record)
        records = sorted(merged, key=lambda r: r.index)
        flush_combination_json(
            out_path,
            config={
                **combo,
                "eps": float(eps),
                "secure_n": secure_n,
                "decision_mode": "targeted",
                "target_policy": "min_logits",
            },
            records=records,
        )
    return records


def run_matrix(args: argparse.Namespace) -> int:
    matrix = _load_json(args.matrix) if args.matrix.is_file() else build_config_matrix()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sanity = _load_json(args.cifar_sanity)
    secure_n = int(matrix.get("sample_policy", {}).get("secure_n", 20))
    only_models = set(args.only_model or [])
    selected = _selected_combinations(matrix, only_models)
    existing_manifest = _load_json(args.out_dir / "run_manifest.json")
    existing_commands = [
        str(value) for value in existing_manifest.get("command_lines", [])
    ]
    current_command = " ".join([sys.executable, *sys.argv])
    command_lines = [*existing_commands]
    if current_command not in command_lines:
        command_lines.append(current_command)
    small_only = only_models and only_models <= {"20x1", "20x2"}
    gate_already_passed = existing_manifest.get("small_regression_passed") is True

    for combo in selected:
        dataset = str(combo["dataset"])
        model = str(combo["model"])
        for eps in combo["epsilons"]:
            eps_f = float(eps)
            out_path = raw_dir / _safe_name(dataset, model, eps_f)
            if dataset == "mnist" and model not in {"20x1", "20x2"} and not gate_already_passed:
                if args.resume and out_path.is_file():
                    continue
                records = _blocked_records(
                    dataset=dataset,
                    model=model,
                    eps=eps_f,
                    secure_n=secure_n,
                    status="blocked_small_regression",
                )
                flush_combination_json(
                    out_path,
                    config={**combo, "eps": eps_f, "secure_n": secure_n},
                    records=records,
                )
                continue
            if dataset == "cifar10":
                status = _cifar_block_status(sanity, model, eps_f)
                if status is None and not gate_already_passed:
                    status = "blocked_small_regression"
                if status is None:
                    status = "blocked_cifar_sanity"
                if args.resume and out_path.is_file():
                    continue
                records = _blocked_records(
                    dataset=dataset,
                    model=model,
                    eps=eps_f,
                    secure_n=secure_n,
                    status=status,
                )
                flush_combination_json(
                    out_path,
                    config={**combo, "eps": eps_f, "secure_n": secure_n},
                    records=records,
                )
                continue
            if dataset == "mnist":
                _run_mnist_combination(
                    combo=combo,
                    eps=eps_f,
                    out_path=out_path,
                    args=args,
                    secure_n=secure_n,
                )
                continue
            raise ValueError(f"Unknown dataset in matrix: {dataset}")
    small_passed, small_rows, small_reasons = _small_regression_gate(
        raw_dir=raw_dir, matrix=matrix
    )
    if small_only:
        gate_value = small_passed
    else:
        gate_value = gate_already_passed and small_passed
    manifest = write_run_manifest(
        args.out_dir / "run_manifest.json",
        matrix=matrix,
        raw_dir=raw_dir,
        selected=selected,
        small_regression_passed=gate_value,
        small_gate_rows=small_rows,
        small_gate_reasons=small_reasons,
        command_lines=command_lines,
    )
    write_nested_cryptflow_state(
        args.out_dir / "nested_cryptflow_state.json",
        repo=args.repo,
        matrix=matrix,
        command_lines=command_lines,
        accounting_models=manifest.get("accounting_models_seen", []),
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sanity = sub.add_parser("cifar-sanity", help="write CIFAR first-100 sanity gate")
    sanity.add_argument(
        "--source-root",
        type=Path,
        default=Path("external/cifar_reference"),
    )
    sanity.add_argument(
        "--out",
        type=Path,
        default=Path("results/cryptflow2_crown_full_eval/cifar_sanity.json"),
    )
    write_matrix = sub.add_parser("write-matrix", help="write frozen Phase 7 matrix")
    write_matrix.add_argument(
        "--out",
        type=Path,
        default=Path("results/cryptflow2_crown_full_eval/config_matrix.json"),
    )
    run = sub.add_parser("run-matrix", help="run the fixed full evaluation matrix")
    run.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/cryptflow2_crown_full_eval"),
    )
    run.add_argument(
        "--matrix",
        type=Path,
        default=Path("results/cryptflow2_crown_full_eval/config_matrix.json"),
    )
    run.add_argument(
        "--cifar-sanity",
        type=Path,
        default=Path("results/cryptflow2_crown_full_eval/cifar_sanity.json"),
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument("--only-model", action="append", default=[])
    run.add_argument("--data-root", type=Path, default=Path("data"))
    run.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    run.add_argument("--backend", choices=["ot", "he"], default="ot")
    run.add_argument("--bitlength", type=int, default=41)
    run.add_argument("--scale-bits", type=int, default=18)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--ip", default="127.0.0.1")
    run.add_argument("--port", type=int, default=45000)
    run.add_argument("--skip-build", action="store_true")
    run.add_argument("--keep-artifacts", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "cifar-sanity":
        payload = run_cifar_sanity(source_root=args.source_root, out_path=args.out)
        print(
            "[cifar-sanity] "
            f"status={payload['sanity_status']} "
            f"secure_cifar_allowed={str(payload['secure_cifar_allowed']).lower()} "
            f"out={args.out}"
        )
        return 0
    if args.command == "write-matrix":
        matrix = build_config_matrix(generated_at=_existing_generated_at(args.out))
        write_json(args.out, matrix)
        print(f"[matrix] out={args.out} combinations={len(matrix['combinations'])}")
        return 0
    if args.command == "run-matrix":
        return run_matrix(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
