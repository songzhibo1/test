#!/usr/bin/env python3
"""Run batch CROWN experiments on a slice of the MNIST test set."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from crown_runner import (  # noqa: E402
    CrownSampleResult,
    prepare_crown_mlp,
    run_ckks_crown_sample,
    run_cryptflow_crown_sample,
    run_nexus_crown_sample,
    run_plain_crown_sample,
)
from mnist_data import load_mnist_test_set  # noqa: E402
from model_eval import (  # noqa: E402
    ModelEvaluator,
    min_logit_targets_from_logits,
    select_candidate_samples,
)


def ensure_json_output_path(out_path: Path) -> None:
    if out_path.suffix.lower() != ".json":
        raise ValueError("Batch experiment output is JSON-only; use an .json path")


def _avg(values: Iterable[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return (sum(present) / len(present)) if present else None


def _total(values: Iterable[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) if present else None


def _max(values: Iterable[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return max(present) if present else None


def summarize_records(
    records: list[CrownSampleResult],
    *,
    candidate_count: int,
    selected_count: int,
) -> dict[str, Any]:
    ok_records = [r for r in records if r.status == "ok"]
    return {
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "evaluated_count": len(records),
        "ok_count": len(ok_records),
        "error_count": len(records) - len(ok_records),
        "certified_count": sum(1 for r in ok_records if r.certified is True),
        "avg_time_ms": _avg(r.time_ms for r in ok_records),
        "total_time_ms": _total(r.time_ms for r in ok_records),
        "avg_comm_bytes": _avg(r.comm_bytes for r in ok_records),
        "total_comm_bytes": _total(r.comm_bytes for r in ok_records),
        "avg_artifact_bytes": _avg(r.nexus_artifact_bytes for r in ok_records),
        "total_artifact_bytes": _total(r.nexus_artifact_bytes for r in ok_records),
        "avg_abs_err_gap": _avg(
            r.nexus_abs_err_gap if r.method == "nexus-crown" else r.abs_err_gap
            for r in ok_records
        ),
        "max_abs_err_gap": _max(
            r.nexus_abs_err_gap if r.method == "nexus-crown" else r.abs_err_gap
            for r in ok_records
        ),
        "avg_rel_err_gap": _avg(r.nexus_rel_err_gap for r in ok_records),
        "max_rel_err_gap": _max(r.nexus_rel_err_gap for r in ok_records),
    }


def write_results(
    out_path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    records: list[CrownSampleResult],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_json_output_path(out_path)
    payload = {
        "config": config,
        "summary": summary,
        "records": [r.to_dict() for r in records],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _print_summary(summary: dict[str, Any], out_path: Path) -> None:
    print(
        "[summary] "
        f"candidates={summary['candidate_count']} "
        f"selected={summary['selected_count']} "
        f"evaluated={summary['evaluated_count']} "
        f"ok={summary['ok_count']} "
        f"errors={summary['error_count']} "
        f"certified={summary['certified_count']}"
    )
    if summary.get("avg_time_ms") is not None:
        print(f"[summary] avg_time_ms={summary['avg_time_ms']}")
    if summary.get("avg_comm_bytes") is not None:
        print(f"[summary] avg_comm_bytes={summary['avg_comm_bytes']}")
    print(f"[out] {out_path}")


def _make_batch_artifact_dir(
    prefix: str, keep_artifacts: bool
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if keep_artifacts:
        return Path(tempfile.mkdtemp(prefix=prefix)), None
    tmp_obj = tempfile.TemporaryDirectory(prefix=prefix)
    return Path(tmp_obj.name), tmp_obj


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--mnist-subtract",
        type=float,
        default=0.0,
        help="Subtract this value after scaling MNIST pixels into [0,1].",
    )
    p.add_argument("--start", required=True, type=int)
    p.add_argument("--num", required=True, type=int)
    p.add_argument("--eps", required=True, type=float)
    p.add_argument(
        "--method",
        choices=["cryptflow-crown", "plain-crown", "ckks-crown", "nexus-crown"],
        required=True,
    )
    p.add_argument("--backend", choices=["ot", "he"], default="ot")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--no-filter-correct", action="store_true")

    p.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    p.add_argument("--bitlength", type=int, default=41)
    p.add_argument("--scale-bits", type=int, default=18)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--ip", default="127.0.0.1")
    p.add_argument("--port", type=int, default=42000)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--keep-artifacts", action="store_true")
    p.add_argument(
        "--ckks-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "he" / "openfhe-ckks",
    )
    p.add_argument("--ckks-depth", type=int, default=60)
    p.add_argument("--ckks-scale-bits", type=int, default=28)
    p.add_argument("--ckks-sign-deg", type=int, default=9)
    p.add_argument("--ckks-recip-deg", type=int, default=9)
    p.add_argument(
        "--nexus-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "he" / "nexus-ckks",
    )
    p.add_argument("--nexus-build-dir", type=Path, default=None)
    p.add_argument("--nexus-max-abs-err", type=float, default=0.05)
    p.add_argument("--nexus-max-rel-err", type=float, default=0.50)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ensure_json_output_path(args.out)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.eps < 0.0:
        raise SystemExit("--eps must be >= 0")

    images, labels = load_mnist_test_set(args.data_root, subtract=args.mnist_subtract)
    evaluator = ModelEvaluator(args.onnx)
    if images.shape[1] != evaluator.input_dim:
        raise SystemExit(
            f"MNIST input dimension {images.shape[1]} != model input dimension {evaluator.input_dim}"
        )

    end = args.start + args.num
    if args.start < 0 or args.num <= 0 or end > images.shape[0]:
        raise SystemExit(
            f"Invalid interval [{args.start}, {end}) for MNIST test size {images.shape[0]}"
        )

    candidate_images = images[args.start:end]
    logits = evaluator.logits_batch(candidate_images)
    preds = np.argmax(logits, axis=1).astype(np.int64)
    targets = min_logit_targets_from_logits(logits, preds)
    full_preds = np.full(images.shape[0], -1, dtype=np.int64)
    full_targets = np.full(images.shape[0], -1, dtype=np.int64)
    full_preds[args.start:end] = preds
    full_targets[args.start:end] = targets
    target_by_index = {i: int(full_targets[i]) for i in range(args.start, end)}
    selected = select_candidate_samples(
        images,
        labels,
        full_preds,
        start=args.start,
        num=args.num,
        filter_correct=not args.no_filter_correct,
    )

    config = {
        "onnx": str(args.onnx),
        "data_root": str(args.data_root),
        "mnist_subtract": args.mnist_subtract,
        "start": args.start,
        "num": args.num,
        "eps": args.eps,
        "method": args.method,
        "filter_correct": not args.no_filter_correct,
        "decision_mode": "targeted",
        "target_policy": "min_logits",
        "backend": args.backend if args.method == "cryptflow-crown" else None,
        "bitlength": args.bitlength if args.method == "cryptflow-crown" else None,
        "scale_bits": args.scale_bits if args.method == "cryptflow-crown" else None,
        "threads": args.threads if args.method == "cryptflow-crown" else None,
        "ckks_repo": str(args.ckks_repo) if args.method == "ckks-crown" else None,
        "ckks_depth": args.ckks_depth if args.method == "ckks-crown" else None,
        "ckks_scale_bits": args.ckks_scale_bits if args.method == "ckks-crown" else None,
        "ckks_sign_deg": args.ckks_sign_deg if args.method == "ckks-crown" else None,
        "ckks_recip_deg": args.ckks_recip_deg if args.method == "ckks-crown" else None,
        "nexus_repo": str(args.nexus_repo) if args.method == "nexus-crown" else None,
        "nexus_build_dir": str(args.nexus_build_dir) if args.method == "nexus-crown" and args.nexus_build_dir is not None else None,
        "nexus_max_abs_err": args.nexus_max_abs_err if args.method == "nexus-crown" else None,
        "nexus_max_rel_err": args.nexus_max_rel_err if args.method == "nexus-crown" else None,
    }

    records: list[CrownSampleResult] = []
    if args.method == "plain-crown":
        for sample in selected:
            records.append(
                run_plain_crown_sample(
                    layers=evaluator.layers,
                    x=sample.x,
                    eps=args.eps,
                    index=sample.index,
                    label=sample.label,
                    pred=sample.pred,
                    target=target_by_index[sample.index],
                )
            )
    elif args.method == "cryptflow-crown":
        batch_dir, batch_tmp = _make_batch_artifact_dir(
            "batch_crown_", args.keep_artifacts
        )
        try:
            mlp_path = batch_dir / f"{args.onnx.stem}.crownmlp"
            prepare_crown_mlp(args.onnx, mlp_path)
            for offset, sample in enumerate(selected):
                records.append(
                    run_cryptflow_crown_sample(
                        onnx_path=args.onnx,
                        x=sample.x,
                        eps=args.eps,
                        index=sample.index,
                        label=sample.label,
                        pred=sample.pred,
                        target=target_by_index[sample.index],
                        backend=args.backend,
                        repo=args.repo,
                        bitlength=args.bitlength,
                        scale_bits=args.scale_bits,
                        threads=args.threads,
                        ip=args.ip,
                        port=args.port + offset,
                        timeout=args.timeout,
                        skip_build=args.skip_build,
                        keep_artifacts=args.keep_artifacts,
                        mlp_path=mlp_path,
                    )
                )
        finally:
            if batch_tmp is not None:
                batch_tmp.cleanup()
    elif args.method == "ckks-crown":
        batch_dir, batch_tmp = _make_batch_artifact_dir(
            "batch_ckks_crown_", args.keep_artifacts
        )
        try:
            mlp_path = batch_dir / f"{args.onnx.stem}.crownmlp"
            prepare_crown_mlp(args.onnx, mlp_path)
            for sample in selected:
                records.append(
                    run_ckks_crown_sample(
                        onnx_path=args.onnx,
                        x=sample.x,
                        eps=args.eps,
                        index=sample.index,
                        label=sample.label,
                        pred=sample.pred,
                        target=target_by_index[sample.index],
                        repo=args.ckks_repo,
                        depth=args.ckks_depth,
                        scale_bits=args.ckks_scale_bits,
                        sign_deg=args.ckks_sign_deg,
                        recip_deg=args.ckks_recip_deg,
                        timeout=args.timeout,
                        skip_build=args.skip_build,
                        keep_artifacts=args.keep_artifacts,
                        mlp_path=mlp_path,
                    )
                )
        finally:
            if batch_tmp is not None:
                batch_tmp.cleanup()
    else:
        batch_dir, batch_tmp = _make_batch_artifact_dir(
            "batch_nexus_crown_", args.keep_artifacts
        )
        try:
            mlp_path = batch_dir / f"{args.onnx.stem}.crownmlp"
            prepare_crown_mlp(args.onnx, mlp_path)
            for sample in selected:
                records.append(
                    run_nexus_crown_sample(
                        onnx_path=args.onnx,
                        x=sample.x,
                        eps=args.eps,
                        index=sample.index,
                        label=sample.label,
                        pred=sample.pred,
                        target=target_by_index[sample.index],
                        repo=args.nexus_repo,
                        timeout=args.timeout,
                        skip_build=args.skip_build,
                        keep_artifacts=args.keep_artifacts,
                        build_dir=args.nexus_build_dir,
                        mlp_path=mlp_path,
                        nexus_max_abs_err=args.nexus_max_abs_err,
                        nexus_max_rel_err=args.nexus_max_rel_err,
                    )
                )
        finally:
            if batch_tmp is not None:
                batch_tmp.cleanup()

    summary = summarize_records(
        records, candidate_count=args.num, selected_count=len(selected)
    )
    write_results(args.out, config, summary, records)
    _print_summary(summary, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
