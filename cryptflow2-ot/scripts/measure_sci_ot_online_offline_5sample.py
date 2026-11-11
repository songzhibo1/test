#!/usr/bin/env python3
"""Measure SCI-OT paper-split phase metrics on 5 samples per model."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from cifar10_data import CIFAR_INPUT_DIM, CIFAR_TEST_BATCH  # noqa: E402
from crown_runner import (  # noqa: E402
    CrownSampleResult,
    prepare_crown_mlp,
    run_cryptflow_crown_sample,
    run_plain_crown_sample,
)
from mnist_data import load_mnist_test_set  # noqa: E402
from model_eval import ModelEvaluator, min_logit_targets_from_logits  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    dataset: str
    model: str
    onnx: Path
    eps: float
    bitlength: int
    scale_bits: int
    timeout: int
    memory_gb: int


SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "mnist",
        "2x20",
        ROOT / "models/mnistfc-paper/mnist-paper-2x20.onnx",
        0.015,
        41,
        17,
        300,
        16,
    ),
    ModelSpec(
        "mnist",
        "3x20",
        ROOT / "models/mnistfc-paper/mnist-paper-3x20.onnx",
        0.015,
        41,
        17,
        300,
        16,
    ),
    ModelSpec(
        "mnist",
        "256x2",
        ROOT / "models/mnistfc/mnist-net_256x2.onnx",
        0.015,
        41,
        17,
        1200,
        32,
    ),
    ModelSpec(
        "mnist",
        "256x4",
        ROOT / "models/mnistfc/mnist-net_256x4.onnx",
        0.015,
        41,
        17,
        2400,
        48,
    ),
    ModelSpec(
        "mnist",
        "256x6",
        ROOT / "models/mnistfc/mnist-net_256x6.onnx",
        0.015,
        41,
        17,
        3600,
        64,
    ),
    ModelSpec(
        "cifar10",
        "5x100",
        ROOT / "models/cifar10/cifar10_5x100.onnx",
        0.002,
        39,
        13,
        2400,
        48,
    ),
    ModelSpec(
        "cifar10",
        "7x100",
        ROOT / "models/cifar10/cifar10_7x100.onnx",
        0.001,
        39,
        13,
        3600,
        64,
    ),
    ModelSpec(
        "cifar10",
        "10x200",
        ROOT / "models/cifar10/cifar10_10x200.onnx",
        0.001,
        39,
        13,
        5400,
        96,
    ),
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def avg(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def total(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) if present else None


def scaled_avg(values: list[float | int | None], scale: float) -> float | None:
    value = avg(values)
    return value / scale if value is not None else None


def load_dataset(spec: ModelSpec, data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    if spec.dataset == "mnist":
        return load_mnist_test_set(data_root)
    if spec.dataset == "cifar10":
        test_file = Path(data_root) / CIFAR_TEST_BATCH
        with test_file.open("rb") as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = pickle.load(f, encoding="bytes")
        raw = np.asarray(data[b"data"], dtype=np.float32)
        labels = np.asarray(data[b"labels"], dtype=np.int64)
        if raw.ndim != 2 or raw.shape[1] != CIFAR_INPUT_DIM:
            raise ValueError(f"CIFAR10 data shape must be (N, 3072), got {raw.shape}")
        chw = raw.reshape((-1, 3, 32, 32))
        nhwc_flat = np.transpose(chw, (0, 2, 3, 1)).reshape((-1, CIFAR_INPUT_DIM))
        return ((nhwc_flat / 255.0) - 0.5).astype(np.float32).copy(), labels.copy()
    raise ValueError(f"unknown dataset: {spec.dataset}")


def candidate_indices(
    *,
    evaluator: ModelEvaluator,
    images: np.ndarray,
    labels: np.ndarray,
    scan_prefix: int,
    samples: int,
) -> list[dict[str, Any]]:
    upper = min(scan_prefix, int(labels.shape[0]))
    logits = evaluator.logits_batch(images[:upper])
    preds = np.argmax(logits, axis=1).astype(np.int64)
    targets = min_logit_targets_from_logits(logits, preds)
    selected: list[dict[str, Any]] = []
    for index in range(upper):
        label = int(labels[index])
        pred = int(preds[index])
        if pred != label:
            continue
        selected.append(
            {
                "index": index,
                "label": label,
                "pred": pred,
                "target": int(targets[index]),
                "x": np.asarray(images[index], dtype=np.float64).reshape(-1),
            }
        )
        if len(selected) >= samples:
            break
    return selected


def summarize_records(records: list[CrownSampleResult]) -> dict[str, Any]:
    ok = [record for record in records if record.status == "ok"]
    return {
        "attempted": len(records),
        "ok": len(ok),
        "errors": len(records) - len(ok),
        "indices": [record.index for record in ok],
        "avg_total_time_s": scaled_avg([record.time_ms for record in ok], 1000.0),
        "avg_total_comm_mb": scaled_avg([record.comm_bytes for record in ok], 1_000_000.0),
        "avg_paper_offline_time_s": scaled_avg(
            [record.paper_offline_time_ms for record in ok], 1000.0
        ),
        "avg_paper_online_time_s": scaled_avg(
            [record.paper_online_time_ms for record in ok], 1000.0
        ),
        "avg_paper_offline_comm_mb": scaled_avg(
            [record.paper_offline_comm_bytes for record in ok], 1_000_000.0
        ),
        "avg_paper_online_comm_mb": scaled_avg(
            [record.paper_online_comm_bytes for record in ok], 1_000_000.0
        ),
        "avg_split_fallback_calls": avg([record.split_ot_fallback_calls for record in ok]),
        "avg_split_fallback_ots": avg([record.split_ot_fallback_ots for record in ok]),
        "total_split_fallback_calls": total(
            [record.split_ot_fallback_calls for record in ok]
        ),
        "total_split_fallback_ots": total([record.split_ot_fallback_ots for record in ok]),
        "paper_accounting_complete_all": all(
            bool(record.paper_accounting_complete) for record in ok
        )
        if ok
        else False,
        "statuses": {
            status: sum(1 for record in records if record.status == status)
            for status in sorted({record.status for record in records})
        },
    }


def load_existing_records(path: Path) -> list[CrownSampleResult]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    return [CrownSampleResult(**record) for record in payload.get("records", [])]


def run_spec(
    *,
    spec: ModelSpec,
    args: argparse.Namespace,
    spec_offset: int,
) -> dict[str, Any]:
    out_path = args.out_dir / "raw" / f"{spec.dataset}_{spec.model}.json"
    records = load_existing_records(out_path) if args.resume else []
    completed = {record.index for record in records if record.status == "ok"}
    evaluator = ModelEvaluator(spec.onnx)
    images, labels = load_dataset(spec, args.data_root)
    if images.shape[1] != evaluator.input_dim:
        raise ValueError(
            f"{spec.dataset}/{spec.model}: data dim {images.shape[1]} != model input dim {evaluator.input_dim}"
        )
    selected = candidate_indices(
        evaluator=evaluator,
        images=images,
        labels=labels,
        scan_prefix=args.scan_prefix,
        samples=args.samples,
    )
    if len(selected) < args.samples:
        raise ValueError(
            f"{spec.dataset}/{spec.model}: only {len(selected)} correct samples in first {args.scan_prefix}"
        )
    selected_indices = {int(item["index"]) for item in selected}
    records = [record for record in records if int(record.index) in selected_indices]
    completed = {record.index for record in records if record.status == "ok"}

    mlp_path = args.out_dir / "prepared_models" / f"{spec.onnx.stem}.crownmlp"
    mlp_path.parent.mkdir(parents=True, exist_ok=True)
    if not mlp_path.is_file():
        prepare_crown_mlp(spec.onnx, mlp_path)

    env = {"CROWN_OT_PRECOMP_BATCH": str(args.precomp_batch)}
    for sample_offset, item in enumerate(selected):
        if int(item["index"]) in completed:
            continue
        print(
            "[run] "
            f"{spec.dataset}/{spec.model} sample={sample_offset + 1}/{len(selected)} "
            f"index={item['index']} eps={spec.eps} bl={spec.bitlength} sf={spec.scale_bits}",
            flush=True,
        )
        plain = run_plain_crown_sample(
            layers=evaluator.layers,
            x=item["x"],
            eps=spec.eps,
            index=int(item["index"]),
            label=int(item["label"]),
            pred=int(item["pred"]),
            target=int(item["target"]),
        )
        secure = run_cryptflow_crown_sample(
            onnx_path=spec.onnx,
            x=item["x"],
            eps=spec.eps,
            index=int(item["index"]),
            label=int(item["label"]),
            pred=int(item["pred"]),
            target=int(item["target"]),
            backend="ot",
            repo=args.repo,
            bitlength=spec.bitlength,
            scale_bits=spec.scale_bits,
            threads=args.threads,
            ip=args.ip,
            port=args.base_port + spec_offset * 1000 + sample_offset,
            timeout=spec.timeout,
            memory_limit_gb=spec.memory_gb,
            skip_build=args.skip_build,
            keep_artifacts=args.keep_artifacts,
            mlp_path=mlp_path,
            extra_env=env,
        )
        records.append(
            CrownSampleResult(
                **{
                    **secure.to_dict(),
                    "dataset": spec.dataset,
                    "model": spec.model,
                    "plain_gap": plain.plain_gap,
                    "plaintext_group": "robust"
                    if plain.plain_gap is not None and plain.plain_gap > 0
                    else "not-certified",
                }
            )
        )
        write_spec_payload(out_path, spec, selected, records, args)

    write_spec_payload(out_path, spec, selected, records, args)
    return json.loads(out_path.read_text())


def write_spec_payload(
    out_path: Path,
    spec: ModelSpec,
    selected: list[dict[str, Any]],
    records: list[CrownSampleResult],
    args: argparse.Namespace,
) -> None:
    payload = {
        "config": {
            **asdict(spec),
            "onnx": str(spec.onnx),
            "data_root": str(args.data_root),
            "scan_prefix": args.scan_prefix,
            "samples": args.samples,
            "threads": args.threads,
            "precomp_batch": args.precomp_batch,
            "sample_policy": "first correctly classified samples from prefix",
            "cifar_preprocess": "CIFAR pickle CHW -> NHWC flatten, x/255 - 0.5"
            if spec.dataset == "cifar10"
            else None,
            "decision_mode": "targeted",
            "target_policy": "min_logits",
            "paper_accounting_note": (
                "paper_accounting_complete remains false while SCI COT/block-OT "
                "fallback paths are not split into preprocessing."
            ),
        },
        "selected": [
            {
                key: value
                for key, value in item.items()
                if key in {"index", "label", "pred", "target"}
            }
            for item in selected
        ],
        "records": [record.to_dict() for record in records],
        "summary": summarize_records(records),
    }
    write_json(out_path, payload)


def write_summary(out_dir: Path, payloads: list[dict[str, Any]]) -> None:
    rows = []
    for payload in payloads:
        config = payload["config"]
        summary = payload["summary"]
        rows.append(
            {
                "dataset": config["dataset"],
                "model": config["model"],
                "eps": config["eps"],
                "ok": summary["ok"],
                "avg_paper_offline_time_s": summary["avg_paper_offline_time_s"],
                "avg_paper_online_time_s": summary["avg_paper_online_time_s"],
                "avg_total_time_s": summary["avg_total_time_s"],
                "avg_paper_offline_comm_mb": summary["avg_paper_offline_comm_mb"],
                "avg_paper_online_comm_mb": summary["avg_paper_online_comm_mb"],
                "avg_total_comm_mb": summary["avg_total_comm_mb"],
                "avg_split_fallback_calls": summary["avg_split_fallback_calls"],
                "avg_split_fallback_ots": summary["avg_split_fallback_ots"],
                "paper_accounting_complete_all": summary[
                    "paper_accounting_complete_all"
                ],
            }
        )
    write_json(out_dir / "summary.json", {"rows": rows})

    lines = [
        "| Dataset | Model | eps | ok | offline s | online s | total s | offline MB | online MB | total MB | avg fallback calls | avg fallback OTs | complete |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value: Any, digits: int = 2) -> str:
            if value is None:
                return "--"
            if isinstance(value, bool):
                return str(value).lower()
            if isinstance(value, int):
                return str(value)
            return f"{float(value):.{digits}f}"

        lines.append(
            "| {dataset} | {model} | {eps} | {ok} | {offline_s} | {online_s} | {total_s} | {offline_mb} | {online_mb} | {total_mb} | {fallback_calls} | {fallback_ots} | {complete} |".format(
                dataset=row["dataset"],
                model=row["model"],
                eps=fmt(row["eps"], 4),
                ok=row["ok"],
                offline_s=fmt(row["avg_paper_offline_time_s"]),
                online_s=fmt(row["avg_paper_online_time_s"]),
                total_s=fmt(row["avg_total_time_s"]),
                offline_mb=fmt(row["avg_paper_offline_comm_mb"]),
                online_mb=fmt(row["avg_paper_online_comm_mb"]),
                total_mb=fmt(row["avg_total_comm_mb"]),
                fallback_calls=fmt(row["avg_split_fallback_calls"], 1),
                fallback_ots=fmt(row["avg_split_fallback_ots"], 1),
                complete=fmt(row["paper_accounting_complete_all"]),
            )
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/cryptflow2_paper_split_5sample")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--scan-prefix", type=int, default=100)
    parser.add_argument("--precomp-batch", type=int, default=262144)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=54000)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-model", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_specs = [
        spec for spec in SPECS if not args.only_model or spec.model in args.only_model
    ]
    payloads: list[dict[str, Any]] = []
    for spec_offset, spec in enumerate(selected_specs):
        payloads.append(run_spec(spec=spec, args=args, spec_offset=spec_offset))
        write_summary(args.out_dir, payloads)
        print(f"[done] {spec.dataset}/{spec.model}", flush=True)
    write_summary(args.out_dir, payloads)
    print(f"[summary] {args.out_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
