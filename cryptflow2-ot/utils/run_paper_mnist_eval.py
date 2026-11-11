#!/usr/bin/env python3
"""Run paper-model MNIST CryptFlow2-CROWN evaluation with grouped summaries."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from convert_keras_hdf5_to_onnx import export_layers_to_onnx, load_keras_dense_layers  # noqa: E402
from crown_runner import (  # noqa: E402
    CrownSampleResult,
    prepare_crown_mlp,
    run_cryptflow_crown_sample,
    run_plain_crown_sample,
)
from mnist_data import load_mnist_test_set  # noqa: E402
from model_eval import ModelEvaluator, min_logit_targets_from_logits  # noqa: E402
from run_cryptflow_full_eval import build_plaintext_prefilter_records, flush_combination_json, localize_record_logs  # noqa: E402
from summarize_cryptflow_eval import summarize_raw_dir, write_json, write_tables  # noqa: E402


PAPER_MODELS: dict[str, dict[str, Any]] = {
    "2x20": {
        "zip_member": "mnist_2layer_relu_20_best",
        "onnx": Path("models/mnistfc-paper/mnist-paper-2x20.onnx"),
        "epsilons": [0.015, 0.045, 0.1],
        "timeout_sec": 900,
        "memory_limit_gb": 16,
    },
    "3x20": {
        "zip_member": "mnist_3layer_relu_20_best",
        "onnx": Path("models/mnistfc-paper/mnist-paper-3x20.onnx"),
        "epsilons": [0.015, 0.03, 0.1],
        "timeout_sec": 900,
        "memory_limit_gb": 16,
    },
}

MNIST_SUBTRACT = 0.5


def _safe_eps(eps: float) -> str:
    return f"{eps:.17g}".replace(".", "p").replace("-", "m")


def _raw_name(model: str, eps: float) -> str:
    return f"mnist_{model}_eps{_safe_eps(eps)}.json"


def _gap_errors(plain_gap: float | None, secure_gap: float | None) -> tuple[float | None, float | None]:
    if plain_gap is None or secure_gap is None:
        return None, None
    abs_err = abs(float(secure_gap) - float(plain_gap))
    rel_err = abs_err / max(abs(float(plain_gap)), 1e-6)
    return abs_err, rel_err


def ensure_paper_models(zip_path: Path, extract_root: Path) -> dict[str, Path]:
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for spec in PAPER_MODELS.values():
            target = extract_root / str(spec["zip_member"])
            if not target.exists():
                archive.extract(str(spec["zip_member"]), extract_root)
    resolved: dict[str, Path] = {}
    for model_name, spec in PAPER_MODELS.items():
        h5_path = extract_root / str(spec["zip_member"])
        onnx_path = Path(spec["onnx"])
        if not onnx_path.exists():
            layers = load_keras_dense_layers(h5_path)
            export_layers_to_onnx(
                layers,
                onnx_path,
                producer_name="secure-crown-artifact.paper-mnist-converter",
            )
        resolved[model_name] = onnx_path
    return resolved


def run_combination(
    *,
    model_name: str,
    onnx_path: Path,
    eps: float,
    args: argparse.Namespace,
    images: np.ndarray,
    labels: np.ndarray,
) -> list[CrownSampleResult]:
    evaluator = ModelEvaluator(onnx_path)
    if images.shape[1] != evaluator.input_dim:
        raise ValueError(
            f"MNIST input dimension {images.shape[1]} != model input dimension {evaluator.input_dim}"
        )

    prefix_images = images[: args.num_samples]
    logits = evaluator.logits_batch(prefix_images)
    preds = np.argmax(logits, axis=1).astype(np.int64)
    targets = min_logit_targets_from_logits(logits, preds)
    full_preds = np.full(labels.shape[0], -1, dtype=np.int64)
    full_targets = np.full(labels.shape[0], -1, dtype=np.int64)
    full_preds[: args.num_samples] = preds
    full_targets[: args.num_samples] = targets

    plain_results: dict[int, CrownSampleResult] = {}
    for index in range(min(args.num_samples, labels.shape[0])):
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
        model=model_name,
        eps=float(eps),
        images=images,
        labels=labels,
        preds=full_preds,
        targets=full_targets,
        plain_results=plain_results,
        secure_n=args.num_samples,
    )

    out_path = args.raw_dir / _raw_name(model_name, eps)
    combo_config = {
        "dataset": "mnist",
        "model": model_name,
        "onnx": str(onnx_path),
        "eps": float(eps),
        "secure_n": args.num_samples,
        "backend": args.backend,
        "bitlength": args.bitlength,
        "scale_bits": args.scale_bits,
        "threads": args.threads,
        "mnist_subtract": MNIST_SUBTRACT,
        "paper_model": True,
        "decision_mode": "targeted",
        "target_policy": "min_logits",
    }
    flush_combination_json(out_path, config=combo_config, records=records)

    model_dir = args.out_dir / "prepared_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    mlp_path = model_dir / f"{onnx_path.stem}.crownmlp"
    prepare_crown_mlp(onnx_path, mlp_path)

    timeout_sec = int(PAPER_MODELS[model_name]["timeout_sec"])
    memory_limit_gb = int(PAPER_MODELS[model_name]["memory_limit_gb"])
    secure_records_by_index: dict[int, CrownSampleResult] = {}
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
            raw_dir=args.raw_dir,
            combination_stem=out_path.stem,
        )
        gap_abs_err, gap_rel_err = _gap_errors(item["plain_gap"], secure.sci_gap)
        secure_records_by_index[int(item["index"])] = CrownSampleResult(
            **{
                **secure.to_dict(),
                "dataset": "mnist",
                "model": model_name,
                "plaintext_group": str(item["plaintext_group"]),
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
        records = sorted(merged, key=lambda record: record.index)
        flush_combination_json(out_path, config=combo_config, records=records)
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("mnist_models.zip"))
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "privacy_crown_paper_mnist_models",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/cryptflow2_crown_paper_models"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--backend", choices=["ot", "he"], default="ot")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--bitlength", type=int, default=44)
    parser.add_argument("--scale-bits", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=51000)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument(
        "--only-model",
        action="append",
        choices=sorted(PAPER_MODELS),
        default=[],
        help="Restrict execution to one or more paper models.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip combinations whose raw JSON already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir = args.out_dir / "raw"
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    onnx_by_model = ensure_paper_models(args.zip, args.extract_dir)
    images, labels = load_mnist_test_set(args.data_root, subtract=MNIST_SUBTRACT)

    selected_models = args.only_model or sorted(PAPER_MODELS)
    manifest: list[dict[str, Any]] = []
    port_cursor = int(args.port)
    for model_name in selected_models:
        onnx_path = onnx_by_model[model_name]
        for eps in PAPER_MODELS[model_name]["epsilons"]:
            raw_path = args.raw_dir / _raw_name(model_name, float(eps))
            if args.resume and raw_path.exists():
                manifest.append(
                    {
                        "model": model_name,
                        "eps": float(eps),
                        "raw_path": str(raw_path),
                        "status": "resumed_existing",
                    }
                )
                continue
            args.port = port_cursor
            records = run_combination(
                model_name=model_name,
                onnx_path=onnx_path,
                eps=float(eps),
                args=args,
                images=images,
                labels=labels,
            )
            manifest.append(
                {
                    "model": model_name,
                    "eps": float(eps),
                    "raw_path": str(raw_path),
                    "ok_count": sum(1 for record in records if record.status == "ok"),
                    "misclassified": sum(
                        1 for record in records if record.status == "plain_misclassified"
                    ),
                }
            )
            port_cursor += args.num_samples + 10

    summary = summarize_raw_dir(args.raw_dir)
    summary["paper_model_eval"] = {
        "zip": str(args.zip),
        "extract_dir": str(args.extract_dir),
        "num_samples": args.num_samples,
        "mnist_subtract": MNIST_SUBTRACT,
        "bitlength": args.bitlength,
        "scale_bits": args.scale_bits,
        "backend": args.backend,
        "manifest": manifest,
    }
    write_json(args.out_dir / "summary.json", summary)
    write_tables(args.out_dir / "tables", summary)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[paper-eval] models={len(selected_models)} raw={len(list(args.raw_dir.glob('*.json')))} "
        f"summary={args.out_dir / 'summary.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
