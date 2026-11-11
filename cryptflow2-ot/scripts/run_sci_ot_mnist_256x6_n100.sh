#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/cryptflow2_crown_256x6_huan_41b17_reopt_n100}"
CHUNK_DIR="$OUT_DIR/chunks"
ONNX="${ONNX:-$ROOT_DIR/models/mnistfc/mnist-net_256x6.onnx}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

BITLENGTH="${BITLENGTH:-41}"
SCALE_BITS="${SCALE_BITS:-17}"
THREADS="${THREADS:-1}"
TIMEOUT="${TIMEOUT:-3600}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
CHUNK_SIZE_EPS0P015="${CHUNK_SIZE_EPS0P015:-$CHUNK_SIZE}"
CHUNK_SIZE_EPS0P1="${CHUNK_SIZE_EPS0P1:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
PORT_BASE="${PORT_BASE:-57000}"
EPS_LIST="${EPS_LIST:-0.015 0.1}"

mkdir -p "$CHUNK_DIR" "$OUT_DIR/tables"

safe_eps() {
  python3 - "$1" <<'PY'
import sys
eps = float(sys.argv[1])
if abs(eps - 0.015) < 1e-12:
    print("eps0p015")
elif abs(eps - 0.1) < 1e-12:
    print("eps0p1")
else:
    print(f"eps{eps:.17g}".replace(".", "p").replace("-", "m"))
PY
}

run_chunk() {
  local eps="$1"
  local eps_tag="$2"
  local start="$3"
  local port="$4"
  local chunk_size="$5"
  local out="$CHUNK_DIR/${eps_tag}_start${start}_num${chunk_size}.json"

  if [[ -s "$out" ]]; then
    echo "[skip] eps=$eps start=$start existing=$out"
    return 0
  fi

  echo "[run] eps=$eps start=$start num=$chunk_size port=$port"
  uv run python "$ROOT_DIR/utils/run_batch_experiment.py" \
    --method cryptflow-crown \
    --backend ot \
    --onnx "$ONNX" \
    --data-root "$DATA_ROOT" \
    --start "$start" \
    --num "$chunk_size" \
    --eps "$eps" \
    --bitlength "$BITLENGTH" \
    --scale-bits "$SCALE_BITS" \
    --threads "$THREADS" \
    --port "$port" \
    --timeout "$TIMEOUT" \
    --skip-build \
    --out "$out"
}

for eps in $EPS_LIST; do
  eps_tag="$(safe_eps "$eps")"
  eps_port_offset=0
  eps_chunk_size="$CHUNK_SIZE_EPS0P015"
  if [[ "$eps_tag" == "eps0p1" ]]; then
    eps_port_offset=6000
    eps_chunk_size="$CHUNK_SIZE_EPS0P1"
  fi
  for ((start=0; start<NUM_SAMPLES; start+=eps_chunk_size)); do
    port=$((PORT_BASE + eps_port_offset + start * 10))
    run_chunk "$eps" "$eps_tag" "$start" "$port" "$eps_chunk_size"
  done
done

uv run python - "$OUT_DIR" "$BITLENGTH" "$SCALE_BITS" "$THREADS" "$TIMEOUT" "$ONNX" "$DATA_ROOT" $EPS_LIST <<'PY'
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

out_dir = Path(sys.argv[1])
bitlength = int(sys.argv[2])
scale_bits = int(sys.argv[3])
threads = int(sys.argv[4])
timeout = int(sys.argv[5])
onnx = sys.argv[6]
data_root = sys.argv[7]
chunk_dir = out_dir / "chunks"
rel_floor = 1e-6
num_samples = 100
chunk_re = re.compile(r"_start(?P<start>\d+)_num(?P<num>\d+)\.json$")

eps_tags = []
for raw_eps in sys.argv[8:]:
    eps = float(raw_eps)
    if abs(eps - 0.015) < 1e-12:
        tag = "eps0p015"
    elif abs(eps - 0.1) < 1e-12:
        tag = "eps0p1"
    else:
        tag = f"eps{eps:.17g}".replace(".", "p").replace("-", "m")
    eps_tags.append((eps, tag))
if not eps_tags:
    raise SystemExit("no eps values provided")

def mean(values):
    values = [float(v) for v in values if v is not None]
    return statistics.fmean(values) if values else None

def fmt(value):
    if value is None:
        return "--"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "--"
    return f"{value:.6g}"

def group_of(record):
    plain_gap = record.get("plain_gap")
    if plain_gap is None:
        return record.get("plaintext_group") or "unknown"
    return "robust" if float(plain_gap) > 0.0 else "not-certified"

def annotate(record, eps):
    rec = dict(record)
    rec["dataset"] = "mnist"
    rec["model"] = "256x6"
    rec["eps"] = float(eps)
    if rec.get("status") == "ok":
        rec["plaintext_group"] = group_of(rec)
        plain_gap = rec.get("plain_gap")
        secure_gap = rec.get("secure_gap", rec.get("sci_gap"))
        if plain_gap is not None and secure_gap is not None:
            abs_err = abs(float(secure_gap) - float(plain_gap))
            rec["gap_abs_err"] = abs_err
            rec["gap_rel_err"] = abs_err / max(abs(float(plain_gap)), rel_floor)
    return rec

def chunk_meta(path):
    match = chunk_re.search(path.name)
    if not match:
        return None
    return int(match.group("start")), int(match.group("num"))

def prefer_record(old, new):
    old_ok = old.get("status") == "ok"
    new_ok = new.get("status") == "ok"
    if new_ok and not old_ok:
        return new
    if old_ok and not new_ok:
        return old
    old_num = old.get("_chunk_num")
    new_num = new.get("_chunk_num")
    if old_ok and new_ok and old_num != 1 and new_num == 1:
        return new
    return old

def load_eps(eps, tag):
    chunk_files = sorted(chunk_dir.glob(f"{tag}_start*_num*.json"))
    records_by_index = {}
    covered_candidates = set()
    for path in chunk_files:
        meta = chunk_meta(path)
        if meta:
            start, num = meta
            covered_candidates.update(i for i in range(start, min(start + num, num_samples)))
        payload = json.loads(path.read_text())
        for record in payload.get("records", []):
            rec = annotate(record, eps)
            if meta:
                rec["_chunk_file"] = str(path)
                rec["_chunk_start"] = meta[0]
                rec["_chunk_num"] = meta[1]
            key = int(rec.get("index", rec.get("sample_index", -1)))
            if key < 0:
                continue
            if key in records_by_index:
                records_by_index[key] = prefer_record(records_by_index[key], rec)
            else:
                records_by_index[key] = rec
    records = list(records_by_index.values())
    records.sort(key=lambda r: int(r.get("index", r.get("sample_index", -1))))
    statuses = Counter(str(r.get("status")) for r in records)
    selected_indices = {
        int(r.get("index", r.get("sample_index", -1)))
        for r in records
        if int(r.get("index", r.get("sample_index", -1))) >= 0
    }
    eps_payload = {
        "config": {
            "backend": "ot",
            "bitlength": bitlength,
            "chunk_files": [str(path) for path in chunk_files],
            "chunking": "mixed",
            "data_root": str(data_root),
            "decision_mode": "targeted",
            "eps": float(eps),
            "method": "cryptflow-crown",
            "model": "256x6",
            "num": num_samples,
            "onnx": str(onnx),
            "scale_bits": scale_bits,
            "start": 0,
            "target_policy": "min_logits",
            "threads": threads,
            "timeout_s_per_sample": timeout,
        },
        "records": records,
        "summary": {
            "candidate_count": len(covered_candidates),
            "selected_count": len(selected_indices),
            "evaluated_count": len(records),
            "ok_count": statuses.get("ok", 0),
            "error_count": len(records) - statuses.get("ok", 0),
            "certified_count": sum(1 for r in records if r.get("status") == "ok" and r.get("certified") is True),
            "statuses": dict(sorted(statuses.items())),
        },
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(eps_payload, indent=2, sort_keys=True) + "\n")
    return eps_payload

def summarize_group(records, eps, plaintext_group):
    ok = [r for r in records if r.get("status") == "ok"]
    consistent = 0
    for r in ok:
        plain_cert = plaintext_group == "robust"
        if bool(r.get("certified")) == plain_cert:
            consistent += 1
    return {
        "dataset": "mnist",
        "model": "256x6",
        "eps": float(eps),
        "plaintext_group": plaintext_group,
        "ok": len(ok),
        "gap_mae": mean(r.get("gap_abs_err") for r in ok),
        "gap_mre": mean(r.get("gap_rel_err") for r in ok),
        "consistency": (consistent / len(ok)) if ok else None,
        "offline_time_s": mean((r.get("offline_time_ms") or r.get("amortized_offline_time_ms")) / 1000 for r in ok if (r.get("offline_time_ms") or r.get("amortized_offline_time_ms")) is not None),
        "online_time_s": mean(r.get("online_time_ms") / 1000 for r in ok if r.get("online_time_ms") is not None),
        "total_time_s": mean((r.get("total_time_ms") or r.get("time_ms")) / 1000 for r in ok if (r.get("total_time_ms") or r.get("time_ms")) is not None),
        "offline_comm_mb": mean(r.get("offline_comm_bytes") / 1_000_000 for r in ok if r.get("offline_comm_bytes") is not None),
        "online_comm_mb": mean(r.get("online_comm_bytes") / 1_000_000 for r in ok if r.get("online_comm_bytes") is not None),
        "total_comm_mb": mean((r.get("total_comm_bytes") or r.get("comm_bytes")) / 1_000_000 for r in ok if (r.get("total_comm_bytes") or r.get("comm_bytes")) is not None),
    }

eps_payloads = [load_eps(eps, tag) for eps, tag in eps_tags]
groups = []
reliability = []
for payload in eps_payloads:
    eps = payload["config"]["eps"]
    records = payload["records"]
    by_group = defaultdict(list)
    for record in records:
        if record.get("status") == "ok":
            by_group[record.get("plaintext_group", "unknown")].append(record)
    for group_name in sorted(by_group):
        groups.append(summarize_group(by_group[group_name], eps, group_name))
    statuses = Counter(str(r.get("status")) for r in records)
    reliability.append({
        "model": "256x6",
        "eps": eps,
        "candidate_count": payload["summary"]["candidate_count"],
        "selected_count": payload["summary"]["selected_count"],
        "evaluated_count": len(records),
        "ok_count": statuses.get("ok", 0),
        "error_count": len(records) - statuses.get("ok", 0),
        "certified_count": sum(1 for r in records if r.get("status") == "ok" and r.get("certified") is True),
        "statuses": dict(sorted(statuses.items())),
    })

summary = {
    "config": {
        "backend": "cryptflow2-ot",
        "bitlength": bitlength,
        "chunking": {
            "eps0p015": "5-sample chunks",
            "eps0p1": "1-sample chunks, with successful single-sample reruns preferred over earlier port-collision records",
        },
        "model": onnx,
        "paper_label": "7x[256]",
        "relative_error_floor": rel_floor,
        "samples": "first 100 MNIST test samples, correctly classified only",
        "scale_bits": scale_bits,
        "target_policy": "min_logits",
        "threads": threads,
        "timeout_s_per_sample": timeout,
    },
    "groups": groups,
    "reliability": reliability,
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

tables_dir = out_dir / "tables"
tables_dir.mkdir(parents=True, exist_ok=True)
main_lines = [
    "| dataset | model | eps | plaintext group | ok | gap MAE | gap MRE | consistency | offline time s | online time s | total time s | offline communication MB | online communication MB | total communication MB |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for g in groups:
    main_lines.append(
        "| {dataset} | {model} | {eps:.3g} | {plaintext_group} | {ok} | {gap_mae} | {gap_mre} | {consistency} | {offline_time_s} | {online_time_s} | {total_time_s} | {offline_comm_mb} | {online_comm_mb} | {total_comm_mb} |".format(
            dataset=g["dataset"],
            model=g["model"],
            eps=g["eps"],
            plaintext_group=g["plaintext_group"],
            ok=g["ok"],
            gap_mae=fmt(g["gap_mae"]),
            gap_mre=fmt(g["gap_mre"]),
            consistency=fmt(g["consistency"]),
            offline_time_s=fmt(g["offline_time_s"]),
            online_time_s=fmt(g["online_time_s"]),
            total_time_s=fmt(g["total_time_s"]),
            offline_comm_mb=fmt(g["offline_comm_mb"]),
            online_comm_mb=fmt(g["online_comm_mb"]),
            total_comm_mb=fmt(g["total_comm_mb"]),
        )
    )
(tables_dir / "main_table.md").write_text("\n".join(main_lines) + "\n")

rel_lines = [
    "| dataset | model | eps | candidates | selected | evaluated | ok | errors | certified | statuses |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for r in reliability:
    rel_lines.append(
        f"| mnist | 256x6 | {r['eps']:.3g} | {r['candidate_count']} | {r['selected_count']} | {r['evaluated_count']} | {r['ok_count']} | {r['error_count']} | {r['certified_count']} | "
        + ", ".join(f"{k}={v}" for k, v in r["statuses"].items())
        + " |"
    )
(tables_dir / "reliability_table.md").write_text("\n".join(rel_lines) + "\n")

print(f"[aggregate] summary={out_dir / 'summary.json'}")
print(f"[aggregate] table={tables_dir / 'main_table.md'}")
PY

echo "[done] results written under $OUT_DIR"
