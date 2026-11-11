"""Reusable single-sample CROWN runners."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix fallback
    resource = None

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from format_convert import export_crown_mlp_bin, read_crown_mlp_bin

Layer = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class CryptflowParsedMetrics:
    plain_gap: float | None = None
    sci_gap: float | None = None
    abs_err_gap: float | None = None
    time_ms: float | None = None
    comm_bytes: float | None = None
    offline_time_ms: float | None = None
    online_time_ms: float | None = None
    total_time_ms: float | None = None
    offline_comm_bytes: float | None = None
    online_comm_bytes: float | None = None
    total_comm_bytes: float | None = None
    raw_offline_time_ms: float | None = None
    amortized_offline_time_ms: float | None = None
    accounting_model: str | None = None
    perf_schema_version: int | None = None
    paper_offline_time_ms: float | None = None
    paper_online_time_ms: float | None = None
    paper_total_time_ms: float | None = None
    paper_offline_comm_bytes: float | None = None
    paper_online_comm_bytes: float | None = None
    paper_total_comm_bytes: float | None = None
    split_ot_precomp_batch: int | None = None
    split_ot_preprocess_calls: int | None = None
    split_ot_preprocess_ots: int | None = None
    split_ot_fallback_calls: int | None = None
    split_ot_fallback_ots: int | None = None
    paper_accounting_complete: bool | None = None
    paper_accounting_model: str | None = None


@dataclass(frozen=True)
class TwoPartyRunResult:
    party1_exit: int
    party2_exit: int
    party1_stdout: str
    party2_stdout: str
    party1_stdout_path: str | None = None
    party2_stdout_path: str | None = None
    peak_rss_kb: int | None = None

    def __iter__(self):
        yield self.party1_exit
        yield self.party2_exit
        yield self.party1_stdout
        yield self.party2_stdout


@dataclass(frozen=True)
class CkksParsedMetrics:
    plain_gap: float | None = None
    sci_gap: float | None = None
    abs_err_gap: float | None = None
    plain_lb: float | None = None
    plain_ub: float | None = None
    ckks_lb: float | None = None
    ckks_ub: float | None = None
    abs_err_lb: float | None = None
    abs_err_ub: float | None = None
    rel_err_lb: float | None = None
    rel_err_ub: float | None = None
    bound_fields_available: bool = False
    bound_fields_unavailable_reason: str | None = "ckks logs did not emit lower/upper bound fields"
    time_ms: float | None = None
    comm_bytes: float | None = None
    ring_dim: int | None = None
    depth: int | None = None
    effective_depth: int | None = None
    scale_bits: int | None = None
    sign_deg: int | None = None
    recip_deg: int | None = None
    act_range: float | None = None
    coeff_range: float | None = None
    denom_lo: float | None = None
    denom_hi: float | None = None
    batch_size: int | None = None
    security_level: str | None = None
    secret_key_dist: str | None = None
    sample: str | None = None
    bootstrap_coeff_abs: bool | None = None
    bootstrap_slots: int | None = None
    bootstrap_level_enc: int | None = None
    bootstrap_level_dec: int | None = None
    bootstrap_dim1_enc: int | None = None
    bootstrap_dim1_dec: int | None = None
    bootstrap_correction: int | None = None
    bootstrap_iterations: int | None = None
    bootstrap_setup_ms: float | None = None
    bootstrap_keygen_ms: float | None = None
    bootstrap_coeff_abs_count: int = 0
    bootstrap_coeff_abs_total_ms: float | None = None


@dataclass(frozen=True)
class NexusParsedMetrics:
    status: str | None = None
    status_detail: str | None = None
    plain_gap: float | None = None
    nexus_gap: float | None = None
    nexus_abs_err_gap: float | None = None
    nexus_rel_err_gap: float | None = None
    plain_lb: float | None = None
    plain_ub: float | None = None
    nexus_lb: float | None = None
    nexus_ub: float | None = None
    nexus_abs_err_lb: float | None = None
    nexus_abs_err_ub: float | None = None
    nexus_rel_err_lb: float | None = None
    nexus_rel_err_ub: float | None = None
    bound_fields_available: bool = False
    bound_fields_unavailable_reason: str | None = (
        "nexus logs did not emit lower/upper bound fields"
    )
    time_ms: float | None = None
    artifact_bytes: float | None = None
    comm_bytes: float | None = None
    bootstrap_coeff_abs_count: int = 0
    bootstrap_coeff_abs_total_ms: float | None = None
    provider_decryptor_used: bool | None = None
    privacy_notes: str | None = None
    disallowed_helpers_found: str | None = None


@dataclass(frozen=True)
class CrownSampleResult:
    index: int
    label: int
    pred: int
    eps: float
    method: str
    status: str
    certified: bool | None = None
    plain_gap: float | None = None
    sci_gap: float | None = None
    abs_err_gap: float | None = None
    dataset: str | None = None
    model: str | None = None
    plaintext_group: str | None = None
    gap_abs_err: float | None = None
    gap_rel_err: float | None = None
    plain_lb: float | None = None
    plain_ub: float | None = None
    ckks_lb: float | None = None
    ckks_ub: float | None = None
    abs_err_lb: float | None = None
    abs_err_ub: float | None = None
    rel_err_lb: float | None = None
    rel_err_ub: float | None = None
    bound_fields_available: bool = False
    bound_fields_unavailable_reason: str | None = None
    time_ms: float | None = None
    comm_bytes: float | None = None
    ckks_ring_dim: int | None = None
    ckks_depth: int | None = None
    ckks_effective_depth: int | None = None
    ckks_scale_bits: int | None = None
    ckks_sign_deg: int | None = None
    ckks_recip_deg: int | None = None
    ckks_act_range: float | None = None
    ckks_coeff_range: float | None = None
    ckks_denom_lo: float | None = None
    ckks_denom_hi: float | None = None
    ckks_batch_size: int | None = None
    ckks_security_level: str | None = None
    ckks_secret_key_dist: str | None = None
    ckks_sample: str | None = None
    ckks_bootstrap_coeff_abs: bool | None = None
    ckks_bootstrap_slots: int | None = None
    ckks_bootstrap_level_enc: int | None = None
    ckks_bootstrap_level_dec: int | None = None
    ckks_bootstrap_dim1_enc: int | None = None
    ckks_bootstrap_dim1_dec: int | None = None
    ckks_bootstrap_correction: int | None = None
    ckks_bootstrap_iterations: int | None = None
    ckks_bootstrap_setup_ms: float | None = None
    ckks_bootstrap_keygen_ms: float | None = None
    ckks_bootstrap_coeff_abs_count: int = 0
    ckks_bootstrap_coeff_abs_total_ms: float | None = None
    nexus_lb: float | None = None
    nexus_ub: float | None = None
    nexus_gap: float | None = None
    nexus_abs_err_lb: float | None = None
    nexus_abs_err_ub: float | None = None
    nexus_abs_err_gap: float | None = None
    nexus_rel_err_lb: float | None = None
    nexus_rel_err_ub: float | None = None
    nexus_rel_err_gap: float | None = None
    nexus_status_detail: str | None = None
    nexus_artifact_bytes: float | None = None
    nexus_comm_bytes: float | None = None
    nexus_bootstrap_coeff_abs_count: int = 0
    nexus_bootstrap_coeff_abs_total_ms: float | None = None
    nexus_provider_decryptor_used: bool | None = None
    nexus_privacy_notes: str | None = None
    nexus_disallowed_helpers_found: str | None = None
    nexus_stdout_path: str | None = None
    nexus_stderr_path: str | None = None
    nexus_mlp_path: str | None = None
    nexus_x0bin_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    offline_time_ms: float | None = None
    online_time_ms: float | None = None
    total_time_ms: float | None = None
    offline_comm_bytes: float | None = None
    online_comm_bytes: float | None = None
    total_comm_bytes: float | None = None
    raw_offline_time_ms: float | None = None
    amortized_offline_time_ms: float | None = None
    accounting_model: str | None = None
    perf_schema_version: int | None = None
    paper_offline_time_ms: float | None = None
    paper_online_time_ms: float | None = None
    paper_total_time_ms: float | None = None
    paper_offline_comm_bytes: float | None = None
    paper_online_comm_bytes: float | None = None
    paper_total_comm_bytes: float | None = None
    split_ot_precomp_batch: int | None = None
    split_ot_preprocess_calls: int | None = None
    split_ot_preprocess_ots: int | None = None
    split_ot_fallback_calls: int | None = None
    split_ot_fallback_ots: int | None = None
    paper_accounting_complete: bool | None = None
    paper_accounting_model: str | None = None
    resource_status: str | None = None
    memory_limit_gb: int | None = None
    peak_rss_kb: int | None = None
    timeout_count: int = 0
    party1_exit: int | None = None
    party2_exit: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_float(pattern: str, log: str) -> float | None:
    m = re.search(pattern, log)
    return float(m.group(1)) if m else None


def _as_int(pattern: str, log: str) -> int | None:
    m = re.search(pattern, log)
    return int(m.group(1)) if m else None


def _token_pairs(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key] = value
    return out


def _parse_float_value(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int_value(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_bool01_value(value: str | None) -> bool | None:
    parsed = _parse_int_value(value)
    return None if parsed is None else parsed != 0


def _merged_phase_perf_fields(log: str) -> dict[str, Any]:
    rows = [
        _token_pairs(line)
        for line in log.splitlines()
        if line.startswith("[phase_perf]")
    ]
    if not rows:
        return {}

    def floats(name: str) -> list[float]:
        out: list[float] = []
        for row in rows:
            value = _parse_float_value(row.get(name))
            if value is not None:
                out.append(value)
        return out

    def ints(name: str) -> list[int]:
        out: list[int] = []
        for row in rows:
            value = _parse_int_value(row.get(name))
            if value is not None:
                out.append(value)
        return out

    merged: dict[str, Any] = {}
    for name in (
        "offline_time_ms",
        "online_time_ms",
        "total_time_ms",
        "paper_offline_time_ms",
        "paper_online_time_ms",
        "paper_total_time_ms",
    ):
        values = floats(name)
        if values:
            merged[name] = max(values)

    for name in (
        "offline_comm_bytes",
        "online_comm_bytes",
        "total_comm_bytes",
        "paper_offline_comm_bytes",
        "paper_online_comm_bytes",
        "paper_total_comm_bytes",
    ):
        values = floats(name)
        if values:
            merged[name] = sum(values)

    batch_values = ints("split_ot_precomp_batch")
    if batch_values:
        merged["split_ot_precomp_batch"] = max(batch_values)

    for name in (
        "split_ot_preprocess_calls",
        "split_ot_preprocess_ots",
        "split_ot_fallback_calls",
        "split_ot_fallback_ots",
    ):
        values = ints(name)
        if values:
            merged[name] = sum(values)

    complete_values = [
        _parse_bool01_value(row.get("paper_accounting_complete")) for row in rows
    ]
    complete_present = [value for value in complete_values if value is not None]
    if complete_present:
        merged["paper_accounting_complete"] = all(complete_present)

    for row in rows:
        model = row.get("paper_accounting_model")
        if model:
            merged["paper_accounting_model"] = model
            break

    return merged


def parse_cryptflow_logs(log: str) -> CryptflowParsedMetrics:
    perf_fields: dict[str, str] = {}
    for line in log.splitlines():
        if line.startswith("[perf]"):
            perf_fields.update(_token_pairs(line))

    phase_fields = _merged_phase_perf_fields(log)
    total_time_ms = phase_fields.get("total_time_ms")
    if total_time_ms is None:
        total_time_ms = _parse_float_value(perf_fields.get("total_time_ms"))
    legacy_time_ms = _parse_float_value(perf_fields.get("time_ms"))
    total_comm_bytes = phase_fields.get("total_comm_bytes")
    if total_comm_bytes is None:
        total_comm_bytes = _parse_float_value(perf_fields.get("total_comm_bytes"))
    legacy_comm_bytes = _parse_float_value(perf_fields.get("comm_bytes"))
    offline_time_ms = phase_fields.get("offline_time_ms")
    if offline_time_ms is None:
        offline_time_ms = _parse_float_value(perf_fields.get("offline_time_ms"))
    offline_comm_bytes = phase_fields.get("offline_comm_bytes")
    if offline_comm_bytes is None:
        offline_comm_bytes = _parse_float_value(perf_fields.get("offline_comm_bytes"))

    return CryptflowParsedMetrics(
        plain_gap=_as_float(r"\[mini\]\s+plain gap=([0-9eE.+-]+)", log),
        sci_gap=_as_float(r"\[mini\]\s+sci gap=([0-9eE.+-]+)", log),
        abs_err_gap=_as_float(r"\[mini\]\s+abs err gap=([0-9eE.+-]+)", log),
        time_ms=total_time_ms if total_time_ms is not None else legacy_time_ms,
        comm_bytes=total_comm_bytes
        if total_comm_bytes is not None
        else legacy_comm_bytes,
        offline_time_ms=offline_time_ms,
        online_time_ms=phase_fields.get("online_time_ms")
        if phase_fields.get("online_time_ms") is not None
        else _parse_float_value(perf_fields.get("online_time_ms")),
        total_time_ms=total_time_ms,
        offline_comm_bytes=offline_comm_bytes,
        online_comm_bytes=phase_fields.get("online_comm_bytes")
        if phase_fields.get("online_comm_bytes") is not None
        else _parse_float_value(perf_fields.get("online_comm_bytes")),
        total_comm_bytes=total_comm_bytes,
        raw_offline_time_ms=_parse_float_value(
            perf_fields.get("raw_offline_time_ms")
        )
        if perf_fields.get("raw_offline_time_ms") is not None
        else offline_time_ms,
        amortized_offline_time_ms=_parse_float_value(
            perf_fields.get("amortized_offline_time_ms")
        )
        if perf_fields.get("amortized_offline_time_ms") is not None
        else offline_time_ms,
        accounting_model=perf_fields.get("accounting_model"),
        perf_schema_version=_parse_int_value(perf_fields.get("perf_schema_version")),
        paper_offline_time_ms=phase_fields.get("paper_offline_time_ms")
        if phase_fields.get("paper_offline_time_ms") is not None
        else _parse_float_value(perf_fields.get("paper_offline_time_ms")),
        paper_online_time_ms=phase_fields.get("paper_online_time_ms")
        if phase_fields.get("paper_online_time_ms") is not None
        else _parse_float_value(perf_fields.get("paper_online_time_ms")),
        paper_total_time_ms=phase_fields.get("paper_total_time_ms")
        if phase_fields.get("paper_total_time_ms") is not None
        else _parse_float_value(perf_fields.get("paper_total_time_ms")),
        paper_offline_comm_bytes=phase_fields.get("paper_offline_comm_bytes")
        if phase_fields.get("paper_offline_comm_bytes") is not None
        else _parse_float_value(perf_fields.get("paper_offline_comm_bytes")),
        paper_online_comm_bytes=phase_fields.get("paper_online_comm_bytes")
        if phase_fields.get("paper_online_comm_bytes") is not None
        else _parse_float_value(perf_fields.get("paper_online_comm_bytes")),
        paper_total_comm_bytes=phase_fields.get("paper_total_comm_bytes")
        if phase_fields.get("paper_total_comm_bytes") is not None
        else _parse_float_value(perf_fields.get("paper_total_comm_bytes")),
        split_ot_precomp_batch=phase_fields.get("split_ot_precomp_batch")
        if phase_fields.get("split_ot_precomp_batch") is not None
        else _parse_int_value(perf_fields.get("split_ot_precomp_batch")),
        split_ot_preprocess_calls=phase_fields.get("split_ot_preprocess_calls")
        if phase_fields.get("split_ot_preprocess_calls") is not None
        else _parse_int_value(perf_fields.get("split_ot_preprocess_calls")),
        split_ot_preprocess_ots=phase_fields.get("split_ot_preprocess_ots")
        if phase_fields.get("split_ot_preprocess_ots") is not None
        else _parse_int_value(perf_fields.get("split_ot_preprocess_ots")),
        split_ot_fallback_calls=phase_fields.get("split_ot_fallback_calls")
        if phase_fields.get("split_ot_fallback_calls") is not None
        else _parse_int_value(perf_fields.get("split_ot_fallback_calls")),
        split_ot_fallback_ots=phase_fields.get("split_ot_fallback_ots")
        if phase_fields.get("split_ot_fallback_ots") is not None
        else _parse_int_value(perf_fields.get("split_ot_fallback_ots")),
        paper_accounting_complete=phase_fields.get("paper_accounting_complete")
        if phase_fields.get("paper_accounting_complete") is not None
        else _parse_bool01_value(perf_fields.get("paper_accounting_complete")),
        paper_accounting_model=phase_fields.get("paper_accounting_model")
        or perf_fields.get("paper_accounting_model"),
    )


def parse_ckks_logs(log: str) -> CkksParsedMetrics:
    ckks_fields: dict[str, str] = {}
    bootstrap_setup_ms = None
    bootstrap_keygen_ms = None
    bootstrap_coeff_abs_ms: list[float] = []

    for line in log.splitlines():
        if line.startswith("[ckks]"):
            ckks_fields.update(_token_pairs(line))
        elif line.startswith("[bootstrap]"):
            fields = _token_pairs(line)
            if "setup_ms" in fields:
                bootstrap_setup_ms = _parse_float_value(fields.get("setup_ms"))
            if "keygen_ms" in fields:
                bootstrap_keygen_ms = _parse_float_value(fields.get("keygen_ms"))
            if "coeff_abs_ms" in fields:
                value = _parse_float_value(fields.get("coeff_abs_ms"))
                if value is not None:
                    bootstrap_coeff_abs_ms.append(value)

    plain_lb = _as_float(r"\[mini\]\s+plain lb=([0-9eE.+-]+)", log)
    plain_ub = _as_float(r"\[mini\]\s+plain lb=[0-9eE.+-]+\s+ub=([0-9eE.+-]+)", log)
    ckks_lb = _as_float(r"\[mini\]\s+ckks lb=([0-9eE.+-]+)", log)
    ckks_ub = _as_float(r"\[mini\]\s+ckks lb=[0-9eE.+-]+\s+ub=([0-9eE.+-]+)", log)
    bound_fields_available = all(
        value is not None for value in (plain_lb, plain_ub, ckks_lb, ckks_ub)
    )
    bound_fields_unavailable_reason = (
        None
        if bound_fields_available
        else "ckks logs did not emit lower/upper bound fields"
    )
    coeff_abs_total = sum(bootstrap_coeff_abs_ms) if bootstrap_coeff_abs_ms else None

    return CkksParsedMetrics(
        plain_gap=_as_float(r"\[mini\]\s+plain gap=([0-9eE.+-]+)", log),
        sci_gap=_as_float(r"\[mini\]\s+ckks gap=([0-9eE.+-]+)", log),
        abs_err_gap=_as_float(r"\[mini\]\s+abs err gap=([0-9eE.+-]+)", log),
        plain_lb=plain_lb,
        plain_ub=plain_ub,
        ckks_lb=ckks_lb,
        ckks_ub=ckks_ub,
        abs_err_lb=_as_float(r"\[mini\]\s+abs err lb=([0-9eE.+-]+)", log),
        abs_err_ub=_as_float(r"\[mini\]\s+abs err lb=[0-9eE.+-]+\s+ub=([0-9eE.+-]+)", log),
        rel_err_lb=_as_float(r"\[mini\]\s+rel err lb=([0-9eE.+-]+)", log),
        rel_err_ub=_as_float(r"\[mini\]\s+rel err lb=[0-9eE.+-]+\s+ub=([0-9eE.+-]+)", log),
        bound_fields_available=bound_fields_available,
        bound_fields_unavailable_reason=bound_fields_unavailable_reason,
        time_ms=_as_float(r"\[perf\]\s+time_ms=([0-9eE.+-]+)", log),
        comm_bytes=_as_float(r"\[perf\].*comm_bytes=([0-9eE.+-]+)", log),
        ring_dim=_parse_int_value(ckks_fields.get("ring_dim")),
        depth=_parse_int_value(ckks_fields.get("depth")),
        effective_depth=_parse_int_value(ckks_fields.get("effective_depth")),
        scale_bits=_parse_int_value(ckks_fields.get("scale_bits")),
        sign_deg=_parse_int_value(ckks_fields.get("sign_deg")),
        recip_deg=_parse_int_value(ckks_fields.get("recip_deg")),
        act_range=_parse_float_value(ckks_fields.get("act_range")),
        coeff_range=_parse_float_value(ckks_fields.get("coeff_range")),
        denom_lo=_parse_float_value(ckks_fields.get("denom_lo")),
        denom_hi=_parse_float_value(ckks_fields.get("denom_hi")),
        batch_size=_parse_int_value(ckks_fields.get("batch_size")),
        security_level=ckks_fields.get("security_level"),
        secret_key_dist=ckks_fields.get("secret_key_dist"),
        sample=ckks_fields.get("sample"),
        bootstrap_coeff_abs=_parse_bool01_value(ckks_fields.get("bootstrap_coeff_abs")),
        bootstrap_slots=_parse_int_value(ckks_fields.get("bootstrap_slots")),
        bootstrap_level_enc=_parse_int_value(ckks_fields.get("bootstrap_level_enc")),
        bootstrap_level_dec=_parse_int_value(ckks_fields.get("bootstrap_level_dec")),
        bootstrap_dim1_enc=_parse_int_value(ckks_fields.get("bootstrap_dim1_enc")),
        bootstrap_dim1_dec=_parse_int_value(ckks_fields.get("bootstrap_dim1_dec")),
        bootstrap_correction=_parse_int_value(ckks_fields.get("bootstrap_correction")),
        bootstrap_iterations=_parse_int_value(ckks_fields.get("bootstrap_iterations")),
        bootstrap_setup_ms=bootstrap_setup_ms,
        bootstrap_keygen_ms=bootstrap_keygen_ms,
        bootstrap_coeff_abs_count=len(bootstrap_coeff_abs_ms),
        bootstrap_coeff_abs_total_ms=coeff_abs_total,
    )


def parse_nexus_logs(log: str) -> NexusParsedMetrics:
    nexus_fields: dict[str, str] = {}
    bootstrap_coeff_abs_ms: list[float] = []

    for line in log.splitlines():
        if line.startswith("[nexus]"):
            nexus_fields.update(_token_pairs(line))
        elif line.startswith("[bootstrap]"):
            fields = _token_pairs(line)
            if "coeff_abs_ms" in fields:
                value = _parse_float_value(fields.get("coeff_abs_ms"))
                if value is not None:
                    bootstrap_coeff_abs_ms.append(value)

    plain_lb = _as_float(r"\[mini\]\s+plain lb=([0-9eE.+-]+)", log)
    plain_ub = _as_float(r"\[mini\]\s+plain lb=[0-9eE.+-]+\s+ub=([0-9eE.+-]+)", log)
    nexus_lb = _as_float(r"\[mini\]\s+nexus lb=([0-9eE.+-]+)", log)
    nexus_ub = _as_float(r"\[mini\]\s+nexus lb=[0-9eE.+-]+\s+ub=([0-9eE.+-]+)", log)
    bound_fields_available = all(
        value is not None for value in (plain_lb, plain_ub, nexus_lb, nexus_ub)
    )
    bound_fields_unavailable_reason = (
        None
        if bound_fields_available
        else "nexus logs did not emit lower/upper bound fields"
    )
    coeff_abs_total = sum(bootstrap_coeff_abs_ms) if bootstrap_coeff_abs_ms else None

    return NexusParsedMetrics(
        status=nexus_fields.get("status"),
        status_detail=nexus_fields.get("status_detail"),
        plain_gap=_as_float(r"\[mini\]\s+plain gap=([0-9eE.+-]+)", log),
        nexus_gap=_as_float(r"\[mini\]\s+nexus gap=([0-9eE.+-]+)", log),
        nexus_abs_err_gap=_as_float(r"\[mini\]\s+abs err gap=([0-9eE.+-]+)", log),
        nexus_rel_err_gap=_as_float(r"\[mini\]\s+rel err gap=([0-9eE.+-]+)", log),
        plain_lb=plain_lb,
        plain_ub=plain_ub,
        nexus_lb=nexus_lb,
        nexus_ub=nexus_ub,
        nexus_abs_err_lb=_as_float(r"\[mini\]\s+abs err lb=([0-9eE.+-]+)", log),
        nexus_abs_err_ub=_as_float(r"\[mini\]\s+abs err lb=[0-9eE.+-]+\s+ub=([0-9eE.+-]+)", log),
        nexus_rel_err_lb=_as_float(r"\[mini\]\s+rel err lb=([0-9eE.+-]+)", log),
        nexus_rel_err_ub=_as_float(r"\[mini\]\s+rel err lb=[0-9eE.+-]+\s+ub=([0-9eE.+-]+)", log),
        bound_fields_available=bound_fields_available,
        bound_fields_unavailable_reason=bound_fields_unavailable_reason,
        time_ms=_as_float(r"\[perf\]\s+time_ms=([0-9eE.+-]+)", log),
        artifact_bytes=_as_float(r"\[perf\].*artifact_bytes=([0-9eE.+-]+)", log),
        comm_bytes=_as_float(r"\[perf\].*comm_bytes=([0-9eE.+-]+)", log),
        bootstrap_coeff_abs_count=len(bootstrap_coeff_abs_ms),
        bootstrap_coeff_abs_total_ms=coeff_abs_total,
        provider_decryptor_used=_parse_bool01_value(
            nexus_fields.get("provider_decryptor_used")
        ),
        privacy_notes=nexus_fields.get("privacy_notes"),
        disallowed_helpers_found=nexus_fields.get("disallowed_helpers_found"),
    )


def _ckks_result_fields(parsed: CkksParsedMetrics) -> dict[str, object]:
    return {
        "plain_lb": parsed.plain_lb,
        "plain_ub": parsed.plain_ub,
        "ckks_lb": parsed.ckks_lb,
        "ckks_ub": parsed.ckks_ub,
        "abs_err_lb": parsed.abs_err_lb,
        "abs_err_ub": parsed.abs_err_ub,
        "rel_err_lb": parsed.rel_err_lb,
        "rel_err_ub": parsed.rel_err_ub,
        "bound_fields_available": parsed.bound_fields_available,
        "bound_fields_unavailable_reason": parsed.bound_fields_unavailable_reason,
        "ckks_ring_dim": parsed.ring_dim,
        "ckks_depth": parsed.depth,
        "ckks_effective_depth": parsed.effective_depth,
        "ckks_scale_bits": parsed.scale_bits,
        "ckks_sign_deg": parsed.sign_deg,
        "ckks_recip_deg": parsed.recip_deg,
        "ckks_act_range": parsed.act_range,
        "ckks_coeff_range": parsed.coeff_range,
        "ckks_denom_lo": parsed.denom_lo,
        "ckks_denom_hi": parsed.denom_hi,
        "ckks_batch_size": parsed.batch_size,
        "ckks_security_level": parsed.security_level,
        "ckks_secret_key_dist": parsed.secret_key_dist,
        "ckks_sample": parsed.sample,
        "ckks_bootstrap_coeff_abs": parsed.bootstrap_coeff_abs,
        "ckks_bootstrap_slots": parsed.bootstrap_slots,
        "ckks_bootstrap_level_enc": parsed.bootstrap_level_enc,
        "ckks_bootstrap_level_dec": parsed.bootstrap_level_dec,
        "ckks_bootstrap_dim1_enc": parsed.bootstrap_dim1_enc,
        "ckks_bootstrap_dim1_dec": parsed.bootstrap_dim1_dec,
        "ckks_bootstrap_correction": parsed.bootstrap_correction,
        "ckks_bootstrap_iterations": parsed.bootstrap_iterations,
        "ckks_bootstrap_setup_ms": parsed.bootstrap_setup_ms,
        "ckks_bootstrap_keygen_ms": parsed.bootstrap_keygen_ms,
        "ckks_bootstrap_coeff_abs_count": parsed.bootstrap_coeff_abs_count,
        "ckks_bootstrap_coeff_abs_total_ms": parsed.bootstrap_coeff_abs_total_ms,
    }


def _nexus_result_fields(parsed: NexusParsedMetrics) -> dict[str, object]:
    return {
        "plain_lb": parsed.plain_lb,
        "plain_ub": parsed.plain_ub,
        "nexus_lb": parsed.nexus_lb,
        "nexus_ub": parsed.nexus_ub,
        "nexus_gap": parsed.nexus_gap,
        "nexus_abs_err_lb": parsed.nexus_abs_err_lb,
        "nexus_abs_err_ub": parsed.nexus_abs_err_ub,
        "nexus_abs_err_gap": parsed.nexus_abs_err_gap,
        "nexus_rel_err_lb": parsed.nexus_rel_err_lb,
        "nexus_rel_err_ub": parsed.nexus_rel_err_ub,
        "nexus_rel_err_gap": parsed.nexus_rel_err_gap,
        "bound_fields_available": parsed.bound_fields_available,
        "bound_fields_unavailable_reason": parsed.bound_fields_unavailable_reason,
        "nexus_status_detail": parsed.status_detail,
        "nexus_artifact_bytes": parsed.artifact_bytes,
        "nexus_comm_bytes": parsed.comm_bytes,
        "nexus_bootstrap_coeff_abs_count": parsed.bootstrap_coeff_abs_count,
        "nexus_bootstrap_coeff_abs_total_ms": parsed.bootstrap_coeff_abs_total_ms,
        "nexus_provider_decryptor_used": parsed.provider_decryptor_used,
        "nexus_privacy_notes": parsed.privacy_notes,
        "nexus_disallowed_helpers_found": parsed.disallowed_helpers_found,
    }


def _nexus_quality_failure_reason(
    parsed: NexusParsedMetrics,
    *,
    max_abs_err: float,
    max_rel_err: float,
) -> str | None:
    abs_values = [
        ("lb", parsed.nexus_abs_err_lb),
        ("ub", parsed.nexus_abs_err_ub),
        ("gap", parsed.nexus_abs_err_gap),
    ]
    rel_values = [
        ("lb", parsed.nexus_rel_err_lb),
        ("ub", parsed.nexus_rel_err_ub),
        ("gap", parsed.nexus_rel_err_gap),
    ]
    missing: list[str] = []
    if parsed.plain_gap is None or parsed.nexus_gap is None:
        missing.append("gap")
    if parsed.nexus_abs_err_gap is None or parsed.nexus_rel_err_gap is None:
        missing.append("gap_error")
    if not parsed.bound_fields_available:
        missing.append("bounds")
    if parsed.bound_fields_available:
        for name, value in abs_values[:2]:
            if value is None:
                missing.append(f"{name}_abs_err")
        for name, value in rel_values[:2]:
            if value is None:
                missing.append(f"{name}_rel_err")
    if missing:
        return "missing required NEXUS accuracy fields: " + ",".join(missing)
    for name, value in abs_values:
        if value is not None and value > max_abs_err:
            return f"nexus {name} abs error {value} exceeds {max_abs_err}"
    for name, value in rel_values:
        if value is not None and value > max_rel_err:
            return f"nexus {name} rel error {value} exceeds {max_rel_err}"
    return None


def _l1_norm(v: np.ndarray) -> float:
    return float(np.sum(np.abs(v)))


def _relu_state_from_bounds(ub: np.ndarray, lb: np.ndarray) -> np.ndarray:
    state = np.zeros(ub.shape[0], dtype=np.int64)
    state[np.maximum(0.0, ub) == 0.0] = -1
    state[np.maximum(0.0, lb) > 0.0] = 1
    return state


def default_target_label(pred: int, num_classes: int) -> int:
    if num_classes < 2:
        raise ValueError("targeted CROWN requires at least 2 output classes")
    if pred < 0 or pred >= num_classes:
        raise ValueError("pred is out of range for target selection")
    return 1 if pred == 0 else 0


def _first_layer_bounds(layer: Layer, x0: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    w, b = layer
    ax0 = w @ x0
    d = eps * np.sum(np.abs(w), axis=1)
    return ax0 + d + b, ax0 - d + b


def _make_margin_layer_targeted(last: Layer, pred: int, target: int) -> Layer:
    w, b = last
    if pred < 0 or target < 0 or pred >= w.shape[0] or target >= w.shape[0] or pred == target:
        raise ValueError("invalid targeted margin labels")
    return (w[pred : pred + 1] - w[target : target + 1], b[pred : pred + 1] - b[target : target + 1])


def _make_margin_layer_untargeted(last: Layer, pred: int) -> Layer:
    w, b = last
    if pred < 0 or pred >= w.shape[0]:
        raise ValueError("invalid untargeted margin label")
    rows = [j for j in range(w.shape[0]) if j != pred]
    return w[pred : pred + 1] - w[rows], b[pred : pred + 1] - b[rows]


def _standard_relax_bound_kernel(
    layers: Sequence[Layer],
    ubs: Sequence[np.ndarray],
    lbs: Sequence[np.ndarray],
    states: Sequence[np.ndarray],
    x0: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(layers) < 2:
        raise ValueError("standard_relax_bound_kernel requires at least 2 layers")

    nlayer = len(layers)
    last_w, last_b = layers[-1]
    no = last_w.shape[0]

    diags: list[np.ndarray] = [
        np.ones(layers[0][0].shape[1], dtype=np.float64)
    ]
    for i in range(1, nlayer):
        st = states[i - 1]
        alpha = np.maximum(st.astype(np.float64), 0.0)
        idx_unsure = np.nonzero(st == 0)[0]
        if idx_unsure.size:
            upper = ubs[i][idx_unsure]
            lower = lbs[i][idx_unsure]
            alpha[idx_unsure] = upper / (upper - lower)
        diags.append(alpha)

    constants = last_b.astype(np.float64, copy=True)
    uf = np.zeros(no, dtype=np.float64)
    lf = np.zeros(no, dtype=np.float64)

    a = last_w.astype(np.float64, copy=True) * diags[-1]
    for i in range(nlayer - 1, 0, -1):
        constants += a @ layers[i - 1][1]
        idx_unsure = np.nonzero(states[i - 1] == 0)[0]
        lbi = lbs[i]

        for j in range(no):
            lu = np.zeros_like(lbi)
            ll = np.zeros_like(lbi)
            row_unsure = a[j, idx_unsure]
            idx_pos = idx_unsure[np.nonzero(row_unsure > 0.0)[0]]
            idx_neg = idx_unsure[np.nonzero(row_unsure < 0.0)[0]]
            lu[idx_pos] = lbi[idx_pos]
            ll[idx_neg] = lbi[idx_neg]
            uf[j] -= float(np.dot(a[j], lu))
            lf[j] -= float(np.dot(a[j], ll))

        prev_w = layers[i - 1][0]
        if i != 1:
            a = a @ (prev_w * diags[i - 1])
        else:
            a = a @ prev_w

    for j in range(no):
        uf[j] += constants[j] + float(np.dot(a[j], x0)) + eps * _l1_norm(a[j])
        lf[j] += constants[j] + float(np.dot(a[j], x0)) - eps * _l1_norm(a[j])
    return uf, lf


def plain_crown_gap(
    layers: Sequence[Layer],
    x: np.ndarray,
    eps: float,
    pred: int,
    *,
    target: int | None = None,
    untargeted: bool = False,
) -> float:
    if len(layers) < 2:
        raise ValueError("plain_crown_gap requires at least 2 layers")
    x0 = np.asarray(x, dtype=np.float64).reshape(-1)
    ub0 = x0 + eps
    lb0 = x0 - eps
    pre_ub: list[np.ndarray] = []
    pre_lb: list[np.ndarray] = []
    states: list[np.ndarray] = []

    for num in range(len(layers) - 1):
        if num == 0:
            u, l = _first_layer_bounds(layers[0], x0, eps)
        else:
            prefix = layers[: num + 1]
            us2 = [ub0, *pre_ub]
            ls2 = [lb0, *pre_lb]
            u, l = _standard_relax_bound_kernel(prefix, us2, ls2, states, x0, eps)
        pre_ub.append(u)
        pre_lb.append(l)
        states.append(_relu_state_from_bounds(u, l))

    if untargeted:
        margin_layer = _make_margin_layer_untargeted(layers[-1], pred)
    else:
        if target is None:
            target = default_target_label(pred, layers[-1][0].shape[0])
        margin_layer = _make_margin_layer_targeted(layers[-1], pred, target)

    final_layers = [*layers[:-1], margin_layer]
    uf, lf = _standard_relax_bound_kernel(
        final_layers, [ub0, *pre_ub], [lb0, *pre_lb], states, x0, eps
    )
    del uf
    return float(np.min(lf) if untargeted else (lf[0] if lf.size else 0.0))


def run_plain_crown_sample(
    *,
    layers: Sequence[Layer],
    x: np.ndarray,
    eps: float,
    index: int,
    label: int,
    pred: int,
    target: int | None = None,
    untargeted: bool = False,
) -> CrownSampleResult:
    try:
        gap = plain_crown_gap(
            layers, x, eps, pred, target=target, untargeted=untargeted
        )
        return CrownSampleResult(
            index=index,
            label=label,
            pred=pred,
            eps=eps,
            method="plain-crown",
            status="ok",
            certified=gap > 0.0,
            plain_gap=gap,
        )
    except Exception as exc:  # pragma: no cover - defensive record path
        return CrownSampleResult(
            index=index,
            label=label,
            pred=pred,
            eps=eps,
            method="plain-crown",
            status="plain_error",
            error=str(exc),
        )


def write_x0bin(path: Path, x: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(x, dtype="<f8").tobytes())


def ensure_cryptflow_build(repo: Path, backend: str, build_dir: Path) -> None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        subprocess.run(
            [
                "cmake",
                "-S",
                str(repo),
                "-B",
                str(build_dir),
                f"-DCROWN_BACKEND={backend.upper()}",
            ],
            check=True,
        )
    subprocess.run(["cmake", "--build", str(build_dir), "-j"], check=True)


def ensure_ckks_build(repo: Path, build_dir: Path) -> None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        subprocess.run(
            ["cmake", "-S", str(repo), "-B", str(build_dir)],
            check=True,
        )
    subprocess.run(["cmake", "--build", str(build_dir), "-j"], check=True)


def ensure_nexus_build(repo: Path, build_dir: Path) -> None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        subprocess.run(["cmake", "-S", str(repo), "-B", str(build_dir)], check=True)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "nexus_crown_runner", "-j2"],
        check=True,
    )


def _cryptflow_party_args(
    *,
    ip: str,
    port: int,
    threads: int,
    bitlength: int,
    scale_bits: int,
    eps: float,
    mlp_path: Path,
    x0bin_path: Path,
    target: int | None = None,
    include_eps: bool = True,
) -> list[str]:
    args = [
        f"ip={ip}",
        f"p={port}",
        f"nt={threads}",
        f"bl={bitlength}",
        f"sf={scale_bits}",
        f"mlp={mlp_path}",
        f"x0bin={x0bin_path}",
    ]
    if target is not None:
        args.append(f"target={target}")
    if include_eps:
        args.append(f"eps={eps:.17g}")
    return args


def _memory_limit_preexec(memory_limit_gb: int | None):
    if memory_limit_gb is None:
        return None
    if memory_limit_gb <= 0:
        raise ValueError("memory_limit_gb must be positive")
    if resource is None:
        raise RuntimeError("memory_limit_gb requires Unix resource limits")

    limit_bytes = int(memory_limit_gb) * 1024 * 1024 * 1024

    def _apply_limit() -> None:
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        capped_hard = limit_bytes if hard == resource.RLIM_INFINITY else min(hard, limit_bytes)
        resource.setrlimit(resource.RLIMIT_AS, (capped_hard, capped_hard))

    return _apply_limit


def _terminate_process(proc: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
        pass


def _wait_two_party_processes(
    p1: subprocess.Popen[bytes] | subprocess.Popen[str],
    p2: subprocess.Popen[bytes] | subprocess.Popen[str],
    timeout_sec: int,
) -> tuple[int, int]:
    deadline = time.monotonic() + timeout_sec
    rc1: int | None = None
    rc2: int | None = None
    while True:
        rc1 = p1.poll()
        rc2 = p2.poll()
        if rc1 is not None and rc2 is not None:
            return int(rc1), int(rc2)
        if (rc1 is not None and rc1 != 0) or (rc2 is not None and rc2 != 0):
            _terminate_process(p1)
            _terminate_process(p2)
            return int(p1.returncode if p1.returncode is not None else rc1 or 124), int(
                p2.returncode if p2.returncode is not None else rc2 or 124
            )
        if time.monotonic() >= deadline:
            _terminate_process(p1)
            _terminate_process(p2)
            return int(p1.returncode if p1.returncode is not None else 124), int(
                p2.returncode if p2.returncode is not None else 124
            )
        time.sleep(0.05)


def _run_two_party(
    *,
    bin_path: Path,
    ip: str,
    port: int,
    threads: int,
    bitlength: int,
    scale_bits: int,
    eps: float,
    mlp_path: Path,
    x0bin_path: Path,
    target: int | None = None,
    timeout_sec: int,
    memory_limit_gb: int | None = None,
    extra_env: dict[str, str] | None = None,
    log_dir: Path | None = None,
) -> TwoPartyRunResult:
    p1_args = _cryptflow_party_args(
        ip=ip,
        port=port,
        threads=threads,
        bitlength=bitlength,
        scale_bits=scale_bits,
        eps=eps,
        mlp_path=mlp_path,
        x0bin_path=x0bin_path,
        target=target,
        include_eps=False,
    )
    p2_args = _cryptflow_party_args(
        ip=ip,
        port=port,
        threads=threads,
        bitlength=bitlength,
        scale_bits=scale_bits,
        eps=eps,
        mlp_path=mlp_path,
        x0bin_path=x0bin_path,
        target=target,
        include_eps=True,
    )
    p1_log = tempfile.NamedTemporaryFile(
        prefix="crown_p1_", suffix=".log", dir=log_dir, delete=False
    )
    p2_log = tempfile.NamedTemporaryFile(
        prefix="crown_p2_", suffix=".log", dir=log_dir, delete=False
    )
    p1_log.close()
    p2_log.close()

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    preexec_fn = _memory_limit_preexec(memory_limit_gb)

    with open(p1_log.name, "w") as f1, open(p2_log.name, "w") as f2:
        p1 = subprocess.Popen(
            [str(bin_path), "r=1", *p1_args],
            stdout=f1,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=preexec_fn,
        )
        time.sleep(0.3)
        p2 = subprocess.Popen(
            [str(bin_path), "r=2", *p2_args],
            stdout=f2,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=preexec_fn,
        )
        rc1, rc2 = _wait_two_party_processes(p1, p2, timeout_sec)

    out1 = Path(p1_log.name).read_text(errors="replace")
    out2 = Path(p2_log.name).read_text(errors="replace")
    if log_dir is None:
        Path(p1_log.name).unlink(missing_ok=True)
        Path(p2_log.name).unlink(missing_ok=True)
    return TwoPartyRunResult(
        party1_exit=rc1,
        party2_exit=rc2,
        party1_stdout=out1,
        party2_stdout=out2,
        party1_stdout_path=str(p1_log.name) if log_dir is not None else None,
        party2_stdout_path=str(p2_log.name) if log_dir is not None else None,
        peak_rss_kb=_parse_peak_rss_kb(out1 + "\n" + out2),
    )


def _parse_peak_rss_kb(log: str) -> int | None:
    return _as_int(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", log)


def _looks_like_memory_limit(log: str, rc1: int, rc2: int) -> bool:
    lower = log.lower()
    if "address already in use" in lower or "error: bind" in lower:
        return False
    memory_markers = (
        "memory limit",
        "out of memory",
        "cannot allocate memory",
        "maximum resident set size",
        "oom",
    )
    return rc1 in (137, -9) or rc2 in (137, -9) or any(
        marker in lower for marker in memory_markers
    )


def _cryptflow_resource_status(rc1: int, rc2: int, log: str) -> str:
    if rc1 == 0 and rc2 == 0:
        return "ok"
    if rc1 == 124 and rc2 == 124:
        return "timeout"
    if _looks_like_memory_limit(log, rc1, rc2):
        return "memory_limit"
    return "secure_error"


def prepare_crown_mlp(onnx_path: Path, out_path: Path) -> int:
    export_crown_mlp_bin(onnx_path, out_path)
    input_dim, _layers = read_crown_mlp_bin(out_path)
    return input_dim


def run_cryptflow_crown_sample(
    *,
    onnx_path: Path,
    x: np.ndarray,
    eps: float,
    index: int,
    label: int,
    pred: int,
    target: int | None = None,
    backend: str,
    repo: Path,
    bitlength: int = 41,
    scale_bits: int = 18,
    threads: int = 1,
    ip: str = "127.0.0.1",
    port: int = 42000,
    timeout: int = 300,
    memory_limit_gb: int | None = None,
    skip_build: bool = False,
    mlp_path: Path | None = None,
    keep_artifacts: bool = False,
    extra_env: dict[str, str] | None = None,
) -> CrownSampleResult:
    tmp_obj: tempfile.TemporaryDirectory[str] | None = None
    if keep_artifacts:
        tmp_dir = Path(tempfile.mkdtemp(prefix="crown_run_"))
    else:
        tmp_obj = tempfile.TemporaryDirectory(prefix="crown_run_")
        tmp_dir = Path(tmp_obj.name)
    try:
        repo = repo.resolve()
        build_dir = repo / f"build-{backend}"
        bin_path = build_dir / "src" / "cryptflow2_crown"
        if not skip_build:
            ensure_cryptflow_build(repo, backend, build_dir)
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary not found: {bin_path}")

        actual_mlp = mlp_path or (tmp_dir / f"{onnx_path.stem}.crownmlp")
        if mlp_path is None:
            input_dim = prepare_crown_mlp(onnx_path, actual_mlp)
            _, layers = read_crown_mlp_bin(actual_mlp)
        else:
            input_dim, layers = read_crown_mlp_bin(actual_mlp)
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_arr.size != input_dim:
            raise ValueError(f"x dimension {x_arr.size} != model input dimension {input_dim}")
        x0bin_path = tmp_dir / "x0.bin"
        write_x0bin(x0bin_path, x_arr)
        output_dim = layers[-1][0].shape[0] if layers else max(pred + 1, 2)
        actual_target = target if target is not None else default_target_label(pred, output_dim)

        run = _run_two_party(
            bin_path=bin_path,
            ip=ip,
            port=port,
            threads=threads,
            bitlength=bitlength,
            scale_bits=scale_bits,
            eps=eps,
            mlp_path=actual_mlp,
            x0bin_path=x0bin_path,
            target=actual_target,
            timeout_sec=timeout,
            memory_limit_gb=memory_limit_gb,
            extra_env=extra_env,
            log_dir=tmp_dir if keep_artifacts else None,
        )
        rc1 = run.party1_exit
        rc2 = run.party2_exit
        out1 = run.party1_stdout
        out2 = run.party2_stdout
        combined_log = out1 + "\n" + out2
        parsed = parse_cryptflow_logs(combined_log)
        status = _cryptflow_resource_status(rc1, rc2, combined_log)
        error = None
        if status != "ok":
            error = "\n".join(
                ln
                for ln in combined_log.splitlines()
                if (
                    "failed" in ln.lower()
                    or "error" in ln.lower()
                    or "memory" in ln.lower()
                    or "killed" in ln.lower()
                    or "maximum resident set size" in ln.lower()
                )
            ) or f"cryptflow exits party1={rc1} party2={rc2}"
        return CrownSampleResult(
            index=index,
            label=label,
            pred=pred,
            eps=eps,
            method="cryptflow-crown",
            status=status,
            certified=(parsed.sci_gap > 0.0) if parsed.sci_gap is not None else None,
            plain_gap=parsed.plain_gap,
            sci_gap=parsed.sci_gap,
            abs_err_gap=parsed.abs_err_gap,
            time_ms=parsed.time_ms,
            comm_bytes=parsed.comm_bytes,
            stdout_path=run.party1_stdout_path,
            stderr_path=run.party2_stdout_path,
            offline_time_ms=parsed.offline_time_ms,
            online_time_ms=parsed.online_time_ms,
            total_time_ms=parsed.total_time_ms,
            offline_comm_bytes=parsed.offline_comm_bytes,
            online_comm_bytes=parsed.online_comm_bytes,
            total_comm_bytes=parsed.total_comm_bytes,
            raw_offline_time_ms=parsed.raw_offline_time_ms,
            amortized_offline_time_ms=parsed.amortized_offline_time_ms,
            accounting_model=parsed.accounting_model,
            perf_schema_version=parsed.perf_schema_version,
            paper_offline_time_ms=parsed.paper_offline_time_ms,
            paper_online_time_ms=parsed.paper_online_time_ms,
            paper_total_time_ms=parsed.paper_total_time_ms,
            paper_offline_comm_bytes=parsed.paper_offline_comm_bytes,
            paper_online_comm_bytes=parsed.paper_online_comm_bytes,
            paper_total_comm_bytes=parsed.paper_total_comm_bytes,
            split_ot_precomp_batch=parsed.split_ot_precomp_batch,
            split_ot_preprocess_calls=parsed.split_ot_preprocess_calls,
            split_ot_preprocess_ots=parsed.split_ot_preprocess_ots,
            split_ot_fallback_calls=parsed.split_ot_fallback_calls,
            split_ot_fallback_ots=parsed.split_ot_fallback_ots,
            paper_accounting_complete=parsed.paper_accounting_complete,
            paper_accounting_model=parsed.paper_accounting_model,
            resource_status=status,
            memory_limit_gb=memory_limit_gb,
            peak_rss_kb=run.peak_rss_kb or _parse_peak_rss_kb(combined_log),
            timeout_count=1 if status == "timeout" else 0,
            party1_exit=rc1,
            party2_exit=rc2,
            error=error,
        )
    except Exception as exc:
        return CrownSampleResult(
            index=index,
            label=label,
            pred=pred,
            eps=eps,
            method="cryptflow-crown",
            status="secure_error",
            resource_status="secure_error",
            memory_limit_gb=memory_limit_gb,
            error=str(exc),
        )
    finally:
        if tmp_obj is not None:
            tmp_obj.cleanup()


def run_ckks_crown_sample(
    *,
    onnx_path: Path,
    x: np.ndarray,
    eps: float,
    index: int,
    label: int,
    pred: int,
    target: int | None = None,
    repo: Path,
    depth: int = 60,
    scale_bits: int = 28,
    sign_deg: int = 9,
    recip_deg: int = 9,
    timeout: int = 300,
    skip_build: bool = False,
    mlp_path: Path | None = None,
    keep_artifacts: bool = False,
) -> CrownSampleResult:
    tmp_obj: tempfile.TemporaryDirectory[str] | None = None
    if keep_artifacts:
        tmp_dir = Path(tempfile.mkdtemp(prefix="ckks_crown_run_"))
    else:
        tmp_obj = tempfile.TemporaryDirectory(prefix="ckks_crown_run_")
        tmp_dir = Path(tmp_obj.name)
    try:
        repo = repo.resolve()
        build_dir = repo / "build"
        bin_path = build_dir / "ckks_crown"
        if not skip_build:
            ensure_ckks_build(repo, build_dir)
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary not found: {bin_path}")

        actual_mlp = mlp_path or (tmp_dir / f"{onnx_path.stem}.crownmlp")
        input_dim = (
            prepare_crown_mlp(onnx_path, actual_mlp)
            if mlp_path is None
            else read_crown_mlp_bin(actual_mlp)[0]
        )
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_arr.size != input_dim:
            raise ValueError(f"x dimension {x_arr.size} != model input dimension {input_dim}")
        x0bin_path = tmp_dir / "x0.bin"
        write_x0bin(x0bin_path, x_arr)

        argv = [
            str(bin_path),
            f"mlp={actual_mlp}",
            f"x0bin={x0bin_path}",
            f"eps={eps:.17g}",
            f"depth={depth}",
            f"scale={scale_bits}",
            f"sign_deg={sign_deg}",
            f"recip_deg={recip_deg}",
        ]
        if target is not None:
            argv.append(f"target={target}")
        try:
            proc = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            combined_log = (proc.stdout or "") + ("\n" if proc.stderr else "") + (proc.stderr or "")
            parsed = parse_ckks_logs(combined_log)
            status = "ok" if proc.returncode == 0 and parsed.sci_gap is not None else "error"
            error = None
            if status != "ok":
                error = "\n".join(
                    ln
                    for ln in combined_log.splitlines()
                    if "failed" in ln.lower() or "error" in ln.lower()
                ) or (
                    "ckks_crown produced no parseable ckks gap"
                    if proc.returncode == 0
                    else f"ckks_crown exit={proc.returncode}"
                )
            return CrownSampleResult(
                index=index,
                label=label,
                pred=pred,
                eps=eps,
                method="ckks-crown",
                status=status,
                certified=(parsed.sci_gap > 0.0) if parsed.sci_gap is not None else None,
                plain_gap=parsed.plain_gap,
                sci_gap=parsed.sci_gap,
                abs_err_gap=parsed.abs_err_gap,
                time_ms=parsed.time_ms,
                comm_bytes=0 if parsed.comm_bytes is None else parsed.comm_bytes,
                **_ckks_result_fields(parsed),
                party1_exit=proc.returncode,
                error=error,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or ""
            err = exc.stderr or ""
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            if isinstance(err, bytes):
                err = err.decode(errors="replace")
            combined_log = out + ("\n" if err else "") + err
            parsed = parse_ckks_logs(combined_log)
            return CrownSampleResult(
                index=index,
                label=label,
                pred=pred,
                eps=eps,
                method="ckks-crown",
                status="error",
                certified=None,
                plain_gap=parsed.plain_gap,
                sci_gap=parsed.sci_gap,
                abs_err_gap=parsed.abs_err_gap,
                time_ms=parsed.time_ms,
                comm_bytes=0 if parsed.comm_bytes is None else parsed.comm_bytes,
                **_ckks_result_fields(parsed),
                party1_exit=124,
                error=f"ckks_crown timed out after {timeout}s",
            )
    except Exception as exc:
        return CrownSampleResult(
            index=index,
            label=label,
            pred=pred,
            eps=eps,
            method="ckks-crown",
            status="error",
            error=str(exc),
        )
    finally:
        if tmp_obj is not None:
            tmp_obj.cleanup()


def run_nexus_crown_sample(
    *,
    onnx_path: Path,
    x: np.ndarray,
    eps: float,
    index: int,
    label: int,
    pred: int,
    repo: Path,
    target: int | None = None,
    timeout: int = 300,
    skip_build: bool = False,
    build_dir: Path | None = None,
    mlp_path: Path | None = None,
    keep_artifacts: bool = False,
    nexus_max_abs_err: float = 0.05,
    nexus_max_rel_err: float = 0.50,
) -> CrownSampleResult:
    tmp_obj: tempfile.TemporaryDirectory[str] | None = None
    if keep_artifacts:
        tmp_dir = Path(tempfile.mkdtemp(prefix="nexus_crown_run_"))
    else:
        tmp_obj = tempfile.TemporaryDirectory(prefix="nexus_crown_run_")
        tmp_dir = Path(tmp_obj.name)
    stdout_path = tmp_dir / "nexus_stdout.log"
    stderr_path = tmp_dir / "nexus_stderr.log"
    try:
        repo = repo.resolve()
        actual_build_dir = build_dir or (repo / "build")
        bin_path = actual_build_dir / "bin" / "nexus_crown_runner"
        if not skip_build:
            ensure_nexus_build(repo, actual_build_dir)
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary not found: {bin_path}")

        actual_mlp = mlp_path or (tmp_dir / f"{onnx_path.stem}.crownmlp")
        input_dim = (
            prepare_crown_mlp(onnx_path, actual_mlp)
            if mlp_path is None
            else read_crown_mlp_bin(actual_mlp)[0]
        )
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_arr.size != input_dim:
            raise ValueError(f"x dimension {x_arr.size} != model input dimension {input_dim}")
        x0bin_path = tmp_dir / "x0.bin"
        write_x0bin(x0bin_path, x_arr)
        actual_target = target if target is not None else (1 if pred == 0 else 0)

        argv = [
            str(bin_path),
            f"mlp={actual_mlp}",
            f"x0bin={x0bin_path}",
            f"eps={eps:.17g}",
            f"pred={pred}",
            f"target={actual_target}",
        ]
        try:
            proc = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            if keep_artifacts:
                stdout_path.write_text(stdout)
                stderr_path.write_text(stderr)
            combined_log = stdout + ("\n" if stderr else "") + stderr
            parsed = parse_nexus_logs(combined_log)
            status = parsed.status
            if status is None:
                status = "native_error" if proc.returncode != 0 else "native_error"
            error = None
            if parsed.provider_decryptor_used is True:
                status = "provider_decryptor_violation"
                error = "NEXUS provider path used decryptor or disallowed helper"
            elif status == "ok":
                reason = _nexus_quality_failure_reason(
                    parsed,
                    max_abs_err=nexus_max_abs_err,
                    max_rel_err=nexus_max_rel_err,
                )
                if reason is not None:
                    status = "approximation_quality_failure"
                    error = reason
            if status != "ok" and error is None:
                error_lines = [
                    ln
                    for ln in combined_log.splitlines()
                    if "failed" in ln.lower() or "error" in ln.lower()
                ]
                error = "\n".join(error_lines) or parsed.status_detail
                if not error:
                    error = (
                        "nexus_crown_runner produced no typed status"
                        if parsed.status is None
                        else f"nexus_crown_runner status={status} exit={proc.returncode}"
                    )
            return CrownSampleResult(
                index=index,
                label=label,
                pred=pred,
                eps=eps,
                method="nexus-crown",
                status=status,
                certified=(parsed.nexus_gap > 0.0) if status == "ok" and parsed.nexus_gap is not None else None,
                plain_gap=parsed.plain_gap,
                sci_gap=None,
                abs_err_gap=parsed.nexus_abs_err_gap,
                time_ms=parsed.time_ms,
                comm_bytes=0 if parsed.comm_bytes is None else parsed.comm_bytes,
                **_nexus_result_fields(parsed),
                nexus_stdout_path=str(stdout_path) if keep_artifacts else None,
                nexus_stderr_path=str(stderr_path) if keep_artifacts else None,
                nexus_mlp_path=str(actual_mlp),
                nexus_x0bin_path=str(x0bin_path) if keep_artifacts else None,
                party1_exit=proc.returncode,
                error=error,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or ""
            err = exc.stderr or ""
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            if isinstance(err, bytes):
                err = err.decode(errors="replace")
            if keep_artifacts:
                stdout_path.write_text(out)
                stderr_path.write_text(err)
            combined_log = out + ("\n" if err else "") + err
            parsed = parse_nexus_logs(combined_log)
            return CrownSampleResult(
                index=index,
                label=label,
                pred=pred,
                eps=eps,
                method="nexus-crown",
                status="timeout",
                certified=None,
                plain_gap=parsed.plain_gap,
                abs_err_gap=parsed.nexus_abs_err_gap,
                time_ms=parsed.time_ms,
                comm_bytes=0 if parsed.comm_bytes is None else parsed.comm_bytes,
                **_nexus_result_fields(parsed),
                nexus_stdout_path=str(stdout_path) if keep_artifacts else None,
                nexus_stderr_path=str(stderr_path) if keep_artifacts else None,
                nexus_mlp_path=str(mlp_path) if mlp_path is not None else None,
                party1_exit=124,
                error=f"nexus_crown_runner timed out after {timeout}s",
            )
    except Exception as exc:
        return CrownSampleResult(
            index=index,
            label=label,
            pred=pred,
            eps=eps,
            method="nexus-crown",
            status="build_failure",
            error=str(exc),
        )
    finally:
        if tmp_obj is not None:
            tmp_obj.cleanup()
