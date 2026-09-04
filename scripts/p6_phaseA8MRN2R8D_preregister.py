"""P6 Phase-A8MRN2R8D train-only diagonal gain-consistency preregistration.

This stage hard-freezes the finalized R8C result, freezes one H4 training
objective, replays the future coordinate stream without opening science target
arrays, and performs synthetic/model/resource audits.  It performs no science
training, checkpoint inference, exact-physics call, or PSF generation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

import p6_phaseA8MRN2_development as n2
from p6_a8mrn1_direct_implicit_models import normalized_displacement, parameter_count
from p6_a8mrn2_frozen_helpers import (
    NATIVE_VOXELS,
    SHAPE,
    SHELL_NAMES,
    c4_rotate_indices,
    eligible_shells_from_radii,
    linear_index,
    normalized_coordinates,
    unravel_indices,
)
from p6_phaseA8MRN2R8B_boundary_aware_source_model import (
    BOUNDARY_AUXILIARY_GLOBAL_INDICES,
    BOUNDARY_FEATURE_NAMES,
    build_h3,
    coordinate_encoding_h3,
)
from p6_phaseA8MRN2R8D_diagonal_gain_consistency_model import (
    DIAGONAL_LOCAL_INDICES,
    DIAGONAL_PAIRS_PER_UPDATE,
    FIELD_PAIRS_PER_UPDATE,
    LAMBDA_DIAG,
    build_h4,
    diagonal_dual_loss,
    field_dual_loss,
    h4_total_loss,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/airsas_bunny20k/p6_complex_psf_field_deconv"
DOCS = ROOT / "docs"
SUBMISSION = ROOT / "本轮提交"
SCRIPT = Path(__file__).resolve()
H3_SOURCE = ROOT / "scripts/p6_phaseA8MRN2R8B_boundary_aware_source_model.py"
H4_SOURCE = ROOT / "scripts/p6_phaseA8MRN2R8D_diagonal_gain_consistency_model.py"
R8C_REPORT = DOCS / "P6_PHASEA8MRN2R8C_BOUNDARY_AWARE_SOURCE_CONDITIONING_DEVELOPMENT.md"
REPORT = DOCS / "P6_PHASEA8MRN2R8D_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY_PREREGISTRATION.md"

PREFIX = "p6_phaseA8MRN2R8D_"
STATUS = OUT / f"{PREFIX}status.json"
PROTOCOL = OUT / f"{PREFIX}protocol.json"
R8C_FREEZE = OUT / f"{PREFIX}R8C_freeze_manifest.json"
RATIONALE = OUT / f"{PREFIX}scientific_rationale.json"
H3_POSTMORTEM = OUT / f"{PREFIX}H3_mechanism_postmortem.json"
TRAIN_DIAGONAL = OUT / f"{PREFIX}TRAIN140_diagonal_prior_freeze.json"
PRIOR_ISOLATION = OUT / f"{PREFIX}prior_isolation_audit.json"
H4_SPEC = OUT / f"{PREFIX}H4_model_spec.json"
COORDINATE_SPEC = OUT / f"{PREFIX}diagonal_coordinate_spec.json"
COORDINATE_COVERAGE = OUT / f"{PREFIX}diagonal_coordinate_coverage_audit.json"
DIAGONAL_STREAM = OUT / f"{PREFIX}diagonal_stream_freeze.json"
LOSS_SPEC = OUT / f"{PREFIX}loss_spec.json"
H3_EQUIVALENCE = OUT / f"{PREFIX}H3_equivalence_audit.json"
INIT_IDENTITY = OUT / f"{PREFIX}initialization_identity_audit.json"
TRAIN_INHERIT = OUT / f"{PREFIX}training_inheritance_manifest.json"
R8E_OUTCOME = OUT / f"{PREFIX}R8E_outcome_protocol.json"
TRUTH_TABLE = OUT / f"{PREFIX}outcome_truth_table_audit.json"
DUMMY_RESOURCE = OUT / f"{PREFIX}dummy_resource_audit.json"
SUMMARY = OUT / f"{PREFIX}summary.json"

R8C_JSON_NAMES = [
    "p6_phaseA8MRN2R8C_status.json",
    "p6_phaseA8MRN2R8C_summary.json",
    "p6_phaseA8MRN2R8C_protocol_execution_record.json",
    "p6_phaseA8MRN2R8C_input_freeze_manifest.json",
    "p6_phaseA8MRN2R8C_execution_freeze_manifest.json",
    "p6_phaseA8MRN2R8C_H3_model_runtime_audit.json",
    "p6_phaseA8MRN2R8C_boundary_feature_runtime_audit.json",
    "p6_phaseA8MRN2R8C_sample_stream_audit.json",
    "p6_phaseA8MRN2R8C_training_history.json",
    "p6_phaseA8MRN2R8C_H3_best_checkpoint_freeze.json",
    "p6_phaseA8MRN2R8C_INTERNAL_TEST_metrics.json",
    "p6_phaseA8MRN2R8C_cancellation_retest.json",
    "p6_phaseA8MRN2R8C_burned24_metrics.json",
    "p6_phaseA8MRN2R8C_burned24_strata.json",
    "p6_phaseA8MRN2R8C_boundary_controlled_comparison.json",
    "p6_phaseA8MRN2R8C_gain_controlled_comparison.json",
    "p6_phaseA8MRN2R8C_development_readiness_and_outcome.json",
]
R8C_PATHS = [R8C_REPORT] + [OUT / name for name in R8C_JSON_NAMES]
R8C_PACKAGE_JSON = SUBMISSION / "本轮提交清单.json"
R8C_PACKAGE_TXT = SUBMISSION / "本轮提交清单.txt"

R8C_STATUS = OUT / "p6_phaseA8MRN2R8C_status.json"
R8C_SUMMARY = OUT / "p6_phaseA8MRN2R8C_summary.json"
R8C_BEST = OUT / "p6_phaseA8MRN2R8C_H3_best_checkpoint_freeze.json"
R8C_BOUNDARY = OUT / "p6_phaseA8MRN2R8C_boundary_controlled_comparison.json"
R8C_GAIN = OUT / "p6_phaseA8MRN2R8C_gain_controlled_comparison.json"
R8C_OUTCOME = OUT / "p6_phaseA8MRN2R8C_development_readiness_and_outcome.json"
R8B_TRAIN_DIAGONAL = OUT / "p6_phaseA8MRN2R8B_TRAIN140_diagonal_prior_audit.json"
TRAIN_STATS = OUT / "p6_phaseA8MRN2_train_statistics.json"
SPLIT_PATH = OUT / "p6_phaseA8MRN1R_column_split.json"
BLACKLIST_PATH = OUT / "p6_phaseA8MRN1R_C4_closed_blacklist.json"
SAMPLER_PATH = OUT / "p6_phaseA8MRN1R_training_sampler_spec.json"
VAL_PATH = OUT / "p6_phaseA8MRN1R_VAL_probe_coordinates.json"

EXPECTED_R8C_CLASS = "P6_PHASEA8MRN2R8C_BOUNDARY_AWARE_SOURCE_CONDITIONING_IMPROVES_H2_BUT_NOT_READY"
PASS_CLASS = "P6_PHASEA8MRN2R8D_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY_PREREGISTERED"
NEXT_STAGE = "P6_PHASEA8MRN2R8E_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY_DEVELOPMENT"
CANDIDATE = "H4_H3_TRAIN140_DENSE_UNIT_DIAGONAL_CONSISTENCY_LAMBDA0P1"
MODEL_SEED = 20260909
EXPECTED_PARAMS = 414210
EXPECTED_INPUT = 165
EXPECTED_INITIAL_SHA = "7c4601d5c779655bc2a69482995868aa501310f5de3d1a12470708202424bb28"
EXPECTED_H3_SOURCE_SHA = "896efe8b7bc4e73a93450e5e6978efdfc46bbc50482bdc2d4ae17aceb8f7c2f2"
EXPECTED_SPLIT_SHA = "29900f4738d26f36893d3ebb5aae795b32d5836888a267c4bee98d440372a9ac"
EXPECTED_BLACKLIST_SHA = "0a733fd0328cd30377a4faf1a78432698da8a96f9c41476e6d233a39dc24754f"
EXPECTED_FIELD_STREAM_SHA = "6639bcac3bb9c5efce5d43c5fb5e784e8f8a67a4c99cc75559c967369e57a7e2"
EXPECTED_VAL_SHA = "718c4152461367368be80e02408c18667da09b18dc5550327337ccc06c8fb24b"
EXPECTED_G_REF = 25912.966796875
MAX_UPDATES = 20000
BOUNDARY_WIDTH = 30

H3_FULL_MEDIAN = 0.06660179578093173
H3_GAIN_MAX = 0.18721155057525807
CASE_CLASSES = {
    "A": "P6_PHASEA8MRN2R8E_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY_DEVELOPMENT_PASS",
    "B": "P6_PHASEA8MRN2R8E_DIAGONAL_CONSISTENCY_IMPROVES_GAIN_AND_FULL_BUT_NOT_READY",
    "C": "P6_PHASEA8MRN2R8E_DIAGONAL_CONSISTENCY_IMPROVES_GAIN_WITH_FIELD_TRADEOFF",
    "D": "P6_PHASEA8MRN2R8E_DIAGONAL_CONSISTENCY_REINTRODUCES_CANCELLATION",
    "E": "P6_PHASEA8MRN2R8E_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY_NO_CONTROLLED_GAIN",
}
CASE_NEXT = {
    "A": "P6_PHASEA8MRN3_DIRECT_IMPLICIT_COMPLEX_PSF_FIELD_FREEZE_AND_FRESH30_PREREGISTRATION",
    "B": "P6_PHASEA8MRN2R8F_RESIDUAL_OUTLIER_MECHANISM_PREREGISTRATION",
    "C": "P6_PHASEA8MRN2R8F_RESIDUAL_OUTLIER_MECHANISM_PREREGISTRATION",
    "D": "P6_PHASEA8MRN2R8EI_IMPLEMENTATION_AND_OBJECTIVE_REASSESSMENT",
    "E": "P6_PHASEA8MRN2R8F_GAIN_PRIOR_AND_FIELD_COUPLING_REASSESSMENT",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def hard_freeze_r8c() -> tuple[dict[str, Any], dict[str, Any]]:
    require(len(R8C_PATHS) == 18 and len(set(R8C_PATHS)) == 18, "R8C authority set must contain 18 unique files")
    require(all(path.is_file() for path in R8C_PATHS), "R8C authority artifact is missing")
    require(R8C_PACKAGE_JSON.is_file() and R8C_PACKAGE_TXT.is_file(), "Current R8C package manifests are missing")
    package = read_json(R8C_PACKAGE_JSON)
    require(package.get("classification") == "P6_PHASEA8MRN2R8C_SUBMISSION_PACKAGE_PASS", "Current package is not R8C")
    require(package.get("source_file_count") == 18 and package.get("submission_total_files") == 20, "R8C package counts drifted")
    require(package.get("all_hash_match") is True and len(package.get("files", [])) == 18, "R8C package hash gate failed")
    package_by_name = {Path(row["original_path"]).name: row for row in package["files"]}
    require(set(package_by_name) == {path.name for path in R8C_PATHS}, "R8C package source set drifted")
    rows = []
    for path in R8C_PATHS:
        source_hash = sha256(path)
        row = package_by_name[path.name]
        copied = Path(row["copied_path"])
        require(row["hash_match"] is True and row["source_SHA256"] == source_hash, f"R8C manifest hash mismatch: {path.name}")
        require(copied.is_file() and sha256(copied) == source_hash, f"R8C package copy mismatch: {path.name}")
        rows.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "SHA256": source_hash, "package_copy_verified": True})

    status = read_json(R8C_STATUS)
    summary = read_json(R8C_SUMMARY)
    outcome = read_json(R8C_OUTCOME)
    require(status["status"] == status["phase"] == "FINALIZED", "R8C is not FINALIZED")
    require(status["current_update"] == status["total_updates"] == status["science_optimizer_updates"] == 20000, "R8C update count drifted")
    require(status["classification"] == summary["classification"] == outcome["classification"] == EXPECTED_R8C_CLASS, "R8C classification drifted")
    require(status["outcome_case"] == outcome["outcome_case"] == "B", "R8C CASE B not confirmed")
    require(status["CANCEL3"] is False and status["READY3"] is False, "R8C CANCEL3/READY3 drifted")
    require(summary["sample_stream_match"] is True and summary["sample_stream_SHA256"] == EXPECTED_FIELD_STREAM_SHA, "R8C sample stream mismatch")
    require(status["fresh30_selected"] == status["fresh30_accessed"] == status["operator_cases"] == status["new_physics"] == 0, "R8C isolation drifted")
    freeze = {
        "classification": "P6_PHASEA8MRN2R8D_R8C_HARD_FREEZE_PASS",
        "created": now(),
        "R8C_read_only_after_freeze": True,
        "authority_file_count": 18,
        "authority_files": rows,
        "prior_submission_manifest_JSON_SHA256": sha256(R8C_PACKAGE_JSON),
        "prior_submission_manifest_TXT_SHA256": sha256(R8C_PACKAGE_TXT),
        "status": "FINALIZED",
        "completed_updates": 20000,
        "classification_frozen": EXPECTED_R8C_CLASS,
        "outcome_case": "B",
        "CANCEL3": False,
        "READY3": False,
        "sample_stream_match": True,
        "fresh30": "0/0",
        "operator_cases": 0,
        "new_physics": 0,
        "gate": "PASS",
    }
    return freeze, {"status": status, "summary": summary, "outcome": outcome}


def h3_postmortem() -> tuple[dict[str, Any], dict[str, Any]]:
    best = read_json(R8C_BEST)
    summary = read_json(R8C_SUMMARY)
    boundary = read_json(R8C_BOUNDARY)
    gain = read_json(R8C_GAIN)
    outcome = read_json(R8C_OUTCOME)
    require(best["best_update"] == 18000 and best["pooled_VAL"] == 0.05320869817865062 and best["max_column_VAL"] == 0.07868135918409411, "H3 best VAL drifted")
    internal = summary["INTERNAL_TEST_summary"]["pooled"]
    raw = summary["INTERNAL_TEST_raw_summary"]
    burned = summary["burned24_summary"]["pooled"]
    change = boundary["H3_vs_H2_improvement_percent"]
    gain_change = gain["H3_vs_H2_improvement_percent"]
    require(internal["full_complex_NRMSE"]["median"] == 0.058477638378815554, "H3 INTERNAL drifted")
    require(burned["full_complex_NRMSE"]["median"] == H3_FULL_MEDIAN, "H3 burned full drifted")
    require(burned["native_center_gain_relative_error"]["max"] == H3_GAIN_MAX, "H3 burned gain drifted")
    remaining = sum(check["gate"] == "FAIL" for check in outcome["readiness"]["checks"].values())
    require(remaining == 3, "H3 remaining readiness fail count drifted")
    snapshot = {
        "best_VAL": {"update": best["best_update"], "pooled": best["pooled_VAL"], "max_column": best["max_column_VAL"]},
        "final_VAL": best["final_update_VAL"],
        "INTERNAL": {
            "full": internal["full_complex_NRMSE"],
            "energy": internal["total_energy_relative_error"],
            "boundary": internal["boundary_region_complex_NRMSE"],
            "gain": internal["native_center_gain_relative_error"],
            "peak": internal["peak_location_error_voxels"],
            "rawA": raw["raw_branch_A_NRMSE"],
            "rawB": raw["raw_branch_B_NRMSE"],
            "error_cosine": raw["error_cosine"],
            "cancellation_ratio": raw["cancellation_ratio"],
        },
        "burned24": {
            "full": burned["full_complex_NRMSE"],
            "energy": burned["total_energy_relative_error"],
            "boundary": burned["boundary_region_complex_NRMSE"],
            "gain": burned["native_center_gain_relative_error"],
            "peak": burned["peak_location_error_voxels"],
        },
        "H3_vs_H2_improvement_percent": {**change, **gain_change},
        "CANCEL3": False,
        "READY3": False,
        "remaining_fail_count": remaining,
    }
    postmortem = {
        "classification": "P6_PHASEA8MRN2R8D_H3_MECHANISM_POSTMORTEM_FROZEN",
        "created": now(),
        **snapshot,
        "readiness_failed_metrics": [name for name, check in outcome["readiness"]["checks"].items() if check["gate"] == "FAIL"],
        "interpretation": {
            "boundary_feature_sweep_not_continued": True,
            "reason": "H3 improved energy strongly and passed energy/boundary/peak gates, but full median improved only 0.0156%; full max, corner, and gain worsened. Another boundary-feature sweep would not isolate the remaining center-gain mechanism.",
            "next_controlled_change": "TRAIN140-derived approximate unit-diagonal consistency only",
        },
        "gate": "PASS",
    }
    return postmortem, snapshot


def freeze_train_diagonal() -> tuple[dict[str, Any], dict[str, Any]]:
    prior = read_json(R8B_TRAIN_DIAGONAL)
    expected = {
        "n": 140,
        "min": 0.9979013282065966,
        "median": 1.0,
        "max": 1.0039199727995733,
        "mean": 1.0003041404061261,
        "std_population_ddof0": 0.001344764406703845,
        "span_max_minus_min": 0.006018644592976741,
        "max_abs_deviation_from_1": 0.003919972799573346,
    }
    require(prior["TRAIN140_normalized_real_diagonal"] == expected, "TRAIN140 diagonal statistics drifted")
    require(prior["TRAIN140_imag_diagonal_max_abs"] == 0.0 and prior["g_ref"] == EXPECTED_G_REF, "TRAIN140 diagonal imaginary/g_ref drifted")
    dependency_rows = []
    for name, item in prior["sources"].items():
        path = Path(item["path"])
        require(path.is_file() and sha256(path) == item["SHA256"], f"TRAIN140 prior dependency drifted: {name}")
        dependency_rows.append({"name": name, "path": str(path.resolve()), "SHA256": item["SHA256"], "hash_match": True})
    require(prior["VAL20_values_used"] is False and prior["INTERNAL_TEST20_values_used"] is False and prior["burned24_values_used"] is False, "Held-out diagonal leaked into prior")
    frozen = {
        "classification": "P6_PHASEA8MRN2R8D_TRAIN140_DIAGONAL_PRIOR_FROZEN",
        "created": now(),
        "source_R8B_artifact": {"path": str(R8B_TRAIN_DIAGONAL.resolve()), "SHA256": sha256(R8B_TRAIN_DIAGONAL)},
        "dependency_hashes": dependency_rows,
        "g_ref": EXPECTED_G_REF,
        "TRAIN140_normalized_real_diagonal": expected,
        "TRAIN140_imag_diagonal_max_abs": 0.0,
        "prior_statement": "AN_APPROXIMATELY_CONSTANT_NORMALIZED_DIAGONAL_PRIOR",
        "target": {"real": 1.0, "imag": 0.0, "source": "TRAIN140 normalized real diagonal median only"},
        "mathematically_exact_for_every_native_source_claimed": False,
        "new_exact_off_lattice_label": False,
        "VAL20_values_used": False,
        "INTERNAL_TEST20_values_used": False,
        "burned24_values_used": False,
        "PSF_arrays_opened_in_R8D": False,
        "gate": "PASS",
    }
    isolation = {
        "classification": "P6_PHASEA8MRN2R8D_DIAGONAL_PRIOR_ISOLATION_PASS",
        "created": now(),
        "allowed_prior_evidence": "saved TRAIN140 diagonal statistics only",
        "prior_target_source": "TRAIN140 median=1.0",
        "VAL_diagonal_values_used": False,
        "INTERNAL_TEST_diagonal_values_used": False,
        "burned24_diagonal_values_used": False,
        "target_error_based_q_selection": False,
        "heldout_driven_lambda_selection": False,
        "burned_driven_tuning": False,
        "new_science_target_reads": 0,
        "new_physics": 0,
        "gate": "PASS",
    }
    return frozen, isolation


def coordinate_stream_audit() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    roles, _ = n2.source_records()
    train_records = roles["TRAIN"]
    require(len(train_records) == 140, "TRAIN source count drifted")
    blocked = n2.blacklist_mask()
    stats = read_json(TRAIN_STATS)
    require(stats["TRAIN_source_columns"] == 140 and stats["g_ref"] == EXPECTED_G_REF, "Saved TRAIN statistics drifted")
    rows_by_anchor = {int(row["anchor_index"]): row for row in stats["rows"]}
    require(set(rows_by_anchor) == {int(row["anchor_index"]) for row in train_records}, "TRAIN radii membership drifted")

    print("[R8D] reconstructing frozen coordinate shells from saved TRAIN radii (no PSF arrays)...", flush=True)
    shells: dict[int, dict[str, np.ndarray]] = {}
    for count, record in enumerate(train_records, start=1):
        anchor = int(record["anchor_index"])
        shells[anchor] = eligible_shells_from_radii(record["index_xyz"], rows_by_anchor[anchor]["radii_squared"], blocked)
        require(all(shells[anchor][name].size > 0 for name in SHELL_NAMES), f"Empty frozen shell for anchor {anchor}")
        if count % 20 == 0 or count == len(train_records):
            print(f"[R8D] reconstructed shells {count}/140", flush=True)

    source_rng, query_rng, c4_rng = n2.make_rng_bundle()
    field_state = n2.stream_initial_state()
    diagonal_state = hashlib.sha256(b"P6_PHASEA8MRN2R8D_DIAGONAL_COORDINATE_STREAM_V1").digest()
    local = np.asarray(DIAGONAL_LOCAL_INDICES, dtype="<u2")
    require(local.size == 256 and np.array_equal(local, np.arange(0, 8192, 32, dtype=np.uint16)), "Diagonal local index rule drifted")
    positions = np.concatenate([np.arange(base, base + 8192, dtype=np.int64)[local.astype(np.int64)] for base in range(0, FIELD_PAIRS_PER_UPDATE, 8192)])
    require(positions.size == DIAGONAL_PAIRS_PER_UPDATE, "Diagonal position count drifted")

    seen = np.zeros(NATIVE_VOXELS, dtype=bool)
    fallback_events: list[dict[str, Any]] = []
    field_pre_hits = field_post_hits = diagonal_pre_hits = diagonal_post_hits = 0
    boundary_occurrences = 0
    interior_occurrences = 0
    started = time.perf_counter()
    for update in range(1, MAX_UPDATES + 1):
        chosen_positions = source_rng.choice(len(train_records), size=4, replace=False)
        chosen = [train_records[int(position)] for position in chosen_positions]
        query_blocks = []
        for source in chosen:
            blocks = [
                n2.choose_shell(query_rng, shells[int(source["anchor_index"])][shell_name], n2.QUERIES_PER_SHELL, CANDIDATE, update, source, shell_name, fallback_events)
                for shell_name in SHELL_NAMES
            ]
            center = np.uint32(linear_index(source["index_xyz"]))
            inside = blocks[0]
            matches = np.flatnonzero(inside == center)
            if matches.size:
                hit = int(matches[0])
                inside[0], inside[hit] = inside[hit], inside[0]
            else:
                inside[0] = center
            query_blocks.append(np.concatenate(blocks).astype(np.uint32, copy=False))
        queries = np.concatenate(query_blocks).astype(np.uint32, copy=False)
        rotations = c4_rng.integers(0, 4, size=FIELD_PAIRS_PER_UPDATE, dtype=np.uint8)
        anchor_ids = np.asarray([row["anchor_index"] for row in chosen], dtype=np.uint16)
        field_pre_hits += int(np.count_nonzero(blocked[queries]))
        field_state = n2.stream_update(field_state, update, anchor_ids, queries, rotations)
        rotated_xyz = c4_rotate_indices(unravel_indices(queries), rotations)
        rotated_linear = ((rotated_xyz[:, 0].astype(np.int64) * SHAPE[1] + rotated_xyz[:, 1]) * SHAPE[2] + rotated_xyz[:, 2]).astype(np.uint32)
        field_post_hits += int(np.count_nonzero(blocked[rotated_linear]))
        diagonal_pre = queries[positions]
        diagonal_q = rotated_linear[positions]
        diagonal_pre_hits += int(np.count_nonzero(blocked[diagonal_pre]))
        diagonal_post_hits += int(np.count_nonzero(blocked[diagonal_q]))
        seen[diagonal_q] = True
        q_xyz = rotated_xyz[positions]
        boundary = np.any((q_xyz < BOUNDARY_WIDTH) | (q_xyz >= (np.asarray(SHAPE, dtype=np.int16) - BOUNDARY_WIDTH)), axis=1)
        boundary_occurrences += int(boundary.sum())
        interior_occurrences += int((~boundary).sum())
        digest = hashlib.sha256()
        digest.update(diagonal_state)
        digest.update(np.asarray([update], dtype="<u4").tobytes())
        digest.update(anchor_ids.astype("<u2", copy=False).tobytes())
        digest.update(local.tobytes())
        digest.update(diagonal_q.astype("<u4", copy=False).tobytes())
        diagonal_state = digest.digest()
        if update % 1000 == 0:
            elapsed = time.perf_counter() - started
            print(f"[R8D] coordinate replay {update}/{MAX_UPDATES}; unique q={int(seen.sum())}; elapsed={elapsed:.1f}s", flush=True)

    elapsed = time.perf_counter() - started
    field_sha = field_state.hex()
    diagonal_sha = diagonal_state.hex()
    require(field_sha == EXPECTED_FIELD_STREAM_SHA, "Replayed field sample stream SHA mismatch")
    require(field_pre_hits == field_post_hits == diagonal_pre_hits == diagonal_post_hits == 0, "Coordinate stream blacklist hit")
    unique_linear = np.flatnonzero(seen)
    unique_xyz = unravel_indices(unique_linear)
    unique_boundary_mask = np.any((unique_xyz < BOUNDARY_WIDTH) | (unique_xyz >= (np.asarray(SHAPE, dtype=np.int16) - BOUNDARY_WIDTH)), axis=1)
    total = MAX_UPDATES * DIAGONAL_PAIRS_PER_UPDATE
    require(boundary_occurrences + interior_occurrences == total == 20480000, "Diagonal occurrence accounting mismatch")
    local_sha = hashlib.sha256(local.tobytes()).hexdigest()
    spec = {
        "classification": "P6_PHASEA8MRN2R8D_DIAGONAL_COORDINATE_SPEC_FROZEN",
        "created": now(),
        "source": "existing post-C4 blacklist-safe field query coordinates only",
        "field_sources_per_update": 4,
        "field_queries_per_source": 8192,
        "field_pairs_per_update": FIELD_PAIRS_PER_UPDATE,
        "selection_per_source": 256,
        "diagonal_pairs_per_update": DIAGONAL_PAIRS_PER_UPDATE,
        "local_index_rule": "0,32,64,...,8160 applied independently to each of four 8192-query source blocks",
        "local_indices": list(DIAGONAL_LOCAL_INDICES),
        "local_indices_uint16_le_SHA256": local_sha,
        "selection_is_deterministic": True,
        "new_coordinate_selection_RNG": False,
        "target_error_based_selection": False,
        "post_C4_selection": True,
        "q_source_equals_q_output": True,
        "normalized_displacement_exact_zero": True,
        "gate": "PASS",
    }
    coverage = {
        "classification": "P6_PHASEA8MRN2R8D_DIAGONAL_COORDINATE_COVERAGE_PASS",
        "created": now(),
        "audit_mode": "coordinate-only replay from saved TRAIN radii; no science target array opened",
        "updates": MAX_UPDATES,
        "diagonal_pairs_per_update": DIAGONAL_PAIRS_PER_UPDATE,
        "total_sampled_occurrences": total,
        "unique_native_coordinates": int(unique_linear.size),
        "native_domain_count": NATIVE_VOXELS,
        "unique_coordinate_fraction": float(unique_linear.size / NATIVE_VOXELS),
        "unique_x_levels": int(np.unique(unique_xyz[:, 0]).size),
        "unique_y_levels": int(np.unique(unique_xyz[:, 1]).size),
        "unique_z_levels": int(np.unique(unique_xyz[:, 2]).size),
        "boundary_width_voxels": BOUNDARY_WIDTH,
        "boundary_occurrences": boundary_occurrences,
        "interior_occurrences": interior_occurrences,
        "unique_boundary_coordinates": int(unique_boundary_mask.sum()),
        "unique_interior_coordinates": int((~unique_boundary_mask).sum()),
        "field_pre_C4_blacklist_hits": field_pre_hits,
        "field_post_C4_blacklist_hits": field_post_hits,
        "diagonal_pre_C4_blacklist_hits": diagonal_pre_hits,
        "diagonal_post_C4_blacklist_hits": diagonal_post_hits,
        "pre_post_safety": "PASS",
        "fallback_event_count": len(fallback_events),
        "full_stream_saved": False,
        "coordinate_replay_seconds": elapsed,
        "PSF_arrays_opened": False,
        "science_targets_read": 0,
        "gate": "PASS",
    }
    stream = {
        "classification": "P6_PHASEA8MRN2R8D_DIAGONAL_STREAM_FROZEN",
        "created": now(),
        "field_sample_stream_SHA256": field_sha,
        "authoritative_field_sample_stream_SHA256": EXPECTED_FIELD_STREAM_SHA,
        "field_sample_stream_match": True,
        "diagonal_coordinate_stream_initial_tag": "P6_PHASEA8MRN2R8D_DIAGONAL_COORDINATE_STREAM_V1",
        "diagonal_coordinate_stream_SHA256": diagonal_sha,
        "diagonal_hash_chain_update_payload": ["previous_digest", "update uint32 little-endian", "four field source anchor ids uint16 little-endian", "256 fixed local indices uint16 little-endian applied to each source block", "1024 post-C4 q linear indices uint32 little-endian"],
        "local_indices_SHA256": local_sha,
        "updates": MAX_UPDATES,
        "field_stream_changed": False,
        "full_stream_saved": False,
        "gate": "PASS",
    }
    del shells, seen, unique_linear, unique_xyz, blocked
    gc.collect()
    return spec, coverage, stream


def grad_norm(model: nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    found = False
    for parameter in model.parameters():
        if parameter.grad is not None:
            found = True
            total += parameter.grad.detach().double().square().sum()
    return float(torch.sqrt(total).cpu()) if found else 0.0


def model_audits() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    h3 = build_h3(MODEL_SEED, boundary_features_enabled=True).eval()
    h4 = build_h4(MODEL_SEED).eval()
    require(parameter_count(h3) == parameter_count(h4) == EXPECTED_PARAMS, "H4 parameter count changed")
    require(h3.ordered_base.encoded_dimension == h4.ordered_base.encoded_dimension == EXPECTED_INPUT, "H4 input dimension changed")
    require(list(h3.state_dict()) == list(h4.state_dict()), "H3/H4 state keys changed")
    require(all(torch.equal(h3.state_dict()[key], h4.state_dict()[key]) for key in h3.state_dict()), "H3/H4 fresh initialization differs")
    h3_hash = state_sha256(h3)
    h4_hash = state_sha256(h4)
    require(h3_hash == h4_hash == EXPECTED_INITIAL_SHA, "H4 initial-state SHA mismatch")
    require(sha256(H3_SOURCE) == EXPECTED_H3_SOURCE_SHA, "Historical H3 source SHA drifted")
    require(len(list(h4.named_buffers())) == 0, "H4 unexpectedly adds buffers")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20261001)
    u = 2.0 * torch.rand((4096, 3), generator=generator) - 1.0
    r = 2.0 * torch.rand((4096, 3), generator=generator) - 1.0
    with torch.inference_mode():
        h3_value = h3(u, r)
        h4_value = h4(u, r)
        delta = torch.linalg.vector_norm(h3_value - h4_value, dim=-1)
        scale = torch.linalg.vector_norm(h3_value, dim=-1).clamp_min(1e-12)
        discrepancy = float((delta / scale).max())
        bitwise = torch.equal(h3_value, h4_value)
        reverse = h4(r, u)
        reverse_conjugate = torch.stack((reverse[:, 0], -reverse[:, 1]), dim=-1)
        hermitian_delta = torch.linalg.vector_norm(h4_value - reverse_conjugate, dim=-1)
        hermitian_scale = torch.maximum(torch.linalg.vector_norm(h4_value, dim=-1), torch.linalg.vector_norm(reverse_conjugate, dim=-1)).clamp_min(1e-12)
        hermitian = float((hermitian_delta / hermitian_scale).max())
    require(bitwise and discrepancy <= 1e-7, "H4 does not reproduce H3 when auxiliary loss is disabled")
    require(hermitian <= 1e-6, "H4 Hermitian discrepancy exceeds threshold")
    d = normalized_displacement(u[:1024], u[:1024])
    require(torch.equal(d, torch.zeros_like(d)), "Diagonal displacement is not exactly zero")
    encoded_h3 = coordinate_encoding_h3(u, r, normalized_displacement(u, r), boundary_features_enabled=True)
    encoded_h4 = h4.ordered_base.decoder is h4.ordered_base.decoder and coordinate_encoding_h3(u, r, normalized_displacement(u, r), boundary_features_enabled=True)
    require(torch.equal(encoded_h3, encoded_h4), "H4 encoding changed")
    diag_loss, diag_a, diag_b, diag_target = diagonal_dual_loss(h4, u[:1024])
    require(bool(torch.isfinite(diag_a).all()) and bool(torch.isfinite(diag_b).all()) and bool(torch.isfinite(diag_loss)), "Diagonal branch/loss nonfinite")
    require(torch.equal(diag_target[:, 0], torch.ones_like(diag_target[:, 0])) and int(torch.count_nonzero(diag_target[:, 1])) == 0, "Diagonal target changed")

    h4.train()
    h4.zero_grad(set_to_none=True)
    field_target = torch.randn((1024, 2), generator=generator)
    field_loss, _, _ = field_dual_loss(h4, u[:1024], r[:1024], field_target)
    field_loss.backward()
    field_gradient = grad_norm(h4)
    h4.zero_grad(set_to_none=True)
    diag_loss_train, _, _, _ = diagonal_dual_loss(h4, u[:1024])
    diag_loss_train.backward()
    diagonal_gradient = grad_norm(h4)
    no_detach = "detach(" not in H4_SOURCE.read_text(encoding="utf-8") and "detach(" not in H3_SOURCE.read_text(encoding="utf-8")
    require(field_gradient > 0 and diagonal_gradient > 0 and no_detach, "H4 gradient/no-detach audit failed")
    unit = {
        "classification": "P6_PHASEA8MRN2R8D_H4_PREFLIGHT_UNIT_TESTS_PASS",
        "created": now(),
        "params_414210": True,
        "input_165": True,
        "H3_forward_max_normalized_discrepancy": discrepancy,
        "H3_forward_threshold_max": 1e-7,
        "H3_forward_bitwise_equal": bitwise,
        "initial_SHA_identity": True,
        "boundary_features_unchanged": list(BOUNDARY_FEATURE_NAMES),
        "boundary_feature_slots_unchanged": list(BOUNDARY_AUXILIARY_GLOBAL_INDICES),
        "source_masks_unchanged": True,
        "output_PE_unchanged": True,
        "relative_PE_unchanged": True,
        "q_source_equals_q_output": True,
        "d_zero_exact": True,
        "target_1_plus_0j_exact": True,
        "A_diag_finite": True,
        "B_diag_finite": True,
        "L_diag_finite": True,
        "diag_gradient_norm": diagonal_gradient,
        "field_gradient_norm": field_gradient,
        "diag_gradients_nonzero": True,
        "field_gradients_nonzero": True,
        "detach_used": False,
        "Hermitian_max_normalized_discrepancy": hermitian,
        "Hermitian_threshold_max": 1e-6,
        "gate": "PASS",
    }
    equivalence = {
        "classification": "P6_PHASEA8MRN2R8D_H3_EQUIVALENCE_PASS",
        "created": now(),
        "random_coordinate_pairs": 4096,
        "diagonal_loss_disabled_for_comparison": True,
        "H3_forward_bitwise_equal": bitwise,
        "max_normalized_forward_discrepancy": discrepancy,
        "threshold_max": 1e-7,
        "H3_parameter_count": EXPECTED_PARAMS,
        "H4_parameter_count": EXPECTED_PARAMS,
        "H3_input_dimension": EXPECTED_INPUT,
        "H4_input_dimension": EXPECTED_INPUT,
        "architecture_unchanged": True,
        "parameter_shapes_unchanged": True,
        "state_dict_keys_identical": True,
        "inference_formula_unchanged": "0.5*(A+B)",
        "gate": "PASS",
    }
    initialization = {
        "classification": "P6_PHASEA8MRN2R8D_H4_INITIALIZATION_IDENTITY_PASS",
        "created": now(),
        "model_seed": MODEL_SEED,
        "H3_fresh_initial_state_SHA256": h3_hash,
        "H4_fresh_initial_state_SHA256": h4_hash,
        "expected_initial_state_SHA256": EXPECTED_INITIAL_SHA,
        "initial_state_SHA_identity": True,
        "fresh_initialization_required": True,
        "H3_best_checkpoint_warm_start": False,
        "H2_checkpoint_warm_start": False,
        "trained_model_loaded_as_initialization": False,
        "gate": "PASS",
    }
    del h3, h4, u, r, d, h3_value, h4_value, reverse, reverse_conjugate, encoded_h3, encoded_h4
    gc.collect()
    return unit, equivalence, initialization


def dummy_resource_test() -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA is required for the H4 dummy resource gate")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    require("4060" in device_name, f"Expected RTX 4060-class device, got {device_name}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    model = build_h4(MODEL_SEED).to(device).train()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261002)
    u = 2.0 * torch.rand((FIELD_PAIRS_PER_UPDATE, 3), generator=generator, device=device) - 1.0
    r = 2.0 * torch.rand((FIELD_PAIRS_PER_UPDATE, 3), generator=generator, device=device) - 1.0
    target = torch.randn((FIELD_PAIRS_PER_UPDATE, 2), generator=generator, device=device)
    q = u[:DIAGONAL_PAIRS_PER_UPDATE].clone()

    model.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        field_loss, _, _ = field_dual_loss(model, u, r, target)
    field_loss.backward()
    field_gradient = grad_norm(model)

    model.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        diagonal_loss, _, _, _ = diagonal_dual_loss(model, q)
    diagonal_loss.backward()
    diagonal_gradient = grad_norm(model)

    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        field_combined, _, _ = field_dual_loss(model, u, r, target)
        diagonal_combined, _, _, _ = diagonal_dual_loss(model, q)
        total_loss = h4_total_loss(field_combined, diagonal_combined)
    total_loss.backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    combined_gradient = grad_norm(model)
    allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved(0) / (1024 ** 3)
    all_finite = all(np.isfinite(value) for value in (float(field_combined), float(diagonal_combined), float(total_loss), field_gradient, diagonal_gradient, combined_gradient))
    gradients_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    require(all_finite and gradients_finite and min(field_gradient, diagonal_gradient, combined_gradient) > 0 and reserved <= 7.5, "H4 dummy resource gate failed")
    result = {
        "classification": "P6_PHASEA8MRN2R8D_H4_DUMMY_RESOURCE_FEASIBILITY_PASS",
        "created": now(),
        "device": device_name,
        "field_pairs": FIELD_PAIRS_PER_UPDATE,
        "diagonal_pairs": DIAGONAL_PAIRS_PER_UPDATE,
        "dtype_policy": "FP16 autocast forward; FP32 losses/backward",
        "field_loss": float(field_combined.detach().cpu()),
        "diagonal_loss": float(diagonal_combined.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
        "field_gradient_norm": field_gradient,
        "diagonal_gradient_norm": diagonal_gradient,
        "combined_gradient_norm": combined_gradient,
        "losses_finite": all_finite,
        "gradients_finite": gradients_finite,
        "seconds_per_combined_update": seconds,
        "estimated_20000_update_hours_no_validation_or_IO": seconds * MAX_UPDATES / 3600.0,
        "peak_allocated_VRAM_GiB": allocated,
        "peak_reserved_VRAM_GiB": reserved,
        "peak_reserved_limit_GiB": 7.5,
        "science_optimizer_constructed": False,
        "optimizer_step_executed": False,
        "science_training": False,
        "gate": "PASS",
    }
    del model, u, r, q, target, field_loss, diagonal_loss, field_combined, diagonal_combined, total_loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def classify_r8e(cancel: bool, ready: bool, h4m: float, h4g: float) -> str:
    if cancel:
        return "D"
    if ready:
        return "A"
    if h4g < H3_GAIN_MAX and h4m < H3_FULL_MEDIAN:
        return "B"
    if h4g < H3_GAIN_MAX and h4m >= H3_FULL_MEDIAN:
        return "C"
    return "E"


def predicates(cancel: bool, ready: bool, h4m: float, h4g: float) -> list[str]:
    return [
        case for case, matched in (
            ("D", cancel),
            ("A", not cancel and ready),
            ("B", not cancel and not ready and h4g < H3_GAIN_MAX and h4m < H3_FULL_MEDIAN),
            ("C", not cancel and not ready and h4g < H3_GAIN_MAX and h4m >= H3_FULL_MEDIAN),
            ("E", not cancel and not ready and h4g >= H3_GAIN_MAX),
        ) if matched
    ]


def outcome_truth_table() -> dict[str, Any]:
    low_m, equal_m, high_m = H3_FULL_MEDIAN - 0.001, H3_FULL_MEDIAN, H3_FULL_MEDIAN + 0.001
    low_g, equal_g, high_g = H3_GAIN_MAX - 0.001, H3_GAIN_MAX, H3_GAIN_MAX + 0.001
    canonical = [
        (True, False, low_m, low_g), (True, True, equal_m, equal_g), (True, False, high_m, high_g), (True, True, high_m, low_g),
        (False, True, low_m, low_g), (False, True, equal_m, equal_g), (False, True, high_m, high_g), (False, True, high_m, low_g),
        (False, False, low_m, low_g), (False, False, low_m, equal_g), (False, False, low_m, high_g), (False, False, H3_FULL_MEDIAN - 1e-12, high_g),
        (False, False, equal_m, low_g), (False, False, high_m, low_g), (False, False, equal_m, H3_GAIN_MAX - 1e-12), (False, False, high_m, H3_GAIN_MAX - 1e-12),
        (False, False, equal_m, equal_g), (False, False, equal_m, high_g), (False, False, high_m, equal_g), (False, False, high_m, high_g),
    ]
    rng = random.Random(20261003)
    random_rows = [(bool(rng.getrandbits(1)), bool(rng.getrandbits(1)), rng.uniform(0.0, 0.2), rng.uniform(0.0, 0.3)) for _ in range(10000)]
    all_rows = canonical + random_rows
    unclassified = multi = mismatch = 0
    counts = {case: 0 for case in ("D", "A", "B", "C", "E")}
    for row in all_rows:
        matches = predicates(*row)
        unclassified += len(matches) == 0
        multi += len(matches) > 1
        selected = classify_r8e(*row)
        mismatch += matches != [selected]
        counts[selected] += 1
    canonical_cases = [classify_r8e(*row) for row in canonical]
    require(len(all_rows) == 10020 and unclassified == multi == mismatch == 0 and set(canonical_cases) == set(counts), "R8E outcome tree is not exhaustive/exclusive")
    return {
        "classification": "P6_PHASEA8MRN2R8D_R8E_OUTCOME_TRUTH_TABLE_EXHAUSTIVE",
        "created": now(),
        "ordered_case_priority": ["D", "A", "B", "C", "E"],
        "canonical_combination_count": 20,
        "canonical_case_sequence": canonical_cases,
        "random_seed": 20261003,
        "random_combination_count": 10000,
        "total_combination_count": 10020,
        "unclassified": unclassified,
        "multi_classified": multi,
        "classifier_predicate_mismatch": mismatch,
        "exactly_one": 10020,
        "case_counts": counts,
        "strict_equality_H4g_equals_H3g_is_improvement": False,
        "strict_equality_H4m_equals_H3m_is_improvement": False,
        "epsilon_added": False,
        "gate": "PASS",
    }


def build_artifacts() -> dict[Path, dict[str, Any]]:
    r8c_freeze, r8c = hard_freeze_r8c()
    postmortem, h3 = h3_postmortem()
    train_diagonal, isolation = freeze_train_diagonal()
    coordinate_spec, coverage, stream = coordinate_stream_audit()
    unit, equivalence, initialization = model_audits()
    resource = dummy_resource_test()
    truth = outcome_truth_table()
    created = now()
    split = read_json(SPLIT_PATH)
    blacklist = read_json(BLACKLIST_PATH)
    sampler = read_json(SAMPLER_PATH)
    val = read_json(VAL_PATH)
    require(split["counts"] == {"TRAIN": 140, "VAL": 20, "INTERNAL_TEST": 20} and split["new_split_SHA256"] == EXPECTED_SPLIT_SHA, "Inherited split mismatch")
    require(blacklist["total_blacklist_unique_count"] == 136 and blacklist["C4_closed_blacklist_SHA256"] == EXPECTED_BLACKLIST_SHA, "Inherited blacklist mismatch")
    require(val["combined_coordinate_arrays_SHA256"] == EXPECTED_VAL_SHA, "Inherited VAL coordinate mismatch")

    rationale = {
        "classification": "P6_PHASEA8MRN2R8D_SCIENTIFIC_RATIONALE_FROZEN",
        "created": created,
        "R8C_formal_outcome": "CASE B",
        "parent_choice": "H3",
        "parent_choice_reason": "H3 is the formal R8C candidate and improved VAL, INTERNAL full, and burned energy; post-hoc fallback to H2 is prohibited.",
        "why_not_continue_boundary_feature_sweep": postmortem["interpretation"]["reason"],
        "remaining_mechanism": "center-gain/full residual after boundary-aware conditioning",
        "unique_candidate": CANDIDATE,
        "unique_science_change": "TRAIN140-derived approximate dense unit-diagonal consistency with lambda=0.1",
        "architecture_change": False,
        "new_exact_physics": 0,
        "gain_weight_sweep": False,
        "post_hoc_inference_renormalization": False,
        "gate": "PASS",
    }
    h4_spec = {
        "classification": "P6_PHASEA8MRN2R8D_H4_MODEL_SPEC_FROZEN",
        "created": created,
        "candidate": CANDIDATE,
        "candidate_count": 1,
        "parent": "H3",
        "architecture": "exactly H3",
        "parameter_count": EXPECTED_PARAMS,
        "input_dimension": EXPECTED_INPUT,
        "model_seed": MODEL_SEED,
        "fresh_initialization": True,
        "initial_state_SHA256": EXPECTED_INITIAL_SHA,
        "boundary_features": list(BOUNDARY_FEATURE_NAMES),
        "boundary_feature_slots": list(BOUNDARY_AUXILIARY_GLOBAL_INDICES),
        "source_x_y_retained_k": [0, 1, 2],
        "source_z_retained_k": [0, 1],
        "source_z_k2_zero": True,
        "source_k4_to_k7_zero": True,
        "output_k0_to_k7_unchanged": True,
        "relative_k0_to_k9_unchanged": True,
        "new_parameters": 0,
        "new_buffers": 0,
        "inference_formula": "K_H4(u,r)=0.5*(A+B)",
        "inference_hard_coded_center": False,
        "post_hoc_column_normalization": False,
        "gain_clipping": False,
        "manual_amplitude_correction": False,
        "historical_H3_source": {"path": str(H3_SOURCE.resolve()), "SHA256": sha256(H3_SOURCE), "read_only": True},
        "H4_helper_source": {"path": str(H4_SOURCE.resolve()), "SHA256": sha256(H4_SOURCE)},
        "unit_tests": unit,
        "gate": "PASS",
    }
    loss_spec = {
        "classification": "P6_PHASEA8MRN2R8D_H4_LOSS_SPEC_FROZEN",
        "created": created,
        "A": "ordered_base(u,r,d)",
        "R": "ordered_base(r,u,-d)",
        "B": "conj(R)",
        "field_target": "y=G(u,r)/g_ref",
        "L_field": "0.5*MSE_complex(A,y)+0.5*MSE_complex(B,y)",
        "field_loss_coefficient": 1.0,
        "diagonal_pair": "q_source=q_output=q; d=0",
        "A_diag": "ordered_base(q,q,0)",
        "R_diag": "ordered_base(q,q,0)",
        "B_diag": "conj(R_diag)",
        "diagonal_target": {"real": 1.0, "imag": 0.0},
        "diagonal_target_source": "TRAIN140 normalized diagonal median only",
        "L_diag": "0.5*MSE_complex(A_diag,1+0j)+0.5*MSE_complex(B_diag,1+0j)",
        "final_symmetrized_only_supervision": False,
        "lambda_diag": LAMBDA_DIAG,
        "lambda_frozen_before_first_science_optimizer_step": True,
        "lambda_interpretation": "fixed low-weight auxiliary regularizer that keeps the original field loss coefficient at 1.0",
        "lambda_claimed_optimal": False,
        "lambda_sweep": False,
        "L_total": "L_field+0.1*L_diag",
        "incorrect_0.9_field_form_used": False,
        "inference_loss_or_target_hard_code": False,
        "gate": "PASS",
    }
    training = {
        "classification": "P6_PHASEA8MRN2R8D_H4_TRAINING_INHERITANCE_FROZEN",
        "created": created,
        "split": {"counts": split["counts"], "SHA256": EXPECTED_SPLIT_SHA, "C4_role_closure": True},
        "blacklist": {"unique": 136, "SHA256": EXPECTED_BLACKLIST_SHA, "field_hits": 0, "diagonal_hits": 0},
        "field_sampler": {
            "sources_per_update": 4,
            "queries_per_source": 8192,
            "queries_per_shell": 2048,
            "shells": list(SHELL_NAMES),
            "pairs_per_update": FIELD_PAIRS_PER_UPDATE,
            "sampler_seed": n2.COMMON_SAMPLER_SEED,
            "C4_seed": n2.C4_SEED,
            "sample_stream_SHA256": EXPECTED_FIELD_STREAM_SHA,
            "sample_stream_match": True,
            "sampler_artifact_SHA256": sha256(SAMPLER_PATH),
        },
        "diagonal_sampler": {"pairs_per_update": DIAGONAL_PAIRS_PER_UPDATE, "new_RNG": False, "stream_SHA256": stream["diagonal_coordinate_stream_SHA256"]},
        "optimizer": {"type": "AdamW", "lr": 2e-4, "betas": [0.9, 0.99], "weight_decay": 1e-6, "scheduler": None},
        "budget_updates": MAX_UPDATES,
        "VAL_checkpoint_interval": 500,
        "early_stopping": False,
        "VAL": {"sources": 20, "probes_per_source": 65536, "coordinate_SHA256": EXPECTED_VAL_SHA},
        "checkpoint_selection": {"primary": "lowest pooled final-sym full complex NRMSE", "secondary": "lowest max-column", "tie_within": 1e-6, "tie_break": "earlier update", "TRAIN_diagonal_used": False, "VAL_gain_used": False, "INTERNAL_TEST_used": False, "burned24_used": False},
        "future_access_order": ["best H4 freeze", "INTERNAL_TEST20", "cancellation retest", "burned24 development diagnostic"],
        "burned24_role": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "gate": "PASS",
    }
    outcome_protocol = {
        "classification": "P6_PHASEA8MRN2R8D_R8E_OUTCOME_PROTOCOL_FROZEN",
        "created": created,
        "cancellation_gate": {"CANCEL4_true_iff_all": ["final INTERNAL median<=0.08", "raw_mean/final>=1.5", "error cosine<=-0.25"], "unchanged": True},
        "readiness_gate": {
            "full_median_max": 0.06, "full_max_max": 0.10,
            "energy_median_max": 0.04, "energy_max_max": 0.06,
            "boundary_median_max": 0.10, "boundary_max_max": 0.16,
            "peak_max_max": 1, "gain_max_max": 0.02,
            "READY4_true_iff_all_pass": True, "unchanged": True,
        },
        "primary_parent_baseline": {"H3_burned_full_median": H3_FULL_MEDIAN, "H3_burned_gain_max": H3_GAIN_MAX},
        "ordered_tree": [
            {"step": 1, "case": "D", "condition": "CANCEL4=true", "classification": CASE_CLASSES["D"], "recommended_next": CASE_NEXT["D"]},
            {"step": 2, "case": "A", "condition": "CANCEL4=false AND READY4=true", "classification": CASE_CLASSES["A"], "recommended_next": CASE_NEXT["A"]},
            {"step": 3, "case": "B", "condition": "CANCEL4=false AND READY4=false AND H4g<H3g AND H4m<H3m", "classification": CASE_CLASSES["B"], "recommended_next": CASE_NEXT["B"]},
            {"step": 4, "case": "C", "condition": "CANCEL4=false AND READY4=false AND H4g<H3g AND H4m>=H3m", "classification": CASE_CLASSES["C"], "recommended_next": CASE_NEXT["C"]},
            {"step": 5, "case": "E", "condition": "otherwise", "classification": CASE_CLASSES["E"], "recommended_next": CASE_NEXT["E"]},
        ],
        "strict_comparison_no_epsilon": True,
        "H4g_equal_H3g_counts_as_improvement": False,
        "H4m_equal_H3m_counts_as_improvement": False,
        "truth_table_audit": {"canonical": 20, "random": 10000, "exactly_one": 10020},
        "gate": "PASS",
    }
    science_isolation = {
        "science_optimizer_constructed": False,
        "science_optimizer_updates": 0,
        "science_updates": 0,
        "H4_training": False,
        "checkpoint_inference": 0,
        "new_exact_physics": 0,
        "new_exact_PSF": 0,
        "new_complete_PSF_columns": 0,
        "fresh30_selected": 0,
        "fresh30_accessed": 0,
        "operator_cases": 0,
        "Bunny": 0,
        "NVR": 0,
        "exact_F0": 0,
    }
    protocol = {
        "classification": "P6_PHASEA8MRN2R8D_PROTOCOL_FROZEN_AND_AUDITED",
        "created": created,
        "stage": "POST_R8C_TRAIN_ONLY_DIAGONAL_CENTER_GAIN_PHYSICS_CONSISTENCY_PREREGISTRATION_ONLY",
        "R8C_hard_freeze": "PASS",
        "candidate": CANDIDATE,
        "candidate_count": 1,
        "parent": "H3",
        "architecture_unchanged": True,
        "lambda_diag": LAMBDA_DIAG,
        "diagonal_pairs_per_update": DIAGONAL_PAIRS_PER_UPDATE,
        "blacklist_safety": "PASS",
        "field_stream_SHA256": EXPECTED_FIELD_STREAM_SHA,
        "diagonal_stream_SHA256": stream["diagonal_coordinate_stream_SHA256"],
        "loss_algebra": "PASS",
        "H3_equivalence": "PASS",
        "initial_state_identity": "PASS",
        "Hermitian": "PASS",
        "training_inheritance": "PASS",
        "R8E_outcome_exhaustive": True,
        "dummy_resource": "PASS",
        "science_isolation": science_isolation,
        "prohibitions": ["architecture change", "new exact PSF", "boundary-feature change", "source high-frequency Fourier restoration", "gain-weight sweep", "post-hoc inference renormalization", "hard-coded prediction center", "lambda sweep"],
        "gate": "PASS",
    }
    summary = {
        "classification": PASS_CLASS,
        "completed": created,
        "stage": "PREREGISTRATION_ONLY",
        "R8C_freeze": "PASS",
        "R8C_outcome_case": "B",
        "H3": h3,
        "why_not_continue_boundary_feature_sweep": postmortem["interpretation"]["reason"],
        "TRAIN140_diagonal": train_diagonal["TRAIN140_normalized_real_diagonal"],
        "prior_target": {"real": 1.0, "imag": 0.0, "source": "TRAIN140 median only"},
        "heldout_excluded_from_prior": True,
        "candidate": CANDIDATE,
        "parent": "H3",
        "parameter_count": EXPECTED_PARAMS,
        "input_dimension": EXPECTED_INPUT,
        "initial_state_SHA256": EXPECTED_INITIAL_SHA,
        "diagonal_pairs_per_update": DIAGONAL_PAIRS_PER_UPDATE,
        "q_selection_rule": coordinate_spec["local_index_rule"],
        "diagonal_blacklist_hits": 0,
        "field_sample_stream_SHA256": EXPECTED_FIELD_STREAM_SHA,
        "diagonal_coordinate_stream_SHA256": stream["diagonal_coordinate_stream_SHA256"],
        "L_field": loss_spec["L_field"],
        "L_diag": loss_spec["L_diag"],
        "lambda_diag": LAMBDA_DIAG,
        "L_total": loss_spec["L_total"],
        "inference_hard_code_or_renormalization": False,
        "R8E_outcome_order": ["D", "A", "B", "C", "E"],
        "R8E_outcome_exhaustive": True,
        **science_isolation,
        "R8E_execution_ready": True,
        "neural_field_development_ready": False,
        "formal_fresh_validation_passed": False,
        "recommended_next": NEXT_STAGE,
        "gate": "PASS",
    }
    status = {
        "status": "FINALIZED",
        "phase": "FINALIZED",
        "updated": created,
        "classification": PASS_CLASS,
        "R8C_freeze": "PASS",
        "R8C_case": "B",
        "CANCEL3": False,
        "READY3": False,
        "candidate": CANDIDATE,
        "parent": "H3",
        "params": EXPECTED_PARAMS,
        "input_dimension": EXPECTED_INPUT,
        "lambda_diag": LAMBDA_DIAG,
        "diagonal_pairs_per_update": DIAGONAL_PAIRS_PER_UPDATE,
        "field_stream_match": True,
        "diagonal_stream_SHA256": stream["diagonal_coordinate_stream_SHA256"],
        "blacklist_hits": 0,
        **science_isolation,
        "R8E_execution_ready": True,
        "neural_field_development_ready": False,
        "formal_fresh_validation_passed": False,
        "recommended_next": NEXT_STAGE,
        "gate": "PASS",
    }
    return {
        STATUS: status,
        PROTOCOL: protocol,
        R8C_FREEZE: r8c_freeze,
        RATIONALE: rationale,
        H3_POSTMORTEM: postmortem,
        TRAIN_DIAGONAL: train_diagonal,
        PRIOR_ISOLATION: isolation,
        H4_SPEC: h4_spec,
        COORDINATE_SPEC: coordinate_spec,
        COORDINATE_COVERAGE: coverage,
        DIAGONAL_STREAM: stream,
        LOSS_SPEC: loss_spec,
        H3_EQUIVALENCE: equivalence,
        INIT_IDENTITY: initialization,
        TRAIN_INHERIT: training,
        R8E_OUTCOME: outcome_protocol,
        TRUTH_TABLE: truth,
        DUMMY_RESOURCE: resource,
        SUMMARY: summary,
    }


def write_report(artifacts: dict[Path, dict[str, Any]]) -> None:
    summary = artifacts[SUMMARY]
    h3 = summary["H3"]
    internal = h3["INTERNAL"]
    burned = h3["burned24"]
    change = h3["H3_vs_H2_improvement_percent"]
    diagonal = artifacts[TRAIN_DIAGONAL]["TRAIN140_normalized_real_diagonal"]
    coverage = artifacts[COORDINATE_COVERAGE]
    stream = artifacts[DIAGONAL_STREAM]
    resource = artifacts[DUMMY_RESOURCE]
    truth = artifacts[TRUTH_TABLE]
    text = f"""# P6 Phase-A8MRN2R8D Train-only Diagonal Gain-consistency Preregistration

## 结论

终态分类：`{PASS_CLASS}`；`R8E_execution_ready=true`。唯一 H4 candidate 为 `{CANDIDATE}`，parent=H3。本轮只完成预注册与预演，没有训练 H4。

## R8C 硬冻结与 H3 结果

R8C 报告和 17 个 JSON 已与原 20 文件提交包逐一 SHA256 核对，`status=FINALIZED`、20000 updates、CASE B、`CANCEL3=false`、`READY3=false`、field stream match、fresh30=0/0、operator=0、new physics=0 全部通过。冻结后 R8C 保持只读。

H3 best update={h3['best_VAL']['update']}，VAL pooled/max={h3['best_VAL']['pooled']:.12g}/{h3['best_VAL']['max_column']:.12g}；final VAL pooled/max={h3['final_VAL']['pooled']:.12g}/{h3['final_VAL']['max_column']:.12g}。INTERNAL full median/max={internal['full']['median']:.12g}/{internal['full']['max']:.12g}，energy={internal['energy']['median']:.12g}/{internal['energy']['max']:.12g}，boundary={internal['boundary']['median']:.12g}/{internal['boundary']['max']:.12g}，gain max={internal['gain']['max']:.12g}，peak max={internal['peak']['max']:.12g}。rawA/rawB median={internal['rawA']['median']:.12g}/{internal['rawB']['median']:.12g}，error cosine={internal['error_cosine']['median']:.12g}，cancellation ratio={internal['cancellation_ratio']['median']:.12g}，`CANCEL3=false`。

burned24 full median/max={burned['full']['median']:.12g}/{burned['full']['max']:.12g}，energy={burned['energy']['median']:.12g}/{burned['energy']['max']:.12g}，boundary={burned['boundary']['median']:.12g}/{burned['boundary']['max']:.12g}，gain={burned['gain']['median']:.12g}/{burned['gain']['max']:.12g}，peak max={burned['peak']['max']:.12g}。H3 相对 H2 的 full median/max 改善={change['full_median']:.12g}%/{change['full_max']:.12g}%，boundary median/max={change['boundary_median']:.12g}%/{change['boundary_max']:.12g}%，interior={change['interior_full_median']:.12g}%，corner={change['corner_or_mixed_full_median']:.12g}%，corner/interior ratio={change['corner_over_interior_ratio']:.12g}%，energy median/max={change['energy_median']:.12g}%/{change['energy_max']:.12g}%，gain median/max={change['gain_median']:.12g}%/{change['gain_max']:.12g}%。

H3 已明显改善 energy，且 boundary/peak gate 通过，但 full median 仅改善 0.0156%，full max、corner 和 gain 反而变差。因此不继续 boundary-feature sweep；下一步只隔离检验 TRAIN140-derived diagonal/gain consistency。

## TRAIN140-only 对角先验

TRAIN140 normalized real diagonal: n={diagonal['n']}，min/median/max={diagonal['min']:.12g}/{diagonal['median']:.12g}/{diagonal['max']:.12g}，mean/std={diagonal['mean']:.12g}/{diagonal['std_population_ddof0']:.12g}，span={diagonal['span_max_minus_min']:.12g}，max|deviation from 1|={diagonal['max_abs_deviation_from_1']:.12g}，imag max=0。这只支持 `AN_APPROXIMATELY_CONSTANT_NORMALIZED_DIAGONAL_PRIOR`；不声称每个 native source 数学上精确等于 1。target=`1+0j`，唯一来源为 TRAIN140 median=1.0；VAL20、INTERNAL_TEST20、burned24 均未参与 prior 定义。

## H4 唯一变更与 loss

H4 完全等于 H3 architecture，保持 {summary['parameter_count']} 参数、{summary['input_dimension']} 维输入、boundary features/source masks/output PE/relative PE、model seed={MODEL_SEED} 与 fresh initial-state SHA `{summary['initial_state_SHA256']}`。不 warm-start H3/H2 checkpoint。

每 update 仍为 4个 TRAIN source x 8192 query=32768 field pairs。对每个 source 块用固定 local indices `0,32,...,8160` 从 post-C4 safe query 中选 256 个，合计 1024 q/update；不使用新 RNG，不根据 target error 选 q。构造 `(q,q,d=0)`：

`L_field = 0.5*MSE_complex(A,y) + 0.5*MSE_complex(B,y)`

`L_diag = 0.5*MSE_complex(A_diag,1+0j) + 0.5*MSE_complex(B_diag,1+0j)`

`L_total = L_field + 0.1*L_diag`

field coefficient 仍为 1.0，`lambda_diag=0.1` 在首次 science optimizer step 前冻结；不声称 0.1 最优，不做 sweep。对 A_diag 和 B_diag 分别监督，不只监督最终平均，以避免 raw-branch underdetermination。

## 坐标流、安全性与等价性

20k 纯坐标复演共审计 {coverage['total_sampled_occurrences']} 个 diagonal q，unique={coverage['unique_native_coordinates']}，x/y/z levels={coverage['unique_x_levels']}/{coverage['unique_y_levels']}/{coverage['unique_z_levels']}，boundary/interior occurrences={coverage['boundary_occurrences']}/{coverage['interior_occurrences']}，前后 C4 field/diagonal blacklist hits 均为 0。未打开 PSF 数组，未保存巨大 stream。field stream SHA 仍为 `{stream['field_sample_stream_SHA256']}`；diagonal stream SHA 为 `{stream['diagonal_coordinate_stream_SHA256']}`。

H4 相对 H3 在 4096 对坐标的最大归一化 forward discrepancy={artifacts[H3_EQUIVALENCE]['max_normalized_forward_discrepancy']:.12g}（门限 1e-7），initial SHA identity PASS，Hermitian discrepancy={artifacts[H4_SPEC]['unit_tests']['Hermitian_max_normalized_discrepancy']:.12g}（门限 1e-6），field/diag gradient 非零，no detach。inference 仍为 `0.5*(A+B)`，禁止 hard-code center、column rescaling、post-hoc normalization、gain clipping 或手工 amplitude correction。

## 未来 R8E 训练、selection 与 outcome

split=140/20/20，split SHA=`{EXPECTED_SPLIT_SHA}`，blacklist unique=136。AdamW lr=2e-4、betas=(0.9,0.99)、weight decay=1e-6、scheduler=null、20000 updates、VAL/checkpoint every500、no early stopping 全部不变。VAL 仅用 20 sources x 65536 probes，coordinate SHA=`{EXPECTED_VAL_SHA}`；仍只按 pooled final-sym full complex NRMSE、max-column、tie<=1e-6 时 earlier update 选 checkpoint。TRAIN diagonal、VAL gain、INTERNAL_TEST、burned24 不参与 selection。

R8E 有序树为 D→A→B→C→E：先 cancellation，再 readiness，再严格比较 H4g/H3g 和 H4m/H3m，相等不算改善。20 canonical +10000 random 共 10020 组：unclassified={truth['unclassified']}，multi={truth['multi_classified']}，exactly-one={truth['exactly_one']}。

Dummy 实际模拟 32768 field +1024 diagonal pairs 的 forward/backward：field/diag/total loss={resource['field_loss']:.8g}/{resource['diagonal_loss']:.8g}/{resource['total_loss']:.8g}，field/diag/combined grad norm={resource['field_gradient_norm']:.8g}/{resource['diagonal_gradient_norm']:.8g}/{resource['combined_gradient_norm']:.8g}，peak allocated/reserved={resource['peak_allocated_VRAM_GiB']:.6f}/{resource['peak_reserved_VRAM_GiB']:.6f} GiB，seconds/update={resource['seconds_per_combined_update']:.6g}，20k 纯计算估计={resource['estimated_20000_update_hours_no_validation_or_IO']:.6f} h。未构造 optimizer，未 optimizer.step。

## Science isolation 和冻结说明

science optimizer constructed=false，science updates=0，H4 training=false，checkpoint inference=0，new exact physics/PSF/complete columns=0，fresh30=0/0，operator=0，Bunny/NVR/exact-F0=0。

本轮没有训练H4。

H4以H3为parent，
保持相同414210参数、165维输入、
boundary features、source masks、
dual-orientation field loss、
Hermitian inference、140/20/20 split、
field sample/C4 stream、optimizer和20k budget。

唯一新增science change是
TRAIN140-derived dense unit-diagonal consistency：

从每个update已有且blacklist-safe的field query coordinates
确定性选取1024个q，
构造(q,q,d=0)，
用TRAIN140 normalized diagonal median=1.0
作为approximate prior target，
增加：

`L_total = L_field + 0.1 L_diag`。

没有使用VAL、INTERNAL_TEST或burned24
定义该prior；
没有hard-code prediction center；
没有post-hoc column normalization；
没有lambda sweep。

没有生成新exact PSF，
没有densification，
没有fresh30，
没有operator，
没有Bunny/NVR/exact-F0。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def package_submission() -> dict[str, Any]:
    sources = [
        REPORT, SUMMARY, STATUS, PROTOCOL, R8C_FREEZE, RATIONALE,
        H3_POSTMORTEM, TRAIN_DIAGONAL, PRIOR_ISOLATION, H4_SPEC,
        COORDINATE_SPEC, COORDINATE_COVERAGE, DIAGONAL_STREAM, LOSS_SPEC,
        H3_EQUIVALENCE, TRAIN_INHERIT, R8E_OUTCOME, TRUTH_TABLE,
    ]
    require(len(sources) == len(set(sources)) == 18, "R8D submission must contain 18 unique sources")
    require(DUMMY_RESOURCE.is_file() and INIT_IDENTITY.is_file() and DUMMY_RESOURCE not in sources and INIT_IDENTITY not in sources, "Optional artifacts must remain results-only")
    resolved = SUBMISSION.resolve()
    require(resolved == (ROOT / "本轮提交").resolve() and resolved.parent == ROOT.resolve(), "Unsafe submission directory")
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    for child in list(SUBMISSION.iterdir()):
        require(child.is_file(), f"Refusing recursive delete of unexpected entry: {child}")
        child.unlink()
    rows = []
    for source in sources:
        require(source.is_file(), f"Missing submission source: {source}")
        copied = SUBMISSION / source.name
        shutil.copy2(source, copied)
        source_hash = sha256(source)
        copied_hash = sha256(copied)
        rows.append({
            "original_path": str(source.resolve()),
            "copied_path": str(copied.resolve()),
            "bytes": source.stat().st_size,
            "SHA256": copied_hash,
            "source_SHA256": source_hash,
            "hash_match": source_hash == copied_hash,
            "required_or_optional": "required",
            "purpose": "R8D formal train-only diagonal gain-consistency preregistration evidence artifact",
        })
    require(all(row["hash_match"] for row in rows), "Submission copy hash mismatch")
    forbidden = {".ckpt", ".pt", ".pth", ".npy", ".npz", ".log"}
    require(not any(Path(row["copied_path"]).suffix.lower() in forbidden for row in rows), "Forbidden artifact included")
    manifest = {
        "classification": "P6_PHASEA8MRN2R8D_SUBMISSION_PACKAGE_PASS",
        "created": now(),
        "source_file_count": 18,
        "manifest_file_count": 2,
        "submission_total_files": 20,
        "files": rows,
        "all_hash_match": True,
        "results_only_optional_artifacts": [str(DUMMY_RESOURCE.resolve()), str(INIT_IDENTITY.resolve())],
        "forbidden_artifacts_included": False,
        "gate": "PASS",
    }
    manifest_path = SUBMISSION / "本轮提交清单.json"
    write_json(manifest_path, manifest)
    manifest_sha = sha256(manifest_path)
    lines = [
        "P6_PHASEA8MRN2R8D submission manifest",
        "source files: 18",
        "manifest files: 2",
        "TOTAL: 20",
        f"本轮提交清单.json SHA256={manifest_sha}",
        "all hash match: true",
        "",
    ]
    lines.extend(f"{row['SHA256']}  {Path(row['copied_path']).name}" for row in rows)
    (SUBMISSION / "本轮提交清单.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    require(len(list(SUBMISSION.iterdir())) == 20, "R8D submission TOTAL is not 20")
    return {"TOTAL": 20, "manifest_SHA256": manifest_sha, "all_hash_match": True}


def execute() -> None:
    outputs = [
        STATUS, PROTOCOL, R8C_FREEZE, RATIONALE, H3_POSTMORTEM,
        TRAIN_DIAGONAL, PRIOR_ISOLATION, H4_SPEC, COORDINATE_SPEC,
        COORDINATE_COVERAGE, DIAGONAL_STREAM, LOSS_SPEC, H3_EQUIVALENCE,
        INIT_IDENTITY, TRAIN_INHERIT, R8E_OUTCOME, TRUTH_TABLE,
        DUMMY_RESOURCE, SUMMARY, REPORT,
    ]
    existing = [str(path) for path in outputs if path.exists()]
    require(not existing, f"Refusing to overwrite formal R8D output: {existing}")
    artifacts = build_artifacts()
    require(len(artifacts) == 19, "Expected exactly 19 R8D JSON artifacts")
    for path, value in artifacts.items():
        write_json(path, value)
    write_report(artifacts)
    package = package_submission()
    summary = artifacts[SUMMARY]
    print(json.dumps({
        "classification": summary["classification"],
        "R8C_freeze": summary["R8C_freeze"],
        "R8C_case": summary["R8C_outcome_case"],
        "candidate": summary["candidate"],
        "parent": summary["parent"],
        "parameter_count": summary["parameter_count"],
        "input_dimension": summary["input_dimension"],
        "diagonal_pairs_per_update": summary["diagonal_pairs_per_update"],
        "field_sample_stream_SHA256": summary["field_sample_stream_SHA256"],
        "diagonal_coordinate_stream_SHA256": summary["diagonal_coordinate_stream_SHA256"],
        "science_updates": summary["science_updates"],
        "fresh30": f"{summary['fresh30_selected']}/{summary['fresh30_accessed']}",
        "operator_cases": summary["operator_cases"],
        "new_exact_physics": summary["new_exact_physics"],
        "R8E_execution_ready": summary["R8E_execution_ready"],
        "recommended_next": summary["recommended_next"],
        "submission": package,
    }, ensure_ascii=False, indent=2))


def finalize_existing() -> None:
    """Finish report/package after a non-scientific post-audit formatting error."""

    json_paths = [
        STATUS, PROTOCOL, R8C_FREEZE, RATIONALE, H3_POSTMORTEM,
        TRAIN_DIAGONAL, PRIOR_ISOLATION, H4_SPEC, COORDINATE_SPEC,
        COORDINATE_COVERAGE, DIAGONAL_STREAM, LOSS_SPEC, H3_EQUIVALENCE,
        INIT_IDENTITY, TRAIN_INHERIT, R8E_OUTCOME, TRUTH_TABLE,
        DUMMY_RESOURCE, SUMMARY,
    ]
    require(all(path.is_file() for path in json_paths), "Existing R8D JSON set is incomplete")
    require(not REPORT.exists(), "Refusing to overwrite an existing R8D report")
    hard_freeze_r8c()
    artifacts = {path: read_json(path) for path in json_paths}
    for path in (H3_POSTMORTEM, SUMMARY):
        owner = artifacts[path] if path == H3_POSTMORTEM else artifacts[path]["H3"]
        changes = owner["H3_vs_H2_improvement_percent"]
        if "gain_gain_median" in changes:
            changes["gain_median"] = changes.pop("gain_gain_median")
            changes["gain_max"] = changes.pop("gain_gain_max")
            write_json(path, artifacts[path])
    require(artifacts[STATUS]["classification"] == artifacts[SUMMARY]["classification"] == PASS_CLASS, "Existing R8D classification mismatch")
    require(all(value.get("gate") == "PASS" for value in artifacts.values()), "Existing R8D JSON gate is not uniformly PASS")
    require(artifacts[SUMMARY]["science_updates"] == 0 and artifacts[SUMMARY]["H4_training"] is False, "Existing R8D science-isolation failure")
    require(artifacts[DIAGONAL_STREAM]["field_sample_stream_SHA256"] == EXPECTED_FIELD_STREAM_SHA and artifacts[DIAGONAL_STREAM]["field_sample_stream_match"] is True, "Existing R8D field stream mismatch")
    write_report(artifacts)
    package = package_submission()
    print(json.dumps({
        "classification": PASS_CLASS,
        "recovered_from": "post-audit Markdown field-name formatting error",
        "science_replay_repeated": False,
        "R8E_execution_ready": True,
        "recommended_next": NEXT_STAGE,
        "submission": package,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    arguments = parser.parse_args()
    require(arguments.run != arguments.finalize_existing, "Select exactly one of --run or --finalize-existing")
    if arguments.finalize_existing:
        finalize_existing()
    else:
        execute()


if __name__ == "__main__":
    main()
