"""Formal P6 Phase-A8MRN2R8E controlled H4 development run.

H4 is architecturally identical to H3.  Its sole science change is the
R8D-frozen TRAIN140-derived dense approximate unit-diagonal consistency loss.
The worker reuses the frozen field sampler, metrics, validation, split, and
access-order implementations and preserves one resumable formal lineage.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

import p6_phaseA8MRN2_development as n2
import p6_phaseA8MRN2R1_diagnostic as r1
import p6_phaseA8MRN2R3_H0_development as r3
import p6_phaseA8MRN2R8D_preregister as r8d
from p6_a8mrn1_direct_implicit_models import parameter_count
from p6_phaseA8MRN2R2_preregister import complex_mse, state_sha256
from p6_phaseA8MRN2R8B_boundary_aware_source_model import (
    BOUNDARY_AUXILIARY_GLOBAL_INDICES,
    BOUNDARY_FEATURE_NAMES,
    build_h3,
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
SUBMISSION = ROOT / "本轮提交"
SCRIPT = Path(__file__).resolve()
MONITOR = ROOT / "scripts/monitor_p6_phaseA8MRN2R8E.ps1"
H4_SOURCE = ROOT / "scripts/p6_phaseA8MRN2R8D_diagonal_gain_consistency_model.py"
H3_SOURCE = ROOT / "scripts/p6_phaseA8MRN2R8B_boundary_aware_source_model.py"
H2_SOURCE = ROOT / "scripts/p6_phaseA8MRN2R6_anisotropic_z_source_model.py"
MODEL_SOURCE = ROOT / "scripts/p6_a8mrn1_direct_implicit_models.py"
R3_SOURCE = ROOT / "scripts/p6_phaseA8MRN2R3_H0_development.py"
R8D_SOURCE = ROOT / "scripts/p6_phaseA8MRN2R8D_preregister.py"
N2_SOURCE = ROOT / "scripts/p6_phaseA8MRN2_development.py"
R1_SOURCE = ROOT / "scripts/p6_phaseA8MRN2R1_diagnostic.py"
RUNTIME_HELPER = ROOT / "scripts/p6_a8mrn2_frozen_helpers.py"
METRIC_HELPER = ROOT / "scripts/p6_a8m1_validation.py"
R50_REFERENCE = ROOT / "scripts/p6_phaseA8MR_reassess.py"
SPLIT_PATH = n2.N1R_SPLIT
BLACKLIST_PATH = n2.N1R_BLACKLIST
REPORT = ROOT / "docs/P6_PHASEA8MRN2R8E_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY_DEVELOPMENT.md"

PREFIX = "p6_phaseA8MRN2R8E_"
STATUS = OUT / f"{PREFIX}status.json"
PROTOCOL_EXEC = OUT / f"{PREFIX}protocol_execution_record.json"
INPUT_FREEZE = OUT / f"{PREFIX}input_freeze_manifest.json"
EXECUTION_FREEZE = OUT / f"{PREFIX}execution_freeze_manifest.json"
MODEL_RUNTIME_AUDIT = OUT / f"{PREFIX}H4_model_runtime_audit.json"
DIAGONAL_RUNTIME_AUDIT = OUT / f"{PREFIX}diagonal_prior_runtime_audit.json"
FIELD_STREAM_AUDIT = OUT / f"{PREFIX}field_sample_stream_audit.json"
DIAGONAL_STREAM_AUDIT = OUT / f"{PREFIX}diagonal_stream_audit.json"
TRAINING_HISTORY = OUT / f"{PREFIX}training_history.json"
BEST_FREEZE = OUT / f"{PREFIX}H4_best_checkpoint_freeze.json"
INTERNAL_METRICS = OUT / f"{PREFIX}INTERNAL_TEST_metrics.json"
INTERNAL_RAW = OUT / f"{PREFIX}INTERNAL_TEST_raw_branch_metrics.json"
CANCELLATION = OUT / f"{PREFIX}cancellation_retest.json"
BURNED_METRICS = OUT / f"{PREFIX}burned24_metrics.json"
BURNED_RAW = OUT / f"{PREFIX}burned24_raw_branch_metrics.json"
BURNED_STRATA = OUT / f"{PREFIX}burned24_strata.json"
H4_H3_COMPARISON = OUT / f"{PREFIX}H4_vs_H3_controlled_comparison.json"
GAIN_COMPARISON = OUT / f"{PREFIX}gain_controlled_comparison.json"
DIAGONAL_SELF = OUT / f"{PREFIX}diagonal_self_consistency.json"
READINESS_OUTCOME = OUT / f"{PREFIX}development_readiness_and_outcome.json"
RESOURCE = OUT / f"{PREFIX}resource_runtime.json"
ISOLATION = OUT / f"{PREFIX}isolation_and_access_order_audit.json"
SUMMARY = OUT / f"{PREFIX}summary.json"
STDOUT_LOG = OUT / f"{PREFIX}stdout.log"
STDERR_LOG = OUT / f"{PREFIX}stderr.log"
WORK_DIR = OUT / "phaseA8MRN2R8E_work"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"

R8D_REPORT = ROOT / "docs/P6_PHASEA8MRN2R8D_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY_PREREGISTRATION.md"
R8D_NAMES = [
    "p6_phaseA8MRN2R8D_status.json", "p6_phaseA8MRN2R8D_summary.json",
    "p6_phaseA8MRN2R8D_protocol.json", "p6_phaseA8MRN2R8D_R8C_freeze_manifest.json",
    "p6_phaseA8MRN2R8D_scientific_rationale.json", "p6_phaseA8MRN2R8D_H3_mechanism_postmortem.json",
    "p6_phaseA8MRN2R8D_TRAIN140_diagonal_prior_freeze.json", "p6_phaseA8MRN2R8D_prior_isolation_audit.json",
    "p6_phaseA8MRN2R8D_H4_model_spec.json", "p6_phaseA8MRN2R8D_diagonal_coordinate_spec.json",
    "p6_phaseA8MRN2R8D_diagonal_coordinate_coverage_audit.json", "p6_phaseA8MRN2R8D_diagonal_stream_freeze.json",
    "p6_phaseA8MRN2R8D_loss_spec.json", "p6_phaseA8MRN2R8D_H3_equivalence_audit.json",
    "p6_phaseA8MRN2R8D_training_inheritance_manifest.json", "p6_phaseA8MRN2R8D_R8E_outcome_protocol.json",
    "p6_phaseA8MRN2R8D_outcome_truth_table_audit.json",
]
R8D_PROTOCOL = OUT / "p6_phaseA8MRN2R8D_protocol.json"
R8D_LOSS_SPEC = OUT / "p6_phaseA8MRN2R8D_loss_spec.json"
R8D_COORDINATE_SPEC = OUT / "p6_phaseA8MRN2R8D_diagonal_coordinate_spec.json"
R8D_COVERAGE = OUT / "p6_phaseA8MRN2R8D_diagonal_coordinate_coverage_audit.json"
R8D_STREAM = OUT / "p6_phaseA8MRN2R8D_diagonal_stream_freeze.json"
R8D_PRIOR = OUT / "p6_phaseA8MRN2R8D_TRAIN140_diagonal_prior_freeze.json"
R8D_OUTCOME = OUT / "p6_phaseA8MRN2R8D_R8E_outcome_protocol.json"
R8D_PACKAGE_MANIFEST = SUBMISSION / "本轮提交清单.json"
R8C_SUMMARY = OUT / "p6_phaseA8MRN2R8C_summary.json"

EXPECTED_R8D_CLASS = "P6_PHASEA8MRN2R8D_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY_PREREGISTERED"
CANDIDATE = "H4_H3_TRAIN140_DENSE_UNIT_DIAGONAL_CONSISTENCY_LAMBDA0P1"
EXPECTED_H4_SOURCE_SHA = "a05996652c19b30d084d92b57942fdbbfe44abef93987a42f8517346dba79c53"
EXPECTED_H3_SOURCE_SHA = "896efe8b7bc4e73a93450e5e6978efdfc46bbc50482bdc2d4ae17aceb8f7c2f2"
EXPECTED_H2_SOURCE_SHA = "1dd370573b6c81db0c0bef6d30aa4921cea57195a4cbf241b42f3623375acf7e"
EXPECTED_INITIAL_SHA = "7c4601d5c779655bc2a69482995868aa501310f5de3d1a12470708202424bb28"
EXPECTED_SPLIT_SHA = "29900f4738d26f36893d3ebb5aae795b32d5836888a267c4bee98d440372a9ac"
EXPECTED_BLACKLIST_SHA = "0a733fd0328cd30377a4faf1a78432698da8a96f9c41476e6d233a39dc24754f"
EXPECTED_VAL_SHA = "718c4152461367368be80e02408c18667da09b18dc5550327337ccc06c8fb24b"
EXPECTED_FIELD_STREAM_SHA = "6639bcac3bb9c5efce5d43c5fb5e784e8f8a67a4c99cc75559c967369e57a7e2"
EXPECTED_DIAGONAL_STREAM_SHA = "395bd3a0e85405db340aab23a0fce4f334b95bf6750b029bbb307b399db526ce"
EXPECTED_PARAMS = 414210
MODEL_SEED = 20260909
MAX_UPDATES = 20000
VAL_INTERVAL = 500
G_REF = 25912.966796875

H2_GAIN_MEDIAN = 0.04118274262184512
H2_GAIN_MAX = 0.14848258760888638
H3_BEST_VAL_POOLED = 0.05320869817865062
H3_BEST_VAL_MAX = 0.07868135918409411
H3_FULL_MEDIAN = 0.06660179578093173
H3_FULL_MAX = 0.12139477236343238
H3_ENERGY_MEDIAN = 0.01788038221192561
H3_ENERGY_MAX = 0.05499024150929448
H3_BOUNDARY_MEDIAN = 0.06634186330782878
H3_BOUNDARY_MAX = 0.12631352870541893
H3_GAIN_MEDIAN = 0.07052240806570562
H3_GAIN_MAX = 0.18721155057525807
H3_PEAK_MAX = 0.0

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

FINAL_OUTPUTS = [
    STATUS, PROTOCOL_EXEC, INPUT_FREEZE, EXECUTION_FREEZE, MODEL_RUNTIME_AUDIT,
    DIAGONAL_RUNTIME_AUDIT, FIELD_STREAM_AUDIT, DIAGONAL_STREAM_AUDIT, TRAINING_HISTORY, BEST_FREEZE,
    INTERNAL_METRICS, INTERNAL_RAW, CANCELLATION, BURNED_METRICS, BURNED_RAW,
    BURNED_STRATA, H4_H3_COMPARISON, GAIN_COMPARISON, DIAGONAL_SELF,
    READINESS_OUTCOME, RESOURCE, ISOLATION, SUMMARY, REPORT,
]


def configure_reused_runtime() -> None:
    """Point reused R3/N2 helpers at the independent R8E lineage."""

    mapping = {
        "SCRIPT": SCRIPT, "MONITOR": MONITOR, "MODEL_SOURCE": MODEL_SOURCE,
        "H0_SOURCE": H4_SOURCE, "REPORT": REPORT, "STATUS": STATUS,
        "PROTOCOL_EXEC": PROTOCOL_EXEC, "INPUT_FREEZE": INPUT_FREEZE,
        "EXECUTION_FREEZE": EXECUTION_FREEZE, "INIT_AUDIT": MODEL_RUNTIME_AUDIT,
        "STREAM_AUDIT": FIELD_STREAM_AUDIT, "COVERAGE": DIAGONAL_STREAM_AUDIT,
        "TRAINING_HISTORY": TRAINING_HISTORY, "BEST_FREEZE": BEST_FREEZE,
        "INTERNAL_METRICS": INTERNAL_METRICS, "INTERNAL_RAW": INTERNAL_RAW,
        "CANCELLATION": CANCELLATION, "BURNED_METRICS": BURNED_METRICS,
        "BURNED_RAW": BURNED_RAW, "BURNED_STRATA": BURNED_STRATA,
        "BASELINE": H4_H3_COMPARISON, "READINESS_OUTCOME": READINESS_OUTCOME,
        "RESOURCE": RESOURCE, "ISOLATION": ISOLATION, "SUMMARY": SUMMARY,
        "STDOUT_LOG": STDOUT_LOG, "STDERR_LOG": STDERR_LOG,
        "WORK_DIR": WORK_DIR, "CHECKPOINT_DIR": CHECKPOINT_DIR,
    }
    for name, value in mapping.items():
        setattr(r3, name, value)
    n2.STATUS = STATUS
    n2.PROTOCOL_EXEC = PROTOCOL_EXEC
    n2.STDERR_LOG = STDERR_LOG


def commandline(resume: bool = False) -> str:
    suffix = " --resume" if resume else ""
    return f'"{sys.executable}" "{SCRIPT.resolve()}" --run --device cuda{suffix}'


def hard_precheck() -> tuple[dict[str, Any], dict[str, Any]]:
    paths = [R8D_REPORT] + [OUT / name for name in R8D_NAMES]
    r3.require(len(paths) == len(set(paths)) == 18, "R8D authority must be report plus 17 JSON artifacts")
    r3.require(all(path.is_file() for path in paths), "R8D authoritative artifact missing")
    r3.require(H4_SOURCE.is_file() and r3.sha256(H4_SOURCE) == EXPECTED_H4_SOURCE_SHA, "H4 authoritative source SHA mismatch")
    r3.require(H3_SOURCE.is_file() and r3.sha256(H3_SOURCE) == EXPECTED_H3_SOURCE_SHA, "H3 parent source SHA mismatch")

    status = r3.read_json(OUT / "p6_phaseA8MRN2R8D_status.json")
    summary = r3.read_json(OUT / "p6_phaseA8MRN2R8D_summary.json")
    r3.require(status["status"] == status["phase"] == "FINALIZED", "R8D is not FINALIZED")
    r3.require(status["classification"] == summary["classification"] == EXPECTED_R8D_CLASS, "R8D classification mismatch")
    r3.require(status["R8E_execution_ready"] is True and summary["R8E_execution_ready"] is True, "R8D did not release R8E")
    r3.require(status["R8C_case"] == summary["R8C_outcome_case"] == "B", "R8C CASE B was not preserved")
    r3.require(status["CANCEL3"] is False and status["READY3"] is False, "R8C cancellation/readiness state drifted")
    for key in ("science_optimizer_updates", "science_updates", "checkpoint_inference", "new_exact_physics", "fresh30_selected", "fresh30_accessed", "operator_cases"):
        r3.require(status[key] == 0, f"R8D isolation mismatch: {key}")
    r3.require(status["H4_training"] is False and status["science_optimizer_constructed"] is False, "H4 was already trained or optimizer was constructed")
    r3.require(summary["candidate"] == CANDIDATE and summary["parent"] == "H3", "R8D H4 identity mismatch")
    r3.require(summary["parameter_count"] == EXPECTED_PARAMS and summary["input_dimension"] == 165, "R8D H4 size mismatch")
    r3.require(summary["initial_state_SHA256"] == EXPECTED_INITIAL_SHA, "R8D initial-state mismatch")
    r3.require(summary["field_sample_stream_SHA256"] == EXPECTED_FIELD_STREAM_SHA, "R8D field stream mismatch")
    r3.require(summary["diagonal_coordinate_stream_SHA256"] == EXPECTED_DIAGONAL_STREAM_SHA, "R8D diagonal stream mismatch")

    package = r3.read_json(R8D_PACKAGE_MANIFEST)
    r3.require(package["classification"] == "P6_PHASEA8MRN2R8D_SUBMISSION_PACKAGE_PASS", "Current package is not R8D")
    r3.require(package["all_hash_match"] is True and package["source_file_count"] == 18 and package["submission_total_files"] == 20, "R8D package manifest invalid")
    packaged = {Path(row["original_path"]).resolve(): row["source_SHA256"] for row in package["files"]}
    r3.require(set(packaged) == {path.resolve() for path in paths}, "R8D packaged source set mismatch")
    for path in paths:
        r3.require(packaged[path.resolve()] == r3.sha256(path), f"R8D packaged artifact changed: {path}")

    rows = [{
        "authority": "R8D",
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "SHA256": r3.sha256(path),
        "packaged_in_R8D_submission": True,
        "read_only_after_freeze": True,
    } for path in paths]
    rows.extend([
        {
            "authority": "R8D_H4_FROZEN_SOURCE", "path": str(H4_SOURCE.resolve()),
            "bytes": H4_SOURCE.stat().st_size, "SHA256": r3.sha256(H4_SOURCE),
            "expected_SHA256": EXPECTED_H4_SOURCE_SHA, "read_only_after_freeze": True,
        },
        {
            "authority": "H3_PARENT_FROZEN_SOURCE", "path": str(H3_SOURCE.resolve()),
            "bytes": H3_SOURCE.stat().st_size, "SHA256": r3.sha256(H3_SOURCE),
            "expected_SHA256": EXPECTED_H3_SOURCE_SHA, "read_only_after_freeze": True,
        },
    ])
    manifest = {
        "classification": "P6_PHASEA8MRN2R8E_R8D_INPUT_HARD_FREEZE_PASS",
        "created": r3.now(), "R8D_precheck": "PASS",
        "R8D_terminal_classification": status["classification"],
        "R8E_execution_ready": True, "R8C_outcome_case": "B",
        "CANCEL3": False, "READY3": False,
        "R8D_artifact_count": 18, "R8D_packaged_artifact_count": 18,
        "H4_source_SHA256": r3.sha256(H4_SOURCE), "H4_source_SHA_match": True,
        "H3_source_SHA256": r3.sha256(H3_SOURCE), "H3_source_SHA_match": True,
        "science_optimizer_updates_before_R8E": 0,
        "H4_previous_science_training": False, "checkpoint_inference": 0,
        "fresh30_selected": 0, "fresh30_accessed": 0, "operator_cases": 0,
        "new_exact_physics": 0, "artifacts": rows,
        "science_identity": {
            "candidate": CANDIDATE, "parent": "H3",
            "parameter_count": EXPECTED_PARAMS, "input_dimension": 165,
            "model_seed": MODEL_SEED, "initial_state_SHA256": EXPECTED_INITIAL_SHA,
            "boundary_features": list(BOUNDARY_FEATURE_NAMES),
            "boundary_slots": list(BOUNDARY_AUXILIARY_GLOBAL_INDICES),
            "source_z_k2_zero": True, "source_k4_to_k7_zero": True,
            "output_absolute_k": list(range(8)), "relative_displacement_k": list(range(10)),
            "split": [140, 20, 20], "split_SHA256": EXPECTED_SPLIT_SHA,
            "blacklist_unique": 136, "blacklist_SHA256": EXPECTED_BLACKLIST_SHA,
            "field_stream_SHA256": EXPECTED_FIELD_STREAM_SHA,
            "diagonal_stream_SHA256": EXPECTED_DIAGONAL_STREAM_SHA,
            "lambda_diag": LAMBDA_DIAG, "g_ref": G_REF,
        },
        "gate": "PASS",
    }
    return manifest, summary


def model_and_resource_preflight(device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    r3.require(device.type == "cuda", "R8E formal preflight requires CUDA")
    unit, equivalence, initialization = r8d.model_audits()
    resource = r8d.dummy_resource_test()
    r3.require(unit["gate"] == equivalence["gate"] == initialization["gate"] == resource["gate"] == "PASS", "R8D-derived H4 preflight failed")
    r3.require(equivalence["random_coordinate_pairs"] >= 4096 and equivalence["max_normalized_forward_discrepancy"] <= 1e-7, "H3 equivalence gate failed")
    r3.require(initialization["H4_fresh_initial_state_SHA256"] == EXPECTED_INITIAL_SHA, "H4 initial-state SHA mismatch")
    r3.require(resource["field_pairs"] == FIELD_PAIRS_PER_UPDATE and resource["diagonal_pairs"] == DIAGONAL_PAIRS_PER_UPDATE, "H4 resource preflight batch mismatch")
    r3.require(resource["optimizer_step_executed"] is False and resource["peak_reserved_VRAM_GiB"] <= 7.5, "H4 resource preflight gate failed")

    prior = r3.read_json(R8D_PRIOR)
    coordinate = r3.read_json(R8D_COORDINATE_SPEC)
    coverage = r3.read_json(R8D_COVERAGE)
    stream = r3.read_json(R8D_STREAM)
    loss = r3.read_json(R8D_LOSS_SPEC)
    expected_diag = {
        "n": 140, "min": 0.9979013282065966, "median": 1.0,
        "max": 1.0039199727995733, "mean": 1.0003041404061261,
        "std_population_ddof0": 0.001344764406703845,
        "span_max_minus_min": 0.006018644592976741,
        "max_abs_deviation_from_1": 0.003919972799573346,
    }
    r3.require(prior["TRAIN140_normalized_real_diagonal"] == expected_diag and prior["TRAIN140_imag_diagonal_max_abs"] == 0.0, "TRAIN140 prior drifted")
    r3.require(prior["VAL20_values_used"] is False and prior["INTERNAL_TEST20_values_used"] is False and prior["burned24_values_used"] is False, "Held-out diagonal leaked into H4 prior")
    r3.require(coordinate["local_indices"] == list(DIAGONAL_LOCAL_INDICES) and coordinate["diagonal_pairs_per_update"] == DIAGONAL_PAIRS_PER_UPDATE, "Diagonal coordinate rule drifted")
    r3.require(coverage["total_sampled_occurrences"] == 20480000 and coverage["unique_native_coordinates"] == 2698300, "R8D diagonal coverage drifted")
    r3.require(coverage["unique_x_levels"] == 150 and coverage["unique_y_levels"] == 150 and coverage["unique_z_levels"] == 120, "R8D diagonal axis coverage drifted")
    r3.require(stream["field_sample_stream_SHA256"] == EXPECTED_FIELD_STREAM_SHA and stream["diagonal_coordinate_stream_SHA256"] == EXPECTED_DIAGONAL_STREAM_SHA, "R8D stream freeze drifted")
    r3.require(loss["lambda_diag"] == LAMBDA_DIAG and loss["L_total"] == "L_field+0.1*L_diag", "R8D loss algebra drifted")

    weighted_diag_grad = LAMBDA_DIAG * resource["diagonal_gradient_norm"]
    weighted_ratio = weighted_diag_grad / resource["field_gradient_norm"]
    model_audit = {
        "classification": "P6_PHASEA8MRN2R8E_H4_MODEL_RUNTIME_AUDIT_PASS",
        "created": r3.now(), "candidate": CANDIDATE, "parent": "H3",
        "H4_source_path": str(H4_SOURCE.resolve()), "H4_source_SHA256": r3.sha256(H4_SOURCE),
        "H4_source_SHA_match": True, "H3_source_SHA_match": r3.sha256(H3_SOURCE) == EXPECTED_H3_SOURCE_SHA,
        "parameter_count": EXPECTED_PARAMS, "input_dimension": 165,
        "model_seed": MODEL_SEED, "initial_state_SHA256": initialization["H4_fresh_initial_state_SHA256"],
        "initial_state_SHA_match": True, "fresh_initialization": True, "warm_start": False,
        "H3_equivalence_pairs": equivalence["random_coordinate_pairs"],
        "H3_forward_bitwise_equal": equivalence["H3_forward_bitwise_equal"],
        "H3_max_normalized_discrepancy": equivalence["max_normalized_forward_discrepancy"],
        "H3_equivalence_threshold_max": 1e-7,
        "boundary_features": list(BOUNDARY_FEATURE_NAMES),
        "boundary_slots": list(BOUNDARY_AUXILIARY_GLOBAL_INDICES),
        "source_z_k2_zero": True, "source_k4_to_k7_zero": True,
        "output_k0_to_k7_unchanged": True, "relative_k0_to_k9_unchanged": True,
        "Hermitian_max_normalized_discrepancy": unit["Hermitian_max_normalized_discrepancy"],
        "Hermitian_threshold_max": 1e-6,
        "field_loss": loss["L_field"], "diagonal_loss": loss["L_diag"],
        "total_loss": loss["L_total"], "lambda_diag": LAMBDA_DIAG,
        "final_inference": "0.5*(A+B)", "new_parameters": 0, "new_inputs": 0,
        "science_optimizer_constructed_at_audit": False,
        "science_optimizer_updates_at_audit": 0,
        "resource_preflight": resource, "gate": "PASS",
    }
    diagonal_audit = {
        "classification": "P6_PHASEA8MRN2R8E_DIAGONAL_PRIOR_RUNTIME_AUDIT_PASS",
        "created": r3.now(), "prior": "TRAIN140-derived approximate unit diagonal",
        "TRAIN140_normalized_real_diagonal": expected_diag,
        "TRAIN140_imag_diagonal_max_abs": 0.0,
        "target": {"real": 1.0, "imag": 0.0},
        "target_source": "TRAIN140 normalized diagonal median only",
        "VAL_diagonal_used": False, "INTERNAL_TEST_diagonal_used": False,
        "burned24_diagonal_used": False, "fresh30_used": False,
        "coordinates_per_update": DIAGONAL_PAIRS_PER_UPDATE,
        "local_indices": list(DIAGONAL_LOCAL_INDICES),
        "new_RNG": False, "loss_or_error_based_selection": False,
        "strata_based_selection": False, "q_source_equals_q_output": True,
        "d_zero_exact": True, "dense_global_diagonal_prior_coverage": True,
        "R8D_coverage_fraction": coverage["unique_coordinate_fraction"],
        "initial_frozen_dummy_gradient_context": {
            "field_grad_norm": resource["field_gradient_norm"],
            "raw_diag_grad_norm": resource["diagonal_gradient_norm"],
            "weighted_diag_grad_norm": weighted_diag_grad,
            "weighted_diag_grad_over_field_grad": weighted_ratio,
            "interpretation": "coefficient-low auxiliary prior whose initial weighted gradient magnitude is comparable to or larger than field gradient on the frozen dummy audit",
            "stopping_gate": False, "lambda_change_allowed": False,
        },
        "runtime_gradient_decomposition": {
            "status": "SKIPPED_TO_PRESERVE_SCIENCE_TRAJECTORY",
            "reason": "No extra backward passes are introduced into the formal AMP/GradScaler training loop.",
        },
        "gate": "PASS",
    }
    return model_audit, diagonal_audit, resource


def diagonal_stream_initial_state() -> bytes:
    return __import__("hashlib").sha256(b"P6_PHASEA8MRN2R8D_DIAGONAL_COORDINATE_STREAM_V1").digest()


def diagonal_stream_update(state: bytes, update: int, anchor_ids: np.ndarray, q_linear: np.ndarray) -> bytes:
    digest = __import__("hashlib").sha256()
    digest.update(state)
    digest.update(np.asarray([update], dtype="<u4").tobytes())
    digest.update(np.asarray(anchor_ids, dtype="<u2").tobytes())
    digest.update(np.asarray(DIAGONAL_LOCAL_INDICES, dtype="<u2").tobytes())
    digest.update(np.asarray(q_linear, dtype="<u4").tobytes())
    return digest.digest()


def prepare_fresh(device: torch.device) -> None:
    collisions = [str(path) for path in FINAL_OUTPUTS if path.exists()]
    r3.require(not collisions, f"R8E formal output collision: {collisions}")
    r3.require(not WORK_DIR.exists(), f"R8E work directory already exists: {WORK_DIR}")
    input_freeze, r8d_summary = hard_precheck()
    model_audit, diagonal_audit, resource = model_and_resource_preflight(device)
    r3.clear_submission()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=False)
    lineage = str(uuid.uuid4())
    r3.write_new_json(INPUT_FREEZE, input_freeze)
    protocol = {
        "classification": "P6_PHASEA8MRN2R8E_EXECUTION_PROTOCOL_FROZEN",
        "created": r3.now(), "formal_lineage_id": lineage,
        "CommandLine": commandline(False), "R8D_precheck": "PASS",
        "candidate": CANDIDATE, "parent": "H3",
        "interpretation_boundary": {
            "stage": "DEVELOPMENT_NOT_FORMAL_FRESH_VALIDATION",
            "tests": "H3_TO_H4_SINGLE_VARIABLE_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY",
            "unique_change": "TRAIN140-derived dense approximate unit-diagonal consistency",
            "does_not_test": "new exact physics, architecture changes, parameter/input changes, lambda sweep, or post-hoc inference correction",
            "candidate_count": 1,
            "dense_global_diagonal_prior_coverage": True,
            "prior_is_not_exact_off_lattice_physics_label": True,
        },
        "parameter_count": EXPECTED_PARAMS, "input_dimension": 165,
        "initial_state_SHA256": EXPECTED_INITIAL_SHA,
        "boundary_features": {
            "names": list(BOUNDARY_FEATURE_NAMES),
            "global_tensor_indices": list(BOUNDARY_AUXILIARY_GLOBAL_INDICES),
            "unchanged_from_H3": True,
        },
        "source_encoding": {
            "retained_raw_xyz": True, "x_retained_k": [0, 1, 2],
            "y_retained_k": [0, 1, 2], "z_retained_k": [0, 1],
            "z_k2_fixed_zero": True, "all_axes_k4_to_k7_fixed_zero": True,
            "k3_semantics": "boundary auxiliary, not Fourier",
        },
        "output_absolute_k": list(range(8)), "relative_displacement_k": list(range(10)),
        "ordered_branches": {"A": "ordered_base(u,r,d)", "R": "ordered_base(r,u,-d)", "B": "conj(R)"},
        "field_loss": "0.5*MSE_complex(A,y)+0.5*MSE_complex(B,y)",
        "field_coefficient": 1.0,
        "diagonal_pair": "source=q, output=q, d=(0,0,0)",
        "diagonal_target": "1+0j",
        "diagonal_loss": "0.5*MSE_complex(A_diag,1+0j)+0.5*MSE_complex(B_diag,1+0j)",
        "lambda_diag": LAMBDA_DIAG,
        "total_loss": "L_field+0.1*L_diag",
        "inference": "0.5*(A+B)",
        "no_post_hoc_inference_intervention": True,
        "initial_frozen_dummy_gradient_context": diagonal_audit["initial_frozen_dummy_gradient_context"],
        "gradient_context_interpretation": "coefficient-low auxiliary prior whose initial weighted gradient magnitude is comparable to or larger than field gradient on the frozen dummy audit",
        "runtime_gradient_decomposition": diagonal_audit["runtime_gradient_decomposition"],
        "g_ref": G_REF, "split": [140, 20, 20], "split_SHA256": EXPECTED_SPLIT_SHA,
        "blacklist_unique": 136, "blacklist_SHA256": EXPECTED_BLACKLIST_SHA,
        "field_sampler": {
            "sources_per_update": 4, "queries_per_source": 8192,
            "pairs_per_update": FIELD_PAIRS_PER_UPDATE,
            "shells": list(n2.SHELL_NAMES), "queries_per_shell": n2.QUERIES_PER_SHELL,
            "center_mandatory": True,
        },
        "diagonal_sampler": {
            "pairs_per_update": DIAGONAL_PAIRS_PER_UPDATE,
            "per_source": 256, "local_indices": list(DIAGONAL_LOCAL_INDICES),
            "from_existing_post_C4_field_queries": True, "new_RNG": False,
        },
        "sampler_seed": n2.COMMON_SAMPLER_SEED, "C4_seed": n2.C4_SEED,
        "field_stream_target_SHA256": EXPECTED_FIELD_STREAM_SHA,
        "diagonal_stream_target_SHA256": EXPECTED_DIAGONAL_STREAM_SHA,
        "optimizer": {"name": "AdamW", "lr": 2e-4, "betas": [0.9, 0.99], "weight_decay": 1e-6, "scheduler": None},
        "AMP": "inherited H3 policy: FP16 autocast forward; FP32 losses/backward; GradScaler enabled",
        "budget": {"updates": MAX_UPDATES, "validation_interval": VAL_INTERVAL, "checkpoint_interval": VAL_INTERVAL, "checkpoints": 40, "early_stopping": False},
        "VAL_selection": {
            "sources": 20, "probes_per_source": 65536, "coordinate_SHA256": EXPECTED_VAL_SHA,
            "primary": "lowest pooled final-sym full complex NRMSE",
            "secondary": "lowest max-column final-sym NRMSE",
            "tie_within": 1e-6, "tie_break": "earlier update",
            "TRAIN_diag_loss_contribution": 0, "VAL_gain_contribution": 0,
            "INTERNAL_TEST_contribution": 0, "burned24_contribution": 0,
            "gradient_ratio_contribution": 0,
        },
        "R8E_outcome_protocol": r3.read_json(R8D_OUTCOME),
        "access_order": ["R8D hard precheck", "H4 identity/equivalence/loss/resource preflight", "execution source freeze", "optimizer construction", "20k H4 training", "VAL-only selection", "H4 best freeze", "network-only diagonal self-consistency", "INTERNAL_TEST", "cancellation retest", "burned24", "controlled comparisons", "readiness and R8E outcome"],
        "actual_execution_timeline": [], "gate": "PASS",
    }
    r3.write_new_json(PROTOCOL_EXEC, protocol)
    source_paths = [
        SCRIPT, MONITOR, H4_SOURCE, H3_SOURCE, MODEL_SOURCE, R3_SOURCE,
        R8D_SOURCE, R8D_PROTOCOL, R8D_LOSS_SPEC, R8D_COORDINATE_SPEC,
        R8D_STREAM, R8D_PRIOR, R8D_OUTCOME, N2_SOURCE, R1_SOURCE,
        RUNTIME_HELPER, METRIC_HELPER, R50_REFERENCE, SPLIT_PATH,
        BLACKLIST_PATH, n2.N1R_VAL_PROBES,
    ]
    execution = {
        "classification": "P6_PHASEA8MRN2R8E_EXECUTION_SOURCE_FREEZE_PASS",
        "created": r3.now(), "formal_lineage_id": lineage,
        "CommandLine": commandline(False),
        "sources": [{"role": path.name, "path": str(path.resolve()), "bytes": path.stat().st_size, "SHA256": r3.sha256(path), "read_only_after_freeze": True} for path in source_paths],
        "worker_SHA256": r3.sha256(SCRIPT), "monitor_SHA256": r3.sha256(MONITOR),
        "H4_helper_SHA256": r3.sha256(H4_SOURCE),
        "H3_parent_source_SHA256": r3.sha256(H3_SOURCE),
        "R8D_protocol_SHA256": r3.sha256(R8D_PROTOCOL),
        "loss_spec_SHA256": r3.sha256(R8D_LOSS_SPEC),
        "coordinate_spec_SHA256": r3.sha256(R8D_COORDINATE_SPEC),
        "field_sampler_helper_SHA256": r3.sha256(RUNTIME_HELPER),
        "metric_helper_SHA256": r3.sha256(METRIC_HELPER),
        "split_SHA256": EXPECTED_SPLIT_SHA, "blacklist_SHA256": EXPECTED_BLACKLIST_SHA,
        "VAL_probe_SHA256": EXPECTED_VAL_SHA,
        "field_stream_SHA256": EXPECTED_FIELD_STREAM_SHA,
        "diagonal_stream_SHA256": EXPECTED_DIAGONAL_STREAM_SHA,
        "initial_state_SHA256": EXPECTED_INITIAL_SHA, "lambda_diag": LAMBDA_DIAG,
        "Python_version": platform.python_version(), "Python_executable": sys.executable,
        "PyTorch_version": torch.__version__, "CUDA_version": torch.version.cuda,
        "GPU": torch.cuda.get_device_name(0), "resource_preflight": resource,
        "science_optimizer_constructed_before_freeze": False,
        "science_optimizer_updates_before_freeze": 0, "gate": "PASS",
    }
    r3.write_new_json(EXECUTION_FREEZE, execution)
    r3.write_new_json(MODEL_RUNTIME_AUDIT, model_audit)
    r3.write_new_json(DIAGONAL_RUNTIME_AUDIT, diagonal_audit)
    r3.write_new_json(FIELD_STREAM_AUDIT, {
        "classification": "P6_PHASEA8MRN2R8E_FIELD_SAMPLE_STREAM_RUNNING",
        "created": r3.now(), "hash_definition": r3.read_json(n2.SAMPLER_AUDIT)["hash_definition"],
        "authoritative_sample_stream_SHA256": EXPECTED_FIELD_STREAM_SHA,
        "current_update": 0, "H4_field_sample_stream_SHA256": n2.stream_initial_state().hex(),
        "match": None, "pre_rotation_blacklist_hits": 0,
        "post_rotation_blacklist_hits": 0, "replacement_fallback_events": [],
        "gate": "PENDING",
    })
    r3.write_new_json(DIAGONAL_STREAM_AUDIT, {
        "classification": "P6_PHASEA8MRN2R8E_DIAGONAL_STREAM_RUNNING",
        "created": r3.now(), "hash_definition": r3.read_json(R8D_STREAM)["diagonal_hash_chain_update_payload"],
        "authoritative_diagonal_stream_SHA256": EXPECTED_DIAGONAL_STREAM_SHA,
        "current_update": 0,
        "H4_diagonal_stream_SHA256": diagonal_stream_initial_state().hex(),
        "match": None, "total_occurrences": 0, "unique_coordinates": 0,
        "coverage_fraction": 0.0, "unique_x_levels": 0,
        "unique_y_levels": 0, "unique_z_levels": 0,
        "pre_C4_blacklist_hits": 0, "post_C4_blacklist_hits": 0,
        "full_stream_saved": False, "gate": "PENDING",
    })
    r3.write_new_json(TRAINING_HISTORY, {
        "classification": "P6_PHASEA8MRN2R8E_H4_TRAINING_RUNNING",
        "created": r3.now(), "formal_lineage_id": lineage, "candidate": CANDIDATE,
        "completed_updates": 0, "VAL_history": [],
        "recent_field_losses": [], "recent_diag_losses": [],
        "recent_weighted_diag_losses": [], "recent_total_losses": [],
        "field_stream_SHA256": n2.stream_initial_state().hex(),
        "diagonal_stream_SHA256": diagonal_stream_initial_state().hex(),
        "runtime_gradient_decomposition": "SKIPPED_TO_PRESERVE_SCIENCE_TRAJECTORY",
        "gate": "PENDING",
    })
    r3.write_new_json(STATUS, {
        "status": "RUNNING", "phase": "R8D_PRECHECK_H4_PREFLIGHT_EXECUTION_FREEZE_PASS",
        "PID": os.getpid(), "CommandLine": commandline(False), "updated": r3.now(),
        "formal_lineage_id": lineage, "R8D_precheck": "PASS",
        "execution_freeze_SHA": r3.sha256(EXECUTION_FREEZE),
        "candidate": CANDIDATE, "parent": "H3",
        "H4_source_SHA_match": True, "params": EXPECTED_PARAMS, "input": 165,
        "initial_SHA_match": True, "lambda_diag": LAMBDA_DIAG,
        "field_pairs_per_update": FIELD_PAIRS_PER_UPDATE,
        "diag_pairs_per_update": DIAGONAL_PAIRS_PER_UPDATE,
        "current_update": 0, "total_updates": MAX_UPDATES,
        "science_optimizer_updates": 0,
        "latest_field_loss": None, "latest_diag_loss": None,
        "latest_weighted_diag_loss": None, "latest_total_loss": None,
        "latest_VAL_pooled": None, "latest_VAL_max": None,
        "best_update": None, "best_VAL_pooled": None, "best_VAL_max": None,
        "field_stream_status": "INITIALIZED", "diag_stream_status": "INITIALIZED",
        "field_blacklist_hits": 0, "diag_blacklist_hits": 0,
        "VAL_opened": False, "VAL_columns_cached": 0,
        "H4_best_frozen": False, "INTERNAL_TEST_opened": False,
        "INTERNAL_TEST_complete": 0, "INTERNAL_TEST_total": 20,
        "internal_full_median": None, "internal_gain_max": None,
        "internal_rawA_median": None, "internal_rawB_median": None,
        "internal_disagreement_median": None, "internal_error_cosine": None,
        "internal_cancellation_ratio": None, "CANCEL4": None,
        "burned_opened": False, "burned_complete": 0, "burned_total": 24,
        "burned_full_median": None, "burned_full_max": None,
        "burned_energy_median": None, "burned_energy_max": None,
        "burned_boundary_median": None, "burned_boundary_max": None,
        "burned_gain_median": None, "burned_gain_max": None,
        "burned_peak_max": None, "READY4": None, "outcome_case": None,
        "classification": "P6_PHASEA8MRN2R8E_RUNNING",
        "recommended_next": None, "new_physics": 0,
        "fresh30_selected": 0, "fresh30_accessed": 0,
        "operator_cases": 0, **r3.gpu_telemetry(),
        "stderr_bytes": r3.stderr_bytes(),
    })
    r3.record_event("R8D_HARD_PRECHECK_PASS", R8E_execution_ready=True)
    r3.record_event("H4_MODEL_EQUIVALENCE_HERMITIAN_LOSS_GRADIENT_RESOURCE_PREFLIGHT_PASS", H4_source_SHA256=EXPECTED_H4_SOURCE_SHA)
    r3.record_event("EXECUTION_SOURCE_FREEZE_CREATED", SHA256=r3.sha256(EXECUTION_FREEZE))


def verify_execution_freeze() -> None:
    freeze = r3.read_json(EXECUTION_FREEZE)
    r3.require(freeze["gate"] == "PASS", "Execution freeze gate failed")
    for row in freeze["sources"]:
        r3.require(r3.sha256(Path(row["path"])) == row["SHA256"], f"Frozen science source changed: {row['path']}")
    r3.require(r3.sha256(R8D_PROTOCOL) == freeze["R8D_protocol_SHA256"], "R8D protocol changed")
    r3.require(r3.sha256(R8D_LOSS_SPEC) == freeze["loss_spec_SHA256"], "R8D loss spec changed")
    r3.require(r3.sha256(R8D_COORDINATE_SPEC) == freeze["coordinate_spec_SHA256"], "R8D coordinate spec changed")
    r3.require(r3.sha256(H4_SOURCE) == EXPECTED_H4_SOURCE_SHA, "H4 source changed")
    r3.require(r3.sha256(H3_SOURCE) == EXPECTED_H3_SOURCE_SHA, "H3 parent source changed")
    input_freeze = r3.read_json(INPUT_FREEZE)
    for row in input_freeze["artifacts"]:
        r3.require(r3.sha256(Path(row["path"])) == row["SHA256"], f"Frozen R8D input changed: {row['path']}")


def checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    update: int,
    source_rng: np.random.Generator,
    query_rng: np.random.Generator,
    c4_rng: np.random.Generator,
    val_history: list[dict[str, Any]],
    field_stream_state: bytes,
    diagonal_stream_state: bytes,
    recent_field: list[float],
    recent_diag: list[float],
    recent_weighted_diag: list[float],
    recent_total: list[float],
    diagonal_coverage: np.ndarray,
    training_wall: float,
    validation_wall: float,
) -> dict[str, Any]:
    return {
        "formal_lineage_id": r3.read_json(STATUS)["formal_lineage_id"],
        "candidate": "H4", "candidate_full_name": CANDIDATE,
        "parent": "H3", "update": update,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "GradScaler_state": scaler.state_dict(),
        "Python_RNG_state": random.getstate(),
        "NumPy_global_RNG_state": np.random.get_state(),
        "source_RNG_state": source_rng.bit_generator.state,
        "query_RNG_state": query_rng.bit_generator.state,
        "C4_RNG_state": c4_rng.bit_generator.state,
        "torch_CPU_RNG_state": torch.get_rng_state(),
        "torch_CUDA_RNG_states": torch.cuda.get_rng_state_all(),
        "VAL_history": val_history,
        "field_stream_incremental_SHA256_state": field_stream_state.hex(),
        "diagonal_stream_incremental_SHA256_state": diagonal_stream_state.hex(),
        "field_stream_hash_scheme": "N2 SHA256 chain V1",
        "diagonal_stream_hash_scheme": "R8D diagonal coordinate SHA256 chain V1",
        "diagonal_coverage_packbits_little": np.packbits(diagonal_coverage, bitorder="little").tobytes(),
        "recent_field_losses": recent_field,
        "recent_diag_losses": recent_diag,
        "recent_weighted_diag_losses": recent_weighted_diag,
        "recent_total_losses": recent_total,
        "training_wall_seconds": training_wall,
        "validation_wall_seconds": validation_wall,
        "g_ref": G_REF, "lambda_diag": LAMBDA_DIAG,
        "initial_state_SHA256": EXPECTED_INITIAL_SHA,
        "H4_source_SHA256": EXPECTED_H4_SOURCE_SHA,
        "H3_source_SHA256": EXPECTED_H3_SOURCE_SHA,
        "N2_train_statistics_SHA256": r3.sha256(n2.TRAIN_STATS),
        "execution_freeze_SHA256": r3.sha256(EXECUTION_FREEZE),
        "R8D_protocol_SHA256": r3.sha256(R8D_PROTOCOL),
        "saved": r3.now(),
    }


def latest_checkpoint() -> Path | None:
    paths = sorted(CHECKPOINT_DIR.glob("H4_update_*.pt"))
    return paths[-1] if paths else None


def restore_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> tuple[int, np.random.Generator, np.random.Generator, np.random.Generator, list[dict[str, Any]], bytes, bytes, list[float], list[float], list[float], list[float], np.ndarray, float, float]:
    checkpoint = torch.load(path, map_location="cuda:0", weights_only=False)
    r3.require(checkpoint["formal_lineage_id"] == r3.read_json(STATUS)["formal_lineage_id"], "Checkpoint lineage mismatch")
    r3.require(checkpoint["candidate"] == "H4" and checkpoint["candidate_full_name"] == CANDIDATE and checkpoint["parent"] == "H3", "Checkpoint candidate mismatch")
    r3.require(checkpoint["execution_freeze_SHA256"] == r3.sha256(EXECUTION_FREEZE), "Checkpoint execution-freeze mismatch")
    r3.require(checkpoint["R8D_protocol_SHA256"] == r3.sha256(R8D_PROTOCOL), "Checkpoint R8D protocol mismatch")
    r3.require(checkpoint["initial_state_SHA256"] == EXPECTED_INITIAL_SHA and checkpoint["H4_source_SHA256"] == EXPECTED_H4_SOURCE_SHA, "Checkpoint H4 identity mismatch")
    r3.require(checkpoint["g_ref"] == G_REF and checkpoint["lambda_diag"] == LAMBDA_DIAG, "Checkpoint g_ref/lambda mismatch")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["GradScaler_state"])
    source_rng, query_rng, c4_rng = n2.make_rng_bundle()
    source_rng.bit_generator.state = checkpoint["source_RNG_state"]
    query_rng.bit_generator.state = checkpoint["query_RNG_state"]
    c4_rng.bit_generator.state = checkpoint["C4_RNG_state"]
    random.setstate(checkpoint["Python_RNG_state"])
    np.random.set_state(checkpoint["NumPy_global_RNG_state"])
    torch.set_rng_state(checkpoint["torch_CPU_RNG_state"])
    torch.cuda.set_rng_state_all(checkpoint["torch_CUDA_RNG_states"])
    packed = np.frombuffer(checkpoint["diagonal_coverage_packbits_little"], dtype=np.uint8)
    coverage = np.unpackbits(packed, bitorder="little")[:n2.NATIVE_VOXELS].astype(bool, copy=True)
    val_history = checkpoint["VAL_history"]
    for row in val_history:
        candidate_path = CHECKPOINT_DIR / f"H4_update_{int(row['update']):05d}.pt"
        if candidate_path.is_file():
            row.setdefault("checkpoint_path", str(candidate_path.resolve()))
            row.setdefault("checkpoint_SHA256", r3.sha256(candidate_path))
    return (
        int(checkpoint["update"]), source_rng, query_rng, c4_rng, val_history,
        bytes.fromhex(checkpoint["field_stream_incremental_SHA256_state"]),
        bytes.fromhex(checkpoint["diagonal_stream_incremental_SHA256_state"]),
        checkpoint.get("recent_field_losses", []),
        checkpoint.get("recent_diag_losses", []),
        checkpoint.get("recent_weighted_diag_losses", []),
        checkpoint.get("recent_total_losses", []), coverage,
        float(checkpoint.get("training_wall_seconds", 0.0)),
        float(checkpoint.get("validation_wall_seconds", 0.0)),
    )


def diagonal_coverage_summary(bitset: np.ndarray, update: int, final: bool) -> dict[str, Any]:
    selected = np.flatnonzero(bitset).astype(np.uint32, copy=False)
    if selected.size:
        xyz = n2.unravel_indices(selected)
        x_levels = int(np.unique(xyz[:, 0]).size)
        y_levels = int(np.unique(xyz[:, 1]).size)
        z_levels = int(np.unique(xyz[:, 2]).size)
    else:
        x_levels = y_levels = z_levels = 0
    return {
        "current_update": update,
        "total_occurrences": update * DIAGONAL_PAIRS_PER_UPDATE,
        "unique_coordinates": int(selected.size),
        "native_domain_count": n2.NATIVE_VOXELS,
        "coverage_fraction": float(selected.size / n2.NATIVE_VOXELS),
        "unique_x_levels": x_levels, "unique_y_levels": y_levels,
        "unique_z_levels": z_levels,
        "pre_C4_blacklist_hits": 0, "post_C4_blacklist_hits": 0,
        "full_stream_saved": False,
        "expected_final_unique_coordinates": 2698300,
        "expected_final_coverage_fraction": 0.9993703703703704,
        "final_reference_match": bool(final and selected.size == 2698300 and x_levels == 150 and y_levels == 150 and z_levels == 120),
    }


def train_h4(
    device: torch.device,
    roles: dict[str, list[dict[str, Any]]],
    blocked: np.ndarray,
    stats: dict[str, Any],
    train_maps: dict[int, np.memmap],
    shells: dict[int, dict[str, np.ndarray]],
    val_cache: n2.ValidationCache,
    tracker: n2.AccessTracker,
    resume_path: Path | None,
) -> dict[str, Any]:
    verify_execution_freeze()
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    model = build_h4(MODEL_SEED).to(device)
    r3.require(state_sha256(model) == EXPECTED_INITIAL_SHA and parameter_count(model) == EXPECTED_PARAMS, "Training H4 identity mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.99), weight_decay=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    r3.record_event("SCIENCE_OPTIMIZER_CONSTRUCTED", prior_science_updates=0, lambda_diag=LAMBDA_DIAG)
    source_rng, query_rng, c4_rng = n2.make_rng_bundle()
    val_history: list[dict[str, Any]] = []
    field_stream_state = n2.stream_initial_state()
    diagonal_stream_state = diagonal_stream_initial_state()
    recent_field: list[float] = []
    recent_diag: list[float] = []
    recent_weighted_diag: list[float] = []
    recent_total: list[float] = []
    diagonal_coverage = np.zeros(n2.NATIVE_VOXELS, dtype=bool)
    training_wall = 0.0
    validation_wall = 0.0
    start_update = 1
    if resume_path is not None:
        restored = restore_checkpoint(resume_path, model, optimizer, scaler)
        (
            last_update, source_rng, query_rng, c4_rng, val_history,
            field_stream_state, diagonal_stream_state, recent_field,
            recent_diag, recent_weighted_diag, recent_total,
            diagonal_coverage, training_wall, validation_wall,
        ) = restored
        start_update = last_update + 1
        r3.record_event("H4_RESUMED", checkpoint=str(resume_path.resolve()), update=last_update)
    else:
        r3.record_event("H4_TRAINING_STARTED", update=1)
    r3.update_status(phase="H4_TRAINING", current_update=start_update - 1, science_optimizer_updates=start_update - 1)
    torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(0)
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss
    io_start = process.io_counters().read_bytes
    started_total = time.perf_counter()
    fallback_events: list[dict[str, Any]] = []
    local = np.asarray(DIAGONAL_LOCAL_INDICES, dtype=np.int64)
    diagonal_positions = np.concatenate([base + local for base in range(0, FIELD_PAIRS_PER_UPDATE, n2.QUERIES_PER_SOURCE)])
    r3.require(diagonal_positions.size == DIAGONAL_PAIRS_PER_UPDATE, "Diagonal position count mismatch")

    for update in range(start_update, MAX_UPDATES + 1):
        loop_start = time.perf_counter()
        output_u, source_r, target, anchor_ids, query_indices, rotations = n2.make_training_batch(
            "H4", update, roles["TRAIN"], train_maps, shells, blocked, stats["g_ref"],
            source_rng, query_rng, c4_rng, fallback_events, tracker,
        )
        field_stream_state = n2.stream_update(field_stream_state, update, anchor_ids, query_indices, rotations)
        rotated = n2.c4_rotate_indices(n2.unravel_indices(query_indices), rotations)
        rotated_linear = ((rotated[:, 0].astype(np.int64) * n2.SHAPE[1] + rotated[:, 1]) * n2.SHAPE[2] + rotated[:, 2])
        r3.require(not bool(np.any(blocked[query_indices])), "Runtime field pre-C4 blacklist hit")
        r3.require(not bool(np.any(blocked[rotated_linear])), "Runtime field post-C4 blacklist hit")
        diagonal_pre = query_indices[diagonal_positions]
        diagonal_linear = rotated_linear[diagonal_positions]
        r3.require(not bool(np.any(blocked[diagonal_pre])), "Runtime diagonal pre-C4 blacklist hit")
        r3.require(not bool(np.any(blocked[diagonal_linear])), "Runtime diagonal post-C4 blacklist hit")
        diagonal_stream_state = diagonal_stream_update(diagonal_stream_state, update, anchor_ids, diagonal_linear)
        diagonal_coverage[diagonal_linear] = True
        diagonal_q = output_u[diagonal_positions]
        output_u = output_u.to(device)
        source_r = source_r.to(device)
        target = target.to(device)
        diagonal_q = diagonal_q.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            field_loss, _, _ = field_dual_loss(model, output_u, source_r, target)
            diag_loss, _, _, _ = diagonal_dual_loss(model, diagonal_q)
            total_loss = h4_total_loss(field_loss, diag_loss)
        r3.require(bool(torch.isfinite(field_loss)) and bool(torch.isfinite(diag_loss)) and bool(torch.isfinite(total_loss)), f"Nonfinite H4 loss at update {update}")
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        r3.require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()), f"Nonfinite H4 gradient at update {update}")
        prior_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        r3.require(scaler.get_scale() >= prior_scale, f"GradScaler skipped optimizer update {update}")

        field_value = float(field_loss.detach().cpu())
        diag_value = float(diag_loss.detach().cpu())
        weighted_diag_value = LAMBDA_DIAG * diag_value
        total_value = float(total_loss.detach().cpu())
        recent_field = (recent_field + [field_value])[-VAL_INTERVAL:]
        recent_diag = (recent_diag + [diag_value])[-VAL_INTERVAL:]
        recent_weighted_diag = (recent_weighted_diag + [weighted_diag_value])[-VAL_INTERVAL:]
        recent_total = (recent_total + [total_value])[-VAL_INTERVAL:]
        training_wall += time.perf_counter() - loop_start
        peak_ram = max(peak_ram, process.memory_info().rss)

        if update % 50 == 0:
            r3.update_status(
                phase="H4_TRAINING", current_update=update,
                science_optimizer_updates=update,
                latest_field_loss=field_value, latest_diag_loss=diag_value,
                latest_weighted_diag_loss=weighted_diag_value,
                latest_total_loss=total_value,
                field_stream_status=f"RUNNING:{field_stream_state.hex()}",
                diag_stream_status=f"RUNNING:{diagonal_stream_state.hex()}",
                field_blacklist_hits=0, diag_blacklist_hits=0,
                **r3.gpu_telemetry(),
            )

        if update % VAL_INTERVAL == 0:
            verify_execution_freeze()
            validation, duration = n2.validate(model, stats["g_ref"], val_cache, device)
            validation_wall += duration
            field_mean = float(np.mean(recent_field))
            diag_mean = float(np.mean(recent_diag))
            weighted_mean = float(np.mean(recent_weighted_diag))
            total_mean = float(np.mean(recent_total))
            row = {
                "update": update,
                "training_field_loss_recent_mean": field_mean,
                "training_diag_loss_recent_mean": diag_mean,
                "training_weighted_diag_loss_recent_mean": weighted_mean,
                "training_total_loss_recent_mean": total_mean,
                "weighted_diag_loss_over_field_loss": weighted_mean / max(field_mean, 1e-30),
                **validation, "validation_wall_seconds": duration,
                "diagnostic_ratio_used_for_selection": False,
            }
            val_history.append(row)
            checkpoint_path = CHECKPOINT_DIR / f"H4_update_{update:05d}.pt"
            state = checkpoint_state(
                model, optimizer, scaler, update, source_rng, query_rng, c4_rng,
                val_history, field_stream_state, diagonal_stream_state,
                recent_field, recent_diag, recent_weighted_diag, recent_total,
                diagonal_coverage, training_wall, validation_wall,
            )
            n2.atomic_torch_save(state, checkpoint_path)
            row["checkpoint_path"] = str(checkpoint_path.resolve())
            row["checkpoint_SHA256"] = r3.sha256(checkpoint_path)
            best = r3.best_val(val_history)
            r3.require(best is not None, "No H4 VAL best row")
            history = r3.read_json(TRAINING_HISTORY)
            history.update({
                "completed_updates": update, "VAL_history": val_history,
                "recent_field_losses": recent_field,
                "recent_diag_losses": recent_diag,
                "recent_weighted_diag_losses": recent_weighted_diag,
                "recent_total_losses": recent_total,
                "field_stream_SHA256": field_stream_state.hex(),
                "diagonal_stream_SHA256": diagonal_stream_state.hex(),
                "training_wall_seconds": training_wall,
                "validation_wall_seconds": validation_wall,
            })
            r3.atomic_json(TRAINING_HISTORY, history)
            r3.atomic_json(FIELD_STREAM_AUDIT, {
                "classification": "P6_PHASEA8MRN2R8E_FIELD_SAMPLE_STREAM_RUNNING",
                "updated": r3.now(),
                "hash_definition": r3.read_json(FIELD_STREAM_AUDIT)["hash_definition"],
                "authoritative_sample_stream_SHA256": EXPECTED_FIELD_STREAM_SHA,
                "current_update": update,
                "H4_field_sample_stream_SHA256": field_stream_state.hex(),
                "match": None, "pre_rotation_blacklist_hits": 0,
                "post_rotation_blacklist_hits": 0,
                "replacement_fallback_events": fallback_events,
                "gate": "PENDING",
            })
            diagonal_state_row = diagonal_coverage_summary(diagonal_coverage, update, False)
            r3.atomic_json(DIAGONAL_STREAM_AUDIT, {
                "classification": "P6_PHASEA8MRN2R8E_DIAGONAL_STREAM_RUNNING",
                "updated": r3.now(),
                "hash_definition": r3.read_json(R8D_STREAM)["diagonal_hash_chain_update_payload"],
                "authoritative_diagonal_stream_SHA256": EXPECTED_DIAGONAL_STREAM_SHA,
                "H4_diagonal_stream_SHA256": diagonal_stream_state.hex(),
                "match": None, **diagonal_state_row, "gate": "PENDING",
            })
            r3.update_status(
                phase="H4_VALIDATED_CHECKPOINTED", current_update=update,
                latest_field_loss=field_value, latest_diag_loss=diag_value,
                latest_weighted_diag_loss=weighted_diag_value,
                latest_total_loss=total_value,
                latest_VAL_pooled=validation["pooled_VAL_complex_NRMSE"],
                latest_VAL_max=validation["max_per_column_VAL_complex_NRMSE"],
                best_update=best["update"],
                best_VAL_pooled=best["pooled_VAL_complex_NRMSE"],
                best_VAL_max=best["max_per_column_VAL_complex_NRMSE"],
            )
            print(json.dumps({
                "candidate": "H4", "update": update,
                "field": field_mean, "diag": diag_mean,
                "weighted_diag": weighted_mean, "total": total_mean,
                "weighted_diag_over_field": row["weighted_diag_loss_over_field_loss"],
                "VAL_pooled": validation["pooled_VAL_complex_NRMSE"],
                "VAL_max": validation["max_per_column_VAL_complex_NRMSE"],
                "best_update": best["update"],
            }), flush=True)
        del output_u, source_r, target, diagonal_q, field_loss, diag_loss, total_loss

    r3.require(len(val_history) == 40, "Expected exactly 40 H4 VAL points")
    r3.require(field_stream_state.hex() == EXPECTED_FIELD_STREAM_SHA, "H4 field sample stream mismatch")
    r3.require(diagonal_stream_state.hex() == EXPECTED_DIAGONAL_STREAM_SHA, "H4 diagonal stream mismatch")
    best = r3.best_val(val_history)
    r3.require(best is not None, "No H4 best checkpoint")
    best_path = Path(best["checkpoint_path"])
    r3.require(best_path.is_file() and r3.sha256(best_path) == best["checkpoint_SHA256"], "H4 best checkpoint hash mismatch")
    freeze = {
        "classification": "P6_PHASEA8MRN2R8E_H4_BEST_CHECKPOINT_FROZEN",
        "created": r3.now(), "formal_lineage_id": r3.read_json(STATUS)["formal_lineage_id"],
        "candidate": CANDIDATE, "parent": "H3",
        "best_update": best["update"], "checkpoint_path": str(best_path.resolve()),
        "checkpoint_SHA256": r3.sha256(best_path),
        "pooled_VAL": best["pooled_VAL_complex_NRMSE"],
        "max_column_VAL": best["max_per_column_VAL_complex_NRMSE"],
        "final_update_VAL": {
            "update": val_history[-1]["update"],
            "pooled": val_history[-1]["pooled_VAL_complex_NRMSE"],
            "max_column": val_history[-1]["max_per_column_VAL_complex_NRMSE"],
        },
        "H4_source_SHA256": r3.sha256(H4_SOURCE),
        "H3_parent_source_SHA256": r3.sha256(H3_SOURCE),
        "initial_state_SHA256": EXPECTED_INITIAL_SHA,
        "field_stream_SHA256": field_stream_state.hex(),
        "diagonal_stream_SHA256": diagonal_stream_state.hex(),
        "split_SHA256": EXPECTED_SPLIT_SHA,
        "VAL_probe_SHA256": EXPECTED_VAL_SHA,
        "blacklist_SHA256": EXPECTED_BLACKLIST_SHA,
        "lambda_diag": LAMBDA_DIAG,
        "INTERNAL_TEST_access_before_freeze": False,
        "burned24_access_before_freeze": False,
        "selection_inputs": "VAL final-sym only", "gate": "PASS",
    }
    r3.write_new_json(BEST_FREEZE, freeze)
    freeze_sha = r3.sha256(BEST_FREEZE)
    field_artifact = r3.read_json(FIELD_STREAM_AUDIT)
    field_artifact.update({
        "classification": "P6_PHASEA8MRN2R8E_FIELD_SAMPLE_STREAM_MATCH_PASS",
        "completed": r3.now(), "current_update": MAX_UPDATES,
        "H4_field_sample_stream_SHA256": field_stream_state.hex(),
        "match": True, "gate": "PASS",
    })
    r3.atomic_json(FIELD_STREAM_AUDIT, field_artifact)
    final_coverage = diagonal_coverage_summary(diagonal_coverage, MAX_UPDATES, True)
    r3.require(final_coverage["final_reference_match"], "H4 diagonal coverage differs from R8D replay")
    diagonal_artifact = r3.read_json(DIAGONAL_STREAM_AUDIT)
    diagonal_artifact.update({
        "classification": "P6_PHASEA8MRN2R8E_DIAGONAL_STREAM_MATCH_PASS",
        "completed": r3.now(), "current_update": MAX_UPDATES,
        "H4_diagonal_stream_SHA256": diagonal_stream_state.hex(),
        "match": True, **final_coverage, "gate": "PASS",
    })
    r3.atomic_json(DIAGONAL_STREAM_AUDIT, diagonal_artifact)
    history = r3.read_json(TRAINING_HISTORY)
    history.update({
        "classification": "P6_PHASEA8MRN2R8E_H4_TRAINING_COMPLETE",
        "completed": r3.now(), "completed_updates": MAX_UPDATES,
        "VAL_history": val_history,
        "field_stream_SHA256": field_stream_state.hex(),
        "diagonal_stream_SHA256": diagonal_stream_state.hex(),
        "best_update": best["update"], "gate": "PASS",
    })
    r3.atomic_json(TRAINING_HISTORY, history)
    r3.update_status(
        phase="H4_BEST_FROZEN", current_update=MAX_UPDATES,
        science_optimizer_updates=MAX_UPDATES,
        field_stream_status="MATCH_PASS", diag_stream_status="MATCH_PASS",
        H4_best_frozen=True, best_update=best["update"],
        best_VAL_pooled=best["pooled_VAL_complex_NRMSE"],
        best_VAL_max=best["max_per_column_VAL_complex_NRMSE"],
    )
    best_time = r3.record_event("H4_BEST_CHECKPOINT_FROZEN", best_update=best["update"], freeze_SHA256=freeze_sha)
    resource = {
        "training_wall_seconds": training_wall,
        "validation_wall_seconds": validation_wall,
        "training_and_validation_total_seconds": time.perf_counter() - started_total,
        "updates_per_second_training": MAX_UPDATES / max(training_wall, 1e-30),
        "peak_GPU_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_GPU_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "peak_system_RAM_bytes": int(peak_ram),
        "process_OS_read_bytes_delta_training": int(process.io_counters().read_bytes - io_start),
    }
    del optimizer, scaler
    return {
        "model": model, "best": freeze,
        "best_freeze_SHA256": freeze_sha, "best_time": best_time,
        "diagonal_coverage": final_coverage, "resource": resource,
        "val_history": val_history,
    }


def load_best(best: dict[str, Any], device: torch.device) -> torch.nn.Module:
    path = Path(best["checkpoint_path"])
    r3.require(r3.sha256(path) == best["checkpoint_SHA256"], "H4 best checkpoint changed")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_h4(MODEL_SEED).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def diagonal_self_consistency(model: torch.nn.Module, blocked: np.ndarray, device: torch.device) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(20261004))
    eligible = np.flatnonzero(~blocked).astype(np.uint32, copy=False)
    selected = rng.choice(eligible, size=65536, replace=False).astype(np.uint32, copy=False)
    values = []
    with torch.inference_mode():
        for start in range(0, selected.size, 32768):
            q = torch.from_numpy(n2.normalized_coordinates(n2.unravel_indices(selected[start:start + 32768]))).to(device)
            values.append(model(q, q).detach().cpu().numpy().astype(np.float64, copy=False))
    prediction = np.concatenate(values)
    real = prediction[:, 0]
    imag_abs = np.abs(prediction[:, 1])
    error = np.sqrt((real - 1.0) ** 2 + prediction[:, 1] ** 2)
    stats = lambda array: {
        "n": int(array.size), "min": float(np.min(array)),
        "median": float(np.median(array)), "max": float(np.max(array)),
        "mean": float(np.mean(array)), "std_population_ddof0": float(np.std(array, ddof=0)),
    }
    return {
        "classification": "P6_PHASEA8MRN2R8E_DIAGONAL_SELF_CONSISTENCY_COMPLETE",
        "completed": r3.now(), "role": "NETWORK_SELF_CONSISTENCY_DIAGNOSTIC_ONLY",
        "sample_seed": 20261004, "sample_count": 65536,
        "sample_domain": "uniform without replacement from C4-blacklist-safe native coordinates",
        "q_source_equals_q_output": True, "d_zero_exact": True,
        "predicted_diagonal_real": stats(real),
        "predicted_diagonal_imag_abs": stats(imag_abs),
        "absolute_Kqq_minus_1": stats(error),
        "new_exact_physics_accessed": False,
        "heldout_exact_diagonal_target_accessed": False,
        "used_for_checkpoint_selection": False,
        "prediction_arrays_saved": False, "gate": "PASS",
    }


def cancellation_retest(raw_summary: dict[str, Any]) -> dict[str, Any]:
    final = raw_summary["final_symmetrized_NRMSE"]["median"]
    raw_a = raw_summary["raw_branch_A_NRMSE"]["median"]
    raw_b = raw_summary["raw_branch_B_NRMSE"]["median"]
    raw_mean = 0.5 * (raw_a + raw_b)
    cosine = raw_summary["error_cosine"]["median"]
    checks = {
        "final_symmetrized_median": {
            "value": final, "rule": "<=", "threshold": 0.08,
            "gate": "PASS" if final <= 0.08 else "FAIL",
        },
        "raw_mean_vs_final": {
            "raw_A_median": raw_a, "raw_B_median": raw_b,
            "raw_mean_median": raw_mean, "final_median": final,
            "value_ratio": raw_mean / max(final, 1e-30), "rule": ">=",
            "threshold_multiplier": 1.5,
            "gate": "PASS" if raw_mean >= 1.5 * final else "FAIL",
        },
        "error_cosine_median": {
            "value": cosine, "rule": "<=", "threshold": -0.25,
            "gate": "PASS" if cosine <= -0.25 else "FAIL",
        },
    }
    supported = all(row["gate"] == "PASS" for row in checks.values())
    return {
        "classification": "P6_PHASEA8MRN2R8E_HIDDEN_BRANCH_CANCELLATION_RETEST_COMPLETE",
        "completed": r3.now(),
        "HIDDEN_BRANCH_CANCELLATION_SUPPORTED_AFTER_H4": supported,
        "checks": checks, "thresholds_unchanged": True, "gate": "PASS",
    }


def reduction_percent(baseline: float, value: float) -> float:
    return float(100.0 * (baseline - value) / max(abs(baseline), 1e-30))


def controlled_comparisons(
    h4_burned_summary: dict[str, Any],
    h4_burned_rows: list[dict[str, Any]],
    h4_internal_summary: dict[str, Any],
    h4_internal_raw: dict[str, Any],
    best: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    r8c = r3.read_json(R8C_SUMMARY)
    h3 = r8c["burned24_summary"]["pooled"]
    r3.require(h3["full_complex_NRMSE"]["median"] == H3_FULL_MEDIAN and h3["full_complex_NRMSE"]["max"] == H3_FULL_MAX, "Frozen H3 full baseline mismatch")
    r3.require(h3["total_energy_relative_error"]["median"] == H3_ENERGY_MEDIAN and h3["total_energy_relative_error"]["max"] == H3_ENERGY_MAX, "Frozen H3 energy baseline mismatch")
    r3.require(h3["boundary_region_complex_NRMSE"]["median"] == H3_BOUNDARY_MEDIAN and h3["boundary_region_complex_NRMSE"]["max"] == H3_BOUNDARY_MAX, "Frozen H3 boundary baseline mismatch")
    r3.require(h3["native_center_gain_relative_error"]["median"] == H3_GAIN_MEDIAN and h3["native_center_gain_relative_error"]["max"] == H3_GAIN_MAX, "Frozen H3 gain baseline mismatch")
    h4 = h4_burned_summary["pooled"]
    metric_map = {
        "full": "full_complex_NRMSE",
        "energy": "total_energy_relative_error",
        "boundary": "boundary_region_complex_NRMSE",
        "gain": "native_center_gain_relative_error",
    }
    comparison = {
        "classification": "P6_PHASEA8MRN2R8E_H4_VS_H3_CONTROLLED_COMPARISON_COMPLETE",
        "completed": r3.now(), "comparison": "H4 versus frozen H3 parent on identical burned24",
        "H3": {
            "full_median": H3_FULL_MEDIAN, "full_max": H3_FULL_MAX,
            "energy_median": H3_ENERGY_MEDIAN, "energy_max": H3_ENERGY_MAX,
            "boundary_median": H3_BOUNDARY_MEDIAN, "boundary_max": H3_BOUNDARY_MAX,
            "gain_median": H3_GAIN_MEDIAN, "gain_max": H3_GAIN_MAX,
            "peak_max": H3_PEAK_MAX,
        },
        "H4": {
            **{f"{label}_median": h4[metric]["median"] for label, metric in metric_map.items()},
            **{f"{label}_max": h4[metric]["max"] for label, metric in metric_map.items()},
            "peak_max": h4["peak_location_error_voxels"]["max"],
        },
        "H4_vs_H3_reduction_percent": {
            **{f"{label}_median": reduction_percent(h3[metric]["median"], h4[metric]["median"]) for label, metric in metric_map.items()},
            **{f"{label}_max": reduction_percent(h3[metric]["max"], h4[metric]["max"]) for label, metric in metric_map.items()},
        },
        "H3_best_VAL": {"update": r8c["best_checkpoint"]["best_update"], "pooled": r8c["best_checkpoint"]["pooled_VAL"], "max_column": r8c["best_checkpoint"]["max_column_VAL"]},
        "H4_best_VAL": {"update": best["best_update"], "pooled": best["pooled_VAL"], "max_column": best["max_column_VAL"]},
        "H4_final_VAL": best["final_update_VAL"],
        "H4_vs_H3_best_VAL_reduction_percent": {
            "pooled": reduction_percent(H3_BEST_VAL_POOLED, best["pooled_VAL"]),
            "max_column": reduction_percent(H3_BEST_VAL_MAX, best["max_column_VAL"]),
        },
        "H4_INTERNAL_TEST": h4_internal_summary["pooled"],
        "H4_INTERNAL_raw": h4_internal_raw,
        "diagonal_gain_improvement": h4["native_center_gain_relative_error"]["max"] < H3_GAIN_MAX,
        "full_median_improvement": h4["full_complex_NRMSE"]["median"] < H3_FULL_MEDIAN,
        "diagonal_gain_improvement_with_full_field_tradeoff": h4["native_center_gain_relative_error"]["max"] < H3_GAIN_MAX and h4["full_complex_NRMSE"]["median"] >= H3_FULL_MEDIAN,
        "H3_energy_gate_was_pass": True,
        "H4_energy_gate_preserved": h4["total_energy_relative_error"]["median"] <= 0.04 and h4["total_energy_relative_error"]["max"] <= 0.06,
        "H3_boundary_gate_was_pass": True,
        "H4_boundary_gate_preserved": h4["boundary_region_complex_NRMSE"]["median"] <= 0.10 and h4["boundary_region_complex_NRMSE"]["max"] <= 0.16,
        "burned24_role": "BURNED_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "used_for_checkpoint_selection": False, "retraining_triggered": False,
        "lambda_changed_after_result": False, "gate": "PASS",
    }
    top5 = sorted(
        ({
            "name": row["name"], "source_index_xyz": row["source_index_xyz"],
            "stratum": row["stratum"],
            "native_center_gain_relative_error": row["metrics"]["native_center_gain_relative_error"],
            "full_complex_NRMSE": row["metrics"]["full_complex_NRMSE"],
            "boundary_region_complex_NRMSE": row["metrics"]["boundary_region_complex_NRMSE"],
        } for row in h4_burned_rows),
        key=lambda row: (-row["native_center_gain_relative_error"], row["name"]),
    )[:5]
    gain = {
        "classification": "P6_PHASEA8MRN2R8E_GAIN_CONTROLLED_COMPARISON_COMPLETE",
        "completed": r3.now(), "comparison": "H4 versus frozen H3 parent; H2 secondary context",
        "H2": {"gain_median": H2_GAIN_MEDIAN, "gain_max": H2_GAIN_MAX},
        "H3": {"gain_median": H3_GAIN_MEDIAN, "gain_max": H3_GAIN_MAX},
        "H4": {
            "gain_median": h4["native_center_gain_relative_error"]["median"],
            "gain_max": h4["native_center_gain_relative_error"]["max"],
        },
        "H4_vs_H3_reduction_percent": {
            "gain_median": reduction_percent(H3_GAIN_MEDIAN, h4["native_center_gain_relative_error"]["median"]),
            "gain_max": reduction_percent(H3_GAIN_MAX, h4["native_center_gain_relative_error"]["max"]),
        },
        "H4_vs_H2_context_reduction_percent": {
            "gain_median": reduction_percent(H2_GAIN_MEDIAN, h4["native_center_gain_relative_error"]["median"]),
            "gain_max": reduction_percent(H2_GAIN_MAX, h4["native_center_gain_relative_error"]["max"]),
        },
        "top5_H4_gain_residual_sources": top5,
        "diagonal_loss_used": True, "lambda_diag": LAMBDA_DIAG,
        "used_for_checkpoint_selection": False, "retraining_triggered": False,
        "post_hoc_gain_correction": False, "gate": "PASS",
    }
    return comparison, gain


def classify_outcome(cancellation: bool, ready: bool, h4m: float, h4g: float) -> tuple[str, str, str]:
    tree = r3.read_json(R8D_OUTCOME)
    r3.require(tree["gate"] == "PASS" and [row["case"] for row in tree["ordered_tree"]] == ["D", "A", "B", "C", "E"], "R8D R8E outcome tree invalid")
    if cancellation:
        case = "D"
    elif ready:
        case = "A"
    elif h4g < H3_GAIN_MAX and h4m < H3_FULL_MEDIAN:
        case = "B"
    elif h4g < H3_GAIN_MAX and h4m >= H3_FULL_MEDIAN:
        case = "C"
    else:
        case = "E"
    frozen = {row["case"]: row for row in tree["ordered_tree"]}
    r3.require(frozen[case]["classification"] == CASE_CLASSES[case], "R8D case classification mismatch")
    r3.require(frozen[case]["recommended_next"] == CASE_NEXT[case], "R8D case next-stage mismatch")
    return case, CASE_CLASSES[case], CASE_NEXT[case]


def build_package() -> dict[str, Any]:
    sources = [
        REPORT, SUMMARY, STATUS, PROTOCOL_EXEC, INPUT_FREEZE, EXECUTION_FREEZE,
        MODEL_RUNTIME_AUDIT, DIAGONAL_RUNTIME_AUDIT, FIELD_STREAM_AUDIT,
        DIAGONAL_STREAM_AUDIT, TRAINING_HISTORY, BEST_FREEZE, INTERNAL_METRICS,
        CANCELLATION, BURNED_METRICS, BURNED_STRATA, H4_H3_COMPARISON,
        READINESS_OUTCOME,
    ]
    r3.require(len(sources) == 18 and all(path.is_file() for path in sources), "R8E final submission source set incomplete")
    r3.clear_submission()
    rows = []
    for source in sources:
        copied = SUBMISSION / source.name
        shutil.copy2(source, copied)
        source_hash = r3.sha256(source)
        copied_hash = r3.sha256(copied)
        rows.append({
            "original_path": str(source.resolve()), "copied_path": str(copied.resolve()),
            "bytes": copied.stat().st_size, "SHA256": copied_hash,
            "source_SHA256": source_hash, "hash_match": copied_hash == source_hash,
            "required_or_optional": "required",
            "purpose": "R8E formal controlled H4 development evidence artifact",
        })
    r3.require(all(row["hash_match"] for row in rows), "R8E submission hash mismatch")
    manifest = {
        "classification": "P6_PHASEA8MRN2R8E_SUBMISSION_PACKAGE_PASS",
        "created": r3.now(), "source_file_count": 18,
        "manifest_file_count": 2, "submission_total_files": 20,
        "files": rows, "all_hash_match": True,
        "results_only_artifacts": [
            str(GAIN_COMPARISON.resolve()), str(DIAGONAL_SELF.resolve()),
            str(INTERNAL_RAW.resolve()), str(BURNED_RAW.resolve()),
            str(RESOURCE.resolve()), str(ISOLATION.resolve()),
        ],
        "forbidden_artifacts_included": False, "gate": "PASS",
    }
    manifest_path = SUBMISSION / "本轮提交清单.json"
    r3.atomic_json(manifest_path, manifest)
    manifest_sha = r3.sha256(manifest_path)
    lines = [
        "P6_PHASEA8MRN2R8E submission manifest", "source files=18",
        "manifest files=2", "TOTAL=20",
        f"本轮提交清单.json SHA256={manifest_sha}",
        "all hash match=true", "",
    ]
    lines.extend(f"{row['SHA256']}  {Path(row['copied_path']).name}" for row in rows)
    (SUBMISSION / "本轮提交清单.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    r3.require(len(list(SUBMISSION.iterdir())) == 20, "R8E submission TOTAL is not 20")
    return {"TOTAL": 20, "manifest_SHA256": manifest_sha, "all_hash_match": True}


def finalize(
    training: dict[str, Any],
    roles: dict[str, list[dict[str, Any]]],
    blocked: np.ndarray,
    tracker: n2.AccessTracker,
    device: torch.device,
    total_started: float,
) -> dict[str, Any]:
    verify_execution_freeze()
    model = load_best(training["best"], device)
    best_time = training["best_time"]
    r3.require(r3.sha256(BEST_FREEZE) == training["best_freeze_SHA256"], "H4 best-freeze JSON changed")

    diagonal_self = diagonal_self_consistency(model, blocked, device)
    r3.write_new_json(DIAGONAL_SELF, diagonal_self)
    r3.record_event("DIAGONAL_NETWORK_SELF_CONSISTENCY_COMPLETE", exact_physics_access=0)

    test_first = r3.now()
    r3.require(best_time < test_first, "INTERNAL_TEST access did not follow H4 best freeze")
    r3.update_status(phase="INTERNAL_TEST_FIRST_ACCESS", INTERNAL_TEST_opened=True)
    r3.record_event("INTERNAL_TEST_FIRST_ACCESS", best_freeze_SHA256_verified=True)
    test_records = [{**row, "stratum": row["boundary_category"]} for row in roles["INTERNAL_TEST"]]
    test_rows, test_summary, test_raw_summary, test_runtime = r3.evaluate_streaming(model, test_records, "INTERNAL_TEST", tracker, device)
    r3.write_new_json(INTERNAL_RAW, {
        "classification": "P6_PHASEA8MRN2R8E_INTERNAL_TEST_RAW_BRANCH_METRICS_COMPLETE",
        "completed": r3.now(), "best_freeze_SHA256": training["best_freeze_SHA256"],
        "rows": [{"name": row["name"], "source_index_xyz": row["source_index_xyz"], "stratum": row["stratum"], "raw_branch_metrics": row["raw_branch_metrics"]} for row in test_rows],
        "summary": test_raw_summary, "prediction_arrays_saved": False, "gate": "PASS",
    })
    r3.write_new_json(INTERNAL_METRICS, {
        "classification": "P6_PHASEA8MRN2R8E_INTERNAL_TEST_COMPLETE",
        "completed": r3.now(), "role": "C4_ORBIT_HELDOUT_DEVELOPMENT_TEST",
        "best_freeze_SHA256": training["best_freeze_SHA256"],
        "first_access_timestamp": test_first, "access_after_best_freeze": True,
        "ranking_contribution": 0,
        "rows": [{key: value for key, value in row.items() if key != "raw_branch_metrics"} for row in test_rows],
        "summary": test_summary, "raw_branch_summary": test_raw_summary,
        "raw_branch_metrics_path": str(INTERNAL_RAW.resolve()),
        "raw_branch_metrics_SHA256": r3.sha256(INTERNAL_RAW),
        "runtime": test_runtime, "gate": "PASS",
    })
    r3.record_event("INTERNAL_TEST_EVALUATION_COMPLETE", columns=20)
    cancellation = cancellation_retest(test_raw_summary)
    r3.write_new_json(CANCELLATION, cancellation)
    cancel4 = cancellation["HIDDEN_BRANCH_CANCELLATION_SUPPORTED_AFTER_H4"]
    r3.update_status(CANCEL4=cancel4)
    cancel_time = r3.record_event("CANCELLATION_RETEST_COMPLETE", supported=cancel4)

    burned_manifest = r3.read_json(n2.BURNED_MANIFEST)
    burned_records = [{"name": row["name"], "index_xyz": row["index_xyz"], "stratum": row["stratum"], "path": row["exact_path"], "sha256": row["exact_SHA256"]} for row in burned_manifest["rows"]]
    burned_first = r3.now()
    r3.require(best_time < test_first < cancel_time < burned_first, "burned24 access order invalid")
    r3.update_status(phase="BURNED24_FIRST_ACCESS", burned_opened=True)
    r3.record_event("BURNED24_FIRST_ACCESS", best_freeze_SHA256_verified=True, INTERNAL_TEST_complete=True, cancellation_retest_complete=True)
    burned_rows, burned_summary, burned_raw_summary, burned_runtime = r3.evaluate_streaming(model, burned_records, "BURNED24", tracker, device)
    r3.write_new_json(BURNED_RAW, {
        "classification": "P6_PHASEA8MRN2R8E_BURNED24_RAW_BRANCH_METRICS_COMPLETE",
        "completed": r3.now(),
        "rows": [{"name": row["name"], "source_index_xyz": row["source_index_xyz"], "stratum": row["stratum"], "raw_branch_metrics": row["raw_branch_metrics"]} for row in burned_rows],
        "summary": burned_raw_summary, "prediction_arrays_saved": False, "gate": "PASS",
    })
    r3.write_new_json(BURNED_METRICS, {
        "classification": "P6_PHASEA8MRN2R8E_BURNED24_DEVELOPMENT_COMPLETE",
        "completed": r3.now(), "role": "BURNED_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "best_freeze_SHA256": training["best_freeze_SHA256"],
        "first_access_timestamp": burned_first,
        "access_after_best_freeze_INTERNAL_TEST_and_cancellation": True,
        "ranking_contribution": 0,
        "rows": [{key: value for key, value in row.items() if key != "raw_branch_metrics"} for row in burned_rows],
        "summary": burned_summary, "raw_branch_summary": burned_raw_summary,
        "raw_branch_metrics_path": str(BURNED_RAW.resolve()),
        "raw_branch_metrics_SHA256": r3.sha256(BURNED_RAW),
        "runtime": burned_runtime, "gate": "PASS",
    })
    r3.write_new_json(BURNED_STRATA, {
        "classification": "P6_PHASEA8MRN2R8E_BURNED24_SIX_STRATA_COMPLETE",
        "completed": r3.now(),
        "expected_strata": ["interior", "x_boundary_near", "y_boundary_near", "z_low", "z_high", "corner_or_mixed_boundary"],
        "strata": burned_summary["strata"], "strata_reclassified": False,
        "descriptive_only": True, "selection_contribution": 0, "gate": "PASS",
    })
    r3.record_event("BURNED24_EVALUATION_COMPLETE", columns=24)

    comparison, gain_comparison = controlled_comparisons(
        burned_summary, burned_rows, test_summary, test_raw_summary, training["best"],
    )
    r3.write_new_json(H4_H3_COMPARISON, comparison)
    r3.write_new_json(GAIN_COMPARISON, gain_comparison)
    readiness = r3.readiness_gate(burned_summary)
    pooled_test = test_summary["pooled"]
    pooled_burned = burned_summary["pooled"]
    h4_full = pooled_burned["full_complex_NRMSE"]["median"]
    h4_gain = pooled_burned["native_center_gain_relative_error"]["max"]
    case, classification, recommended = classify_outcome(cancel4, readiness["READY"], h4_full, h4_gain)
    outcome = {
        "classification": classification, "completed": r3.now(),
        "R8E_order": ["D", "A", "B", "C", "E"],
        "CANCEL4": cancel4, "READY4": readiness["READY"],
        "H4_full_median": h4_full, "H3_full_median": H3_FULL_MEDIAN,
        "H4_gain_max": h4_gain, "H3_gain_max": H3_GAIN_MAX,
        "outcome_case": case, "readiness": readiness,
        "strict_equalities": {"H4m_equals_H3m": "not improvement", "H4g_equals_H3g": "not improvement"},
        "interpretation_boundary": "H4 differs from H3 only by the preregistered TRAIN140-derived dense approximate unit-diagonal training loss; this remains development, not formal fresh validation.",
        "initial_weighted_diagonal_gradient_context": r3.read_json(DIAGONAL_RUNTIME_AUDIT)["initial_frozen_dummy_gradient_context"],
        "post_hoc_lambda_change": False, "warm_start_retrain": False,
        "hard_center_normalization": False, "column_renormalization": False,
        "gain_clipping": False, "recommended_next": recommended, "gate": "PASS",
    }
    r3.write_new_json(READINESS_OUTCOME, outcome)
    r3.record_event("READINESS_AND_R8E_OUTCOME_CLASSIFIED", READY4=readiness["READY"], CANCEL4=cancel4, case=case, classification=classification)

    checkpoints = sorted(CHECKPOINT_DIR.glob("H4_update_*.pt"))
    r3.require(len(checkpoints) == 40, "Expected 40 H4 checkpoints")
    lineage = r3.read_json(STATUS)["formal_lineage_id"]
    for path in checkpoints:
        header = torch.load(path, map_location="cpu", weights_only=False)
        r3.require(header["formal_lineage_id"] == lineage and header["candidate"] == "H4", "Checkpoint readability/lineage failure")
        del header
    process = psutil.Process(os.getpid())
    total_wall = time.perf_counter() - total_started
    resource = {
        "classification": "P6_PHASEA8MRN2R8E_RESOURCE_RUNTIME_COMPLETE",
        "completed": r3.now(), "preflight": r3.read_json(MODEL_RUNTIME_AUDIT)["resource_preflight"],
        "training": training["resource"], "INTERNAL_TEST": test_runtime,
        "burned24": burned_runtime, "total_wall_seconds": total_wall,
        "GPU_device": torch.cuda.get_device_name(0),
        "peak_GPU_allocated_bytes": training["resource"]["peak_GPU_allocated_bytes"],
        "peak_GPU_reserved_bytes": training["resource"]["peak_GPU_reserved_bytes"],
        "peak_GPU_reserved_GiB": training["resource"]["peak_GPU_reserved_bytes"] / (1024**3),
        "peak_reserved_limit_GiB": 7.5,
        "peak_reserved_gate": "PASS" if training["resource"]["peak_GPU_reserved_bytes"] / (1024**3) <= 7.5 else "FAIL",
        "peak_system_RAM_bytes": max(training["resource"]["peak_system_RAM_bytes"], process.memory_info().rss),
        "PSF_bytes_read": tracker.snapshot(), "checkpoint_count": len(checkpoints),
        "checkpoint_directory": str(CHECKPOINT_DIR.resolve()), "gate": "PASS",
    }
    r3.require(resource["peak_reserved_gate"] == "PASS", "R8E resource gate failed")
    r3.write_new_json(RESOURCE, resource)
    isolation = {
        "classification": "P6_PHASEA8MRN2R8E_ISOLATION_AND_ACCESS_ORDER_PASS",
        "completed": r3.now(), "VAL_gradient_contribution": 0,
        "INTERNAL_TEST_access_before_H4_freeze": False,
        "burned_access_before_H4_freeze": False,
        "burned_access_before_INTERNAL_TEST_completion": False,
        "burned_access_before_cancellation_retest": False,
        "burned_used_for_checkpoint_selection": False,
        "gain_used_for_checkpoint_selection": False,
        "TRAIN_diagonal_loss_used_for_checkpoint_selection": False,
        "gradient_ratio_used_for_checkpoint_selection": False,
        "raw_branch_used_for_checkpoint_selection": False,
        "strata_used_for_checkpoint_selection": False,
        "best_freeze_timestamp": best_time,
        "INTERNAL_TEST_first_access_timestamp": test_first,
        "cancellation_retest_complete_timestamp": cancel_time,
        "burned24_first_access_timestamp": burned_first,
        "timestamp_order_strict": best_time < test_first < cancel_time < burned_first,
        "TRAIN140_diagonal_affects_training": True,
        "prior_defined_from_TRAIN140_only": True,
        "VAL_TEST_burned_fresh30_used_to_define_prior": False,
        "fresh30_selected": 0, "fresh30_accessed": 0,
        "operator_access": False, "operator_cases": 0,
        "new_exact_physics": 0, "new_exact_PSF": 0,
        "new_complete_PSF_columns": 0, "anchor_densification": 0,
        "external_science_access": False, "Bunny_normal": 0,
        "Bunny_deconvolution": 0, "inverse_runs": 0,
        "NVR_updates": 0, "exact_F0_updates": 0,
        "prediction_arrays_saved": 0, "gate": "PASS",
    }
    r3.write_new_json(ISOLATION, isolation)

    summary = {
        "classification": classification, "completed": r3.now(),
        "stage": "CONTROLLED_H4_DEVELOPMENT_NOT_FORMAL_FRESH_VALIDATION",
        "candidate": CANDIDATE, "parent": "H3",
        "parameter_count": EXPECTED_PARAMS, "input_dimension": 165,
        "initial_state_SHA256": EXPECTED_INITIAL_SHA,
        "boundary_features": list(BOUNDARY_FEATURE_NAMES),
        "boundary_slots": list(BOUNDARY_AUXILIARY_GLOBAL_INDICES),
        "source_x_retained_k": [0, 1, 2], "source_y_retained_k": [0, 1, 2],
        "source_z_retained_k": [0, 1], "source_z_k2_zero": True,
        "source_k4_to_k7_zero": True, "output_k0_to_k7_unchanged": True,
        "relative_k0_to_k9_unchanged": True,
        "unique_science_change": "TRAIN140-derived dense approximate unit-diagonal consistency",
        "field_loss": "0.5*MSE_complex(A,y)+0.5*MSE_complex(B,y)",
        "diagonal_loss": "0.5*MSE_complex(A_diag,1+0j)+0.5*MSE_complex(B_diag,1+0j)",
        "lambda_diag": LAMBDA_DIAG, "total_loss": "L_field+0.1*L_diag",
        "diagonal_target": "1+0j",
        "diagonal_pairs_per_update": DIAGONAL_PAIRS_PER_UPDATE,
        "completed_updates": MAX_UPDATES,
        "field_stream_SHA256": EXPECTED_FIELD_STREAM_SHA,
        "field_stream_match": True,
        "diagonal_stream_SHA256": EXPECTED_DIAGONAL_STREAM_SHA,
        "diagonal_stream_match": True,
        "diagonal_coverage": training["diagonal_coverage"],
        "best_checkpoint": training["best"],
        "H4_vs_H3_controlled_comparison": comparison,
        "gain_controlled_comparison": gain_comparison,
        "INTERNAL_TEST_summary": test_summary,
        "INTERNAL_TEST_raw_summary": test_raw_summary,
        "cancellation_retest": cancellation,
        "burned24_summary": burned_summary,
        "burned24_raw_summary": burned_raw_summary,
        "six_strata": burned_summary["strata"],
        "diagonal_self_consistency": diagonal_self,
        "readiness_and_outcome": outcome,
        "resource_runtime": resource, "access_order": isolation,
        "runtime_gradient_decomposition": "SKIPPED_TO_PRESERVE_SCIENCE_TRAJECTORY",
        "fresh30_selected": 0, "fresh30_accessed": 0,
        "operator_cases": 0, "new_physics": 0,
        "neural_field_development_ready": case == "A",
        "formal_fresh_validation_passed": False,
        "recommended_next": recommended,
    }
    r3.write_new_json(SUMMARY, summary)
    strata_lines = "\n".join(
        f"- {name}: full {metrics['full_complex_NRMSE']['median']:.8g}/{metrics['full_complex_NRMSE']['max']:.8g}; "
        f"gain {metrics['native_center_gain_relative_error']['median']:.8g}/{metrics['native_center_gain_relative_error']['max']:.8g}; "
        f"energy {metrics['total_energy_relative_error']['median']:.8g}/{metrics['total_energy_relative_error']['max']:.8g}; "
        f"boundary {metrics['boundary_region_complex_NRMSE']['median']:.8g}/{metrics['boundary_region_complex_NRMSE']['max']:.8g}"
        for name, metrics in burned_summary["strata"].items()
    )
    delta = comparison["H4_vs_H3_reduction_percent"]
    gain_delta = gain_comparison["H4_vs_H3_reduction_percent"]
    self_real = diagonal_self["predicted_diagonal_real"]
    self_imag = diagonal_self["predicted_diagonal_imag_abs"]
    self_error = diagonal_self["absolute_Kqq_minus_1"]
    dummy = r3.read_json(DIAGONAL_RUNTIME_AUDIT)["initial_frozen_dummy_gradient_context"]
    readiness_lines = "\n".join(f"- {name}: value={row['value']:.10g}, threshold {row['rule']} {row['threshold']:.10g}, {row['gate']}" for name, row in readiness["checks"].items())
    report = f"""# P6 Phase-A8MRN2R8E Train-only Diagonal Gain-consistency Development

Classification: {classification}
R8E outcome: CASE {case}
Recommended next: {recommended}

## Frozen controlled experiment

H4 completed exactly 20000 optimizer updates with parent H3, 414210 parameters, 165-dimensional input, fresh initialization, unchanged boundary features/source masks, dual-orientation field supervision, Hermitian inference, frozen 140/20/20 split, field sampler/C4 stream, optimizer, and budget. The sole science change was TRAIN140-derived dense approximate unit-diagonal consistency: 1024 deterministic post-C4 blacklist-safe q per update, (q,q,d=0), target 1+0j, and L_total=L_field+0.1*L_diag.

Field stream SHA {EXPECTED_FIELD_STREAM_SHA} and diagonal stream SHA {EXPECTED_DIAGONAL_STREAM_SHA} both matched. Diagonal replay coverage was {training['diagonal_coverage']['unique_coordinates']}/{training['diagonal_coverage']['native_domain_count']}={100*training['diagonal_coverage']['coverage_fraction']:.8g}%.

Best update {training['best']['best_update']}; best pooled/max VAL {training['best']['pooled_VAL']:.10g}/{training['best']['max_column_VAL']:.10g}; final pooled/max VAL {training['best']['final_update_VAL']['pooled']:.10g}/{training['best']['final_update_VAL']['max_column']:.10g}. Frozen H3 best pooled/max VAL was {H3_BEST_VAL_POOLED:.10g}/{H3_BEST_VAL_MAX:.10g}; H4-vs-H3 best VAL reduction was {comparison['H4_vs_H3_best_VAL_reduction_percent']['pooled']:.6g}%/{comparison['H4_vs_H3_best_VAL_reduction_percent']['max_column']:.6g}%.

## INTERNAL_TEST and cancellation

Full median/max {pooled_test['full_complex_NRMSE']['median']:.10g}/{pooled_test['full_complex_NRMSE']['max']:.10g}; energy {pooled_test['total_energy_relative_error']['median']:.10g}/{pooled_test['total_energy_relative_error']['max']:.10g}; boundary {pooled_test['boundary_region_complex_NRMSE']['median']:.10g}/{pooled_test['boundary_region_complex_NRMSE']['max']:.10g}; gain {pooled_test['native_center_gain_relative_error']['median']:.10g}/{pooled_test['native_center_gain_relative_error']['max']:.10g}; peak max {pooled_test['peak_location_error_voxels']['max']:.10g}.

Raw A/B/disagreement medians {test_raw_summary['raw_branch_A_NRMSE']['median']:.10g}/{test_raw_summary['raw_branch_B_NRMSE']['median']:.10g}/{test_raw_summary['raw_branch_disagreement']['median']:.10g}; error cosine {test_raw_summary['error_cosine']['median']:.10g}; cancellation ratio {test_raw_summary['cancellation_ratio']['median']:.10g}. CANCEL4={cancel4}.

## burned24 gain/full trade-off

Full median/max {pooled_burned['full_complex_NRMSE']['median']:.10g}/{pooled_burned['full_complex_NRMSE']['max']:.10g}; energy {pooled_burned['total_energy_relative_error']['median']:.10g}/{pooled_burned['total_energy_relative_error']['max']:.10g}; boundary {pooled_burned['boundary_region_complex_NRMSE']['median']:.10g}/{pooled_burned['boundary_region_complex_NRMSE']['max']:.10g}; gain {pooled_burned['native_center_gain_relative_error']['median']:.10g}/{pooled_burned['native_center_gain_relative_error']['max']:.10g}; peak max {pooled_burned['peak_location_error_voxels']['max']:.10g}.

H4-vs-H3 full median/max reduction={delta['full_median']:.6g}%/{delta['full_max']:.6g}%; gain median/max reduction={gain_delta['gain_median']:.6g}%/{gain_delta['gain_max']:.6g}%. H4-vs-H2 gain context reduction={gain_comparison['H4_vs_H2_context_reduction_percent']['gain_median']:.6g}%/{gain_comparison['H4_vs_H2_context_reduction_percent']['gain_max']:.6g}%. H3 energy gates were PASS and H4 preservation={comparison['H4_energy_gate_preserved']}; H3 boundary gates were PASS and H4 preservation={comparison['H4_boundary_gate_preserved']}.

{strata_lines}

## Network diagonal self-consistency

For 65536 deterministic blacklist-safe native coordinates, predicted real mean/median/std/min/max={self_real['mean']:.10g}/{self_real['median']:.10g}/{self_real['std_population_ddof0']:.10g}/{self_real['min']:.10g}/{self_real['max']:.10g}; imag-abs median/max={self_imag['median']:.10g}/{self_imag['max']:.10g}; |K(q,q)-1| median/max={self_error['median']:.10g}/{self_error['max']:.10g}. No exact diagonal target or new physics was accessed.

## Readiness and gradient interpretation

{readiness_lines}

READY4={readiness['READY']}. Outcome uses the frozen D->A->B->C->E order.

Initial frozen dummy field/raw-diag/weighted-diag gradient norms={dummy['field_grad_norm']:.10g}/{dummy['raw_diag_grad_norm']:.10g}/{dummy['weighted_diag_grad_norm']:.10g}; weighted-diag/field ratio={dummy['weighted_diag_grad_over_field_grad']:.10g}. This is a coefficient-low auxiliary prior whose initial weighted gradient magnitude is comparable to or larger than the field gradient on the frozen dummy audit. It was not a stopping or retuning gate. Runtime gradient decomposition was skipped to preserve the science trajectory.

## Isolation

The H4 best freeze preceded INTERNAL_TEST; cancellation retest completed before burned24 first access. No post-hoc lambda change, warm-start retrain, hard-coded center, column renormalization, or gain clipping occurred. No new exact PSF, densification, fresh30 selection/access, operator case, Bunny deconvolution, NVR, or exact-F0 was executed.

H4以H3为parent，保持414210参数、165维输入、boundary features、source masks、dual-orientation field supervision、Hermitian inference、140/20/20 split、field sample/C4 stream、optimizer与20k budget。

唯一science change为TRAIN140-derived dense approximate unit-diagonal consistency：每update从已有post-C4 blacklist-safe field queries确定性抽取1024个q，构造(q,q,d=0)，target=1+0j，使用 L_total=L_field+0.1 L_diag。

这个prior在20k coordinate replay中覆盖约99.937%的native source coordinates，因此它是dense global diagonal prior，但不是新的exact off-lattice physics label。

没有使用VAL、INTERNAL_TEST、burned24或fresh30定义prior；没有hard-code center；没有column normalization；没有gain clipping；没有lambda sweep。

没有生成新exact PSF，没有densification，没有fresh30，没有operator，没有Bunny/NVR/exact-F0。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    summary["report_path"] = str(REPORT.resolve())
    summary["report_SHA256"] = r3.sha256(REPORT)
    r3.atomic_json(SUMMARY, summary)
    r3.update_status(
        status="FINALIZED", phase="FINALIZED", current_update=MAX_UPDATES,
        internal_full_median=pooled_test["full_complex_NRMSE"]["median"],
        internal_gain_max=pooled_test["native_center_gain_relative_error"]["max"],
        internal_rawA_median=test_raw_summary["raw_branch_A_NRMSE"]["median"],
        internal_rawB_median=test_raw_summary["raw_branch_B_NRMSE"]["median"],
        internal_disagreement_median=test_raw_summary["raw_branch_disagreement"]["median"],
        internal_error_cosine=test_raw_summary["error_cosine"]["median"],
        internal_cancellation_ratio=test_raw_summary["cancellation_ratio"]["median"],
        CANCEL4=cancel4,
        burned_full_median=pooled_burned["full_complex_NRMSE"]["median"],
        burned_full_max=pooled_burned["full_complex_NRMSE"]["max"],
        burned_energy_median=pooled_burned["total_energy_relative_error"]["median"],
        burned_energy_max=pooled_burned["total_energy_relative_error"]["max"],
        burned_boundary_median=pooled_burned["boundary_region_complex_NRMSE"]["median"],
        burned_boundary_max=pooled_burned["boundary_region_complex_NRMSE"]["max"],
        burned_gain_median=pooled_burned["native_center_gain_relative_error"]["median"],
        burned_gain_max=pooled_burned["native_center_gain_relative_error"]["max"],
        burned_peak_max=pooled_burned["peak_location_error_voxels"]["max"],
        READY4=readiness["READY"], outcome_case=case,
        classification=classification, recommended_next=recommended,
        neural_field_development_ready=case == "A",
        formal_fresh_validation_passed=False,
    )
    package = build_package()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "classification": classification, "outcome_case": case,
        "READY4": readiness["READY"], "CANCEL4": cancel4,
        "recommended_next": recommended, "submission": package,
    }


def run(resume: bool, device: torch.device) -> dict[str, Any]:
    configure_reused_runtime()
    total_started = time.perf_counter()
    if not resume:
        prepare_fresh(device)
    else:
        r3.require(STATUS.is_file() and EXECUTION_FREEZE.is_file() and WORK_DIR.is_dir(), "No resumable R8E lineage")
        verify_execution_freeze()
        r3.require(r3.read_json(STATUS)["status"] in ("RUNNING", "INTERRUPTED_RESUMABLE"), "R8E status is not resumable")
        r3.update_status(status="RUNNING", phase="RESUME_RUNTIME_RESTORE", PID=os.getpid(), CommandLine=commandline(True))
    r3.update_status(phase="TRAIN_RUNTIME_RESTORE")
    roles, _ = n2.source_records()
    blocked = n2.blacklist_mask()
    tracker = n2.AccessTracker()
    stats, train_maps, shells = n2.restore_train_runtime(roles["TRAIN"], blocked)
    r3.require(stats["g_ref"] == G_REF and stats["TRAIN_source_columns"] == 140, "N2 TRAIN statistics mismatch")
    probes = n2.decode_val_probes(roles["VAL"])
    val_cache = n2.ValidationCache(roles["VAL"], probes, tracker)
    resume_path = latest_checkpoint() if resume else None
    if resume:
        r3.require(resume_path is not None, "No 500-update checkpoint exists; restarting a second lineage is forbidden")
    training = train_h4(device, roles, blocked, stats, train_maps, shells, val_cache, tracker, resume_path)
    return finalize(training, roles, blocked, tracker, device, total_started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    args = parser.parse_args()
    r3.require(args.run, "Formal execution requires --run")
    r3.require(torch.cuda.is_available(), "CUDA unavailable")
    device = torch.device("cuda:0")
    try:
        result = run(args.resume, device)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    except BaseException as error:
        if STATUS.is_file():
            try:
                resumable = CHECKPOINT_DIR.is_dir() and any(CHECKPOINT_DIR.glob("H4_update_*.pt"))
                r3.update_status(
                    status="INTERRUPTED_RESUMABLE" if resumable else "FAILED",
                    phase="INTERRUPTED" if resumable else "FAILED",
                    error_type=type(error).__name__, error_message=str(error),
                    traceback=traceback.format_exc(),
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
