"""Tests for the patient-specific starting-dose estimator.

The estimator is fully deterministic (no external services), so all tests
run on the base install with no skips.
"""
from __future__ import annotations

import pytest

from app.tools.dose_estimator import run as estimate_doses


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------
def test_returns_three_modalities() -> None:
    out = estimate_doses(n_mrna_peptides=3, patient_weight_kg=70)
    assert "mrna_vaccine" in out
    assert "bispecific_scfv" in out
    assert "car_t" in out


def test_each_modality_has_provenance_and_rationale() -> None:
    out = estimate_doses(n_mrna_peptides=3, patient_weight_kg=70)
    for key in ("mrna_vaccine", "bispecific_scfv", "car_t"):
        block = out[key]
        assert "provenance" in block, f"{key} missing provenance"
        assert "rationale" in block, f"{key} missing rationale"


# ---------------------------------------------------------------------------
# mRNA — scales linearly with peptide count
# ---------------------------------------------------------------------------
def test_mrna_dose_scales_linearly_with_peptide_count() -> None:
    one = estimate_doses(n_mrna_peptides=1)["mrna_vaccine"]["total_per_dose_ug"]
    three = estimate_doses(n_mrna_peptides=3)["mrna_vaccine"]["total_per_dose_ug"]
    five = estimate_doses(n_mrna_peptides=5)["mrna_vaccine"]["total_per_dose_ug"]
    assert three == pytest.approx(3 * one)
    assert five == pytest.approx(5 * one)


def test_mrna_uses_published_per_peptide_dose() -> None:
    """50 μg per peptide is the BNT122 / mRNA-4157 standard (Rojas2023)."""
    out = estimate_doses(n_mrna_peptides=1)["mrna_vaccine"]
    assert out["ug_per_peptide"] == 50.0
    assert out["provenance"] == "Rojas2023"


def test_mrna_floors_at_one_peptide() -> None:
    """Even if 0 peptides come in, the cassette still gets a 1-peptide minimum dose."""
    out = estimate_doses(n_mrna_peptides=0)["mrna_vaccine"]
    assert out["n_peptides"] == 1


# ---------------------------------------------------------------------------
# Bispecific — uses Hutchings 2021 step-up
# ---------------------------------------------------------------------------
def test_bispecific_step_up_matches_published() -> None:
    out = estimate_doses(n_mrna_peptides=3)["bispecific_scfv"]
    assert out["step_up_priming_mg"] == 0.16
    assert out["step_up_intermediate_mg"] == 0.80
    assert out["full_dose_mg"] == 48.0
    assert out["provenance"] == "Hutchings2021"


def test_bispecific_records_binder_iplddt_when_supplied() -> None:
    out = estimate_doses(n_mrna_peptides=3, binder_iplddt=0.83)["bispecific_scfv"]
    assert out["binder_iplddt_at_design"] == 0.83


def test_bispecific_omits_iplddt_when_not_supplied() -> None:
    out = estimate_doses(n_mrna_peptides=3)["bispecific_scfv"]
    assert "binder_iplddt_at_design" not in out


# ---------------------------------------------------------------------------
# CAR-T — adult vs pediatric scaling
# ---------------------------------------------------------------------------
def test_cart_uses_adult_dose_for_adults() -> None:
    out = estimate_doses(n_mrna_peptides=3, patient_weight_kg=70)["car_t"]
    assert out["target_cell_dose"] == 3.0e8
    assert out["patient_weight_kg"] == 70.0


def test_cart_scales_to_pediatric_when_under_50kg() -> None:
    out = estimate_doses(n_mrna_peptides=3, patient_weight_kg=20)["car_t"]
    # Pediatric: 2e6 cells/kg × 20 kg = 4e7
    assert out["target_cell_dose"] == 4.0e7
    assert out["patient_weight_kg"] == 20.0


def test_cart_provenance_is_maude2018() -> None:
    out = estimate_doses(n_mrna_peptides=3)["car_t"]
    assert out["provenance"] == "Maude2018"


# ---------------------------------------------------------------------------
# Defensive
# ---------------------------------------------------------------------------
def test_zero_or_negative_weight_floors_at_one_kg() -> None:
    """A pathological zero/negative weight shouldn't break the pipeline."""
    out = estimate_doses(n_mrna_peptides=3, patient_weight_kg=0)
    assert out["patient_weight_kg"] >= 1.0


def test_default_weight_is_70kg() -> None:
    out = estimate_doses(n_mrna_peptides=3)
    assert out["patient_weight_kg"] == 70.0
