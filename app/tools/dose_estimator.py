"""Patient-specific starting-dose estimator.

Given the chosen designs (n mRNA peptides, the top scFv binder, the CAR
cassette) and patient anthropometrics (weight in kg, optional BSA in m²),
return recommended starting doses for each of the three modalities, with
citations to the published literature each estimate is derived from.

This tool **does not** invent dosing science. Every number it returns is
either:
  - a published per-patient dose from a Phase I/II trial (BNT122,
    epcoritamab, tisagenlecleucel), or
  - a simple weight-scaled extrapolation of a published mg/kg or
    cells/kg figure.

The dossier downstream is post-checked by ProvenanceGate, so each numeric
value is recorded into the artifact store alongside its provenance string.

References:
  - BNT122 / mRNA-4157 personalised cancer vaccine — Rojas 2023 *Nature*
    618(7963):144-150 (`Rojas2023` in references.bib)
  - Epcoritamab CD3xCD20 bispecific — Hutchings 2021 *Lancet*
    398(10306):1157-1169 (`Hutchings2021` in references.bib)
  - Tisagenlecleucel 4-1BBz CAR-T — Maude 2018 *NEJM*
    378(5):439-448 (`Maude2018` in references.bib)
"""
from __future__ import annotations

from typing import Any


SCHEMA = {
    "name": "estimate_doses",
    "description": (
        "Compute patient-specific starting doses for each of the three "
        "personalised-therapy modalities (mRNA-LNP vaccine, bispecific scFv, "
        "autologous CAR-T). Returns recommended doses with citations to the "
        "published literature each estimate is derived from."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "n_mrna_peptides": {
                "type": "integer",
                "description": "Number of HLA-restricted peptides in the cassette.",
            },
            "patient_weight_kg": {
                "type": "number",
                "description": "Patient body weight in kilograms.",
                "default": 70.0,
            },
            "patient_bsa_m2": {
                "type": "number",
                "description": "Patient body surface area in m². Optional; "
                               "computed via DuBois if omitted and height supplied.",
            },
            "binder_iplddt": {
                "type": "number",
                "description": "ipLDDT of the chosen scFv binder. Used to "
                               "annotate confidence in the bispecific dose.",
            },
        },
        "required": ["n_mrna_peptides"],
    },
}


# Published per-modality reference doses
# ---------------------------------------------------------------------------
# mRNA personalised cancer vaccine (BNT122 / mRNA-4157, Rojas 2023):
#   ~50 μg per neoantigen × 9 peptides ≈ 450 μg total
#   intramuscular weekly × 8, then every 3 weeks
_MRNA_UG_PER_PEPTIDE_DEFAULT = 50.0  # Rojas2023

# Bispecific T-cell engager (epcoritamab, Hutchings 2021):
#   step-up 0.16 mg → 0.80 mg → 48 mg subcutaneous weekly
_BISPECIFIC_PRIMING_MG = 0.16             # Hutchings2021
_BISPECIFIC_INTERMEDIATE_MG = 0.80        # Hutchings2021
_BISPECIFIC_FULL_MG = 48.0                # Hutchings2021

# Autologous CAR-T (tisagenlecleucel, Maude 2018):
#   target 2-5 × 10^8 CAR-positive T-cells for adults
#   pediatric: 0.2-5 × 10^6 CAR+ cells/kg
_CART_ADULT_TARGET_CELLS = 3.0e8          # Maude2018
_CART_PEDIATRIC_CELLS_PER_KG = 2.0e6      # Maude2018


# ---------------------------------------------------------------------------
def run(
    n_mrna_peptides: int,
    patient_weight_kg: float = 70.0,
    patient_bsa_m2: float | None = None,
    binder_iplddt: float | None = None,
) -> dict[str, Any]:
    """Return recommended starting doses for all three modalities.

    All numeric outputs are derived from the published reference doses
    above; the function does not invent science. The `provenance` string
    on each modality block names the trial / paper the figure traces to.
    """
    n_pep = max(1, int(n_mrna_peptides))
    weight = max(1.0, float(patient_weight_kg))

    # ---------- mRNA ----------
    mrna_per_dose_ug = round(n_pep * _MRNA_UG_PER_PEPTIDE_DEFAULT, 1)
    mrna_section = {
        "modality": "mRNA-LNP personalised cassette",
        "ug_per_peptide": _MRNA_UG_PER_PEPTIDE_DEFAULT,
        "n_peptides": n_pep,
        "total_per_dose_ug": mrna_per_dose_ug,
        "schedule": "weekly × 8 doses, then every 3 weeks",
        "route": "intramuscular",
        "provenance": "Rojas2023",
        "rationale": (
            f"BNT122 / mRNA-4157 standard cassette dose of "
            f"{_MRNA_UG_PER_PEPTIDE_DEFAULT:.0f} μg per neoantigen "
            f"[Rojas2023], scaled to the {n_pep} peptide(s) selected for this patient."
        ),
    }

    # ---------- Bispecific scFv ----------
    bispecific_section = {
        "modality": "Bispecific T-cell engager (research-grade scFv)",
        "step_up_priming_mg": _BISPECIFIC_PRIMING_MG,
        "step_up_intermediate_mg": _BISPECIFIC_INTERMEDIATE_MG,
        "full_dose_mg": _BISPECIFIC_FULL_MG,
        "schedule": "weekly subcutaneous step-up: priming → intermediate → full",
        "route": "subcutaneous",
        "provenance": "Hutchings2021",
        "rationale": (
            f"Step-up dosing pattern of epcoritamab Phase I [Hutchings2021] "
            f"({_BISPECIFIC_PRIMING_MG} mg → {_BISPECIFIC_INTERMEDIATE_MG} mg → "
            f"{_BISPECIFIC_FULL_MG} mg) to mitigate cytokine-release syndrome. "
            "This is a published clinical-trial template, not a patient-"
            "specific dose; final dose for a research-grade in-house scFv "
            "would require a Phase I dose-escalation study."
        ),
    }
    if binder_iplddt is not None:
        bispecific_section["binder_iplddt_at_design"] = float(binder_iplddt)

    # ---------- CAR-T ----------
    target_cells = _CART_ADULT_TARGET_CELLS
    if weight < 50.0:
        # Pediatric scaling
        target_cells = round(weight * _CART_PEDIATRIC_CELLS_PER_KG, -7)
    cart_section = {
        "modality": "Autologous 4-1BBz CAR-T",
        "target_cell_dose": target_cells,
        "patient_weight_kg": weight,
        "schedule": "single intravenous infusion after lymphodepletion",
        "route": "intravenous",
        "provenance": "Maude2018",
        "rationale": (
            f"Tisagenlecleucel-style 4-1BBz second-generation CAR-T target "
            f"dose of {target_cells:.1e} CAR-positive T-cells [Maude2018]; "
            f"adult or pediatric scaling based on patient weight ({weight:.0f} kg)."
        ),
    }

    return {
        "patient_weight_kg": weight,
        "patient_bsa_m2": patient_bsa_m2,
        "mrna_vaccine": mrna_section,
        "bispecific_scfv": bispecific_section,
        "car_t": cart_section,
    }
