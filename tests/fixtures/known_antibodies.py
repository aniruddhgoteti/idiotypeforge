"""Reference antibody sequences with known Kabat CDRs.

Used as golden fixtures for the ANARCI numbering tests. Sequences are
public, redistributable (PDB-derived).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Rituximab — anti-CD20 (Mabthera / Rituxan). PDB 2OSL.
# Kabat CDR3-H = "STYYGGDWYFNV"  (12 aa)
# Kabat CDR3-L = "QQWTSNPPT"     (9 aa)
# ---------------------------------------------------------------------------
RITUXIMAB_VH = (
    "QVQLQQPGAELVKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATL"
    "TADKSSSTAYMQLSSLTSEDSAVYYCARSTYYGGDWYFNVWGAGTTVTVSA"
)
RITUXIMAB_VL = (
    "QIVLSQSPAILSASPGEKVTMTCRASSSVSYIHWFQQKPGSSPKPWIYATSNLASGVPVRFSGSGSGTSY"
    "SLTISRVEAEDAATYYCQQWTSNPPTFGGGTKLEIK"
)
RITUXIMAB_CDR3_H = "STYYGGDWYFNV"
RITUXIMAB_CDR3_L = "QQWTSNPPT"


# ---------------------------------------------------------------------------
# Trastuzumab — anti-HER2 (Herceptin). PDB 1N8Z.
# Kabat CDR3-H = "WGGDGFYAMDY"   (11 aa)
# Kabat CDR3-L = "HYTTPPT"       (7 aa)
# ---------------------------------------------------------------------------
TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTI"
    "SADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTD"
    "FTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)
TRASTUZUMAB_CDR3_H = "WGGDGFYAMDY"
TRASTUZUMAB_CDR3_L = "HYTTPPT"
