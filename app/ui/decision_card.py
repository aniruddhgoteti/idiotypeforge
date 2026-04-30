"""Decision card renderer.

One card per top-3 binder candidate. Each card is a self-contained
markdown block that drops into the dossier and the Gradio UI.
"""
from __future__ import annotations

from typing import Any


def render_card(
    candidate_id: str,
    sequence: str,
    iplddt: float,
    ipae: float,
    interface_sasa: float,
    off_target_max_id: float,
    calibrated_p_binder: float,
    rationale: str,
    structure_png_b64: str | None = None,
) -> str:
    """Return a markdown block for a single binder candidate."""
    img_md = (
        f"![structure](data:image/png;base64,{structure_png_b64})\n"
        if structure_png_b64
        else "_(structure render unavailable)_"
    )
    return f"""
### Candidate `{candidate_id}`

{img_md}

| metric | value |
|---|---|
| length | {len(sequence)} aa |
| ipLDDT | {iplddt:.3f} |
| iPAE | {ipae:.2f} Å |
| interface SASA | {interface_sasa:.0f} Å² |
| off-target max identity | {off_target_max_id:.1f} % |
| **calibrated P(binder)** | **{calibrated_p_binder:.2f}** |

**Sequence**
```text
{sequence}
```

**Rationale.** {rationale}
""".strip()
