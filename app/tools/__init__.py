"""Deterministic tools the Gemma 4 agent can call via native function calling.

Each tool is implemented as a stand-alone module with:
  - a `run(...)` function that takes typed Python args and returns a JSON-
    serialisable dict
  - a `SCHEMA` constant (OpenAI-style JSON schema) registered in
    `app.agent.router` so Gemma 4 can discover the tool

Tools are split into two cohorts:

CPU-runnable (work on a laptop with no GPU):
  - number_antibody       (ANARCI)
  - igfold_predict        (CPU mode, ~2–5 min/seq)
  - cdr_liabilities
  - mhcflurry_predict
  - offtarget_search      (MMseqs2 + BLAST system binaries)
  - car_assembler
  - render_structure      (headless PyMOL or NGL)
  - compose_dossier

GPU-required (real implementation needs CUDA; ship with deterministic mock):
  - rfdiffusion_design
  - rescore_complex       (AlphaFold-Multimer / Boltz-2)

The mocks are gated by env var `IDIOTYPEFORGE_USE_MOCKS=1` (default for local
methodology validation). Days 13–14 of the build set the var to 0 and run the
real GPU implementations on GCP A100 spot.
"""
