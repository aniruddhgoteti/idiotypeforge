# IdiotypeForge — 3-minute YouTube shotlist

> **Total runtime: 3:00 max.** Kaggle judging is 40 % Impact & Vision, 30 %
> Video Pitch & Storytelling, 30 % Technical Depth. The video is where most
> of those 100 points live. Optimise for emotional clarity first; demo
> footage second; "look how clever the architecture is" last.

---

## Narrative arc

A → B → C structure:
- **A. The wall** (0:00–1:00) — set up *why this matters* with a real human
  story, not statistics.
- **B. The breakthrough** (1:00–2:15) — show the live demo: a BCR sequence
  in, a complete therapy-design dossier (with doses) out, in real time.
  This is the "wow" cut.
- **C. The future** (2:15–3:00) — close the loop: what this changes for the
  patients on the wrong side of the access curve, and the open invitation
  to build on it.

---

## Shot-by-shot

### A. The wall — 0:00 to 1:00

**Shot 1 · 0:00 – 0:08 · Opening hook**
- B-roll: hands on an infusion-chair armrest; IV pole in shallow focus.
- VO: *"In 2024, I sat in a chemo chair next to a man whose lymphoma had a
  textbook treatment menu I would never see."*

**Shot 2 · 0:08 – 0:25 · The disparity**
- Cuts: ABVD bag → DHAP infusion → pembrolizumab vial → autologous SCT room.
- VO: *"My menu was four lines: ABVD, DHAP, pembrolizumab + GVD, autologous
  stem cell transplant. My survival was a function of access, not biology."*
- (Subtle on-screen text: "Stage IV Hodgkin · 2024.")

**Shot 3 · 0:25 – 0:50 · The wall**
- B-roll: clipboard, marker board with PTCL · ATLL · post-CAR-T relapse
  · primary CNS lymphoma · Burkitt-LMIC.
- VO: *"For a peripheral T-cell lymphoma patient: 30 % five-year survival.
  For a Burkitt patient in sub-Saharan Africa: under 30 %. For a relapsed
  DLBCL patient after CAR-T: six months median survival. The biology of
  personalised therapy was solved in 1985 — every B-cell lymphoma patient
  carries a perfect target on every cancer cell. The reason it never
  reached them was the cost of designing a custom drug per patient."*

**Shot 4 · 0:50 – 1:00 · Pivot**
- On-screen title card (Excalidraw architecture frame, slow zoom):
  **"IdiotypeForge — from a tumour BCR to a personalised therapy dossier
  in under two hours."**

---

### B. The breakthrough — 1:00 to 2:15 (live demo)

This is the dense central minute. **Screen-record the Gradio dashboard live
at 60 fps; speed up to 2× where useful, but never fake the agent log.**

**Shot 5 · 1:00 – 1:15 · The input**
- Screen: `localhost:7860`, Gradio UI.
- Click **"CLL — Subset #2 (Stamatopoulos 2017)"** demo button.
- VO: *"This is one published lymphoma BCR sequence. 116 amino acids of
  heavy chain. 108 amino acids of light chain. The patient's HLA type.
  Three fields — that's all the input."*

**Shot 6 · 1:15 – 1:50 · The agent runs**
- Screen-record the streaming tool-call log (4× speed, captions visible).
- VO: *"Gemma 4 — open-weight, running locally on a laptop with no cloud
  call — orchestrates ten deterministic scientific tools: ANARCI to locate
  the CDR3 idiotype, IgFold to predict the 3D structure, MHCflurry for HLA
  epitopes, RFdiffusion to design custom binders, AlphaFold-Multimer to
  rescore them, the Observed Antibody Space to screen for off-target hits."*
- Subtle on-screen lower thirds: tool icons appearing as each step fires.

**Shot 7 · 1:50 – 2:15 · The output**
- Screen: dossier panel scrolling — slow tilt down through:
  - 1 BCR fingerprint (`ARDANGMDV`)
  - 2 mRNA vaccine peptides (table)
  - 3 designed bispecific binders (decision card)
  - 4 CAR-T cassette
  - 5 safety summary
  - 6 manufacturing brief
  - 7 **dosing**: 150 μg mRNA cassette · 0.16 → 0.80 → 48 mg bispecific
    step-up · 3 × 10⁸ CAR-positive T-cells
- VO: *"Two hours later, on a single laptop: three personalised therapy
  designs. An mRNA vaccine cassette. A bispecific antibody. A CAR-T
  construct. Patient-specific starting doses for each, traced to the
  Phase-I dosing of analogous published trials. A safety screen against
  five million normal antibodies. And — critically — a verification audit
  that proves every number in the dossier came from a deterministic tool
  call. No hallucinated science."*

---

### C. The future — 2:15 to 3:00

**Shot 8 · 2:15 – 2:35 · The verification story (anti-hallucination)**
- Screen: the Verification Audit panel — three green ticks lighting up:
  ✅ MockModeGate · ✅ CitationGate · ✅ ProvenanceGate.
- VO: *"This is the difference between a research demo and an auditable
  scientific artifact. A clinician — or a sceptical regulator — can re-run
  the harness on this dossier and see, for every numeric claim, exactly
  which tool call produced it. If Gemma 4 invents a number, the gate
  rejects the dossier."*

**Shot 9 · 2:35 – 2:50 · The access pivot**
- B-roll mosaic: PTCL ward in São Paulo, Burkitt clinic in Kampala,
  haematology unit in Manila — neutral, dignified, not pity-bait.
- VO: *"The wall that protected the lucky stops condemning the unlucky.
  This pipeline runs on a single laptop, behind a hospital firewall,
  with no cloud bill. Open weights. Open source. CC-BY 4.0. Every line
  of code reproducible by anyone, anywhere."*

**Shot 10 · 2:50 – 3:00 · The close**
- Title card on white: **github.com/aniruddhgoteti/idiotypeforge**.
- Below: *"Built for the Kaggle Gemma 4 Good Hackathon. Open-source forever."*
- VO: *"The biology was right in 1985. The logistics were wrong. We can
  fix the logistics now."*

---

## Production notes

### What to record
- 60 fps screen capture of the Gradio dashboard end-to-end on `cll_subset2`
  (~2 minutes of raw footage)
- A static still of the Excalidraw architecture diagram, plus a slow
  Ken-Burns zoom version
- B-roll: hospital infusion chair, marker board, clipboard, IV pole — these
  can be stock or shot quickly in any healthcare-adjacent setting. **Do
  not** use stock photos of identifiable patients.

### What to write
- VO: ~280 words total. Record at a measured pace — 95 wpm average — so
  there is silence/breath to let the demo footage breathe. **Do not rush**.
  If you run over, cut from Shot 1 (the personal opener) before cutting
  from Shot 6 (the demo).
- Lower-third tool-name captions as each agent step fires — viewer needs
  to *see* tools being called, not just hear about them.
- On-screen labels for every key number that will become a video
  thumbnail candidate: "150 μg mRNA cassette", "verification PASSED",
  "0 % off-target".

### What NOT to do
- **No stock medical voiceover.** First-person is the differentiator;
  every other hackathon submission will sound like a Khan Academy clip.
- **No false claims.** Don't say "this could cure lymphoma" — say "this
  could finally make personalised therapy economic for the patients who
  need it most."
- **No mock screen-recordings.** The agent log must be real. Judges will
  click through to GitHub and verify.
- **No ABVD/DHAP brand-name medical visuals you don't have rights to.**
  Stylise the cuts (silhouette, hand-held shots, blurred labels) if in
  doubt.

### Music
- One under-bed track, soft piano or low ambient pad — **no swelling
  string crescendos**. The story is the engine; the music is the floor.
  Suggested: Ólafur Arnalds "Saman", Nils Frahm "Says", or a public-
  domain piano piece. Avoid YouTube's licence-flagged catalogue.

### Cover image / thumbnail
- 1280 × 720 PNG.
- Left half: a microscope/H&E shot with a six-month calendar overlay
  (BiovaxID baseline). Right half: a laptop with the IdiotypeForge UI
  visible and a "1:47:32" timer overlay.
- Title bar across: **"From tumour sample to therapy dossier in hours."**
- Source the calendar / laptop in Figma; use Excalidraw for the diagram
  inset.

---

## Submission checklist

When the video is ready:

- [ ] Upload to YouTube as **public** (not unlisted).
- [ ] Title: *"IdiotypeForge — Personalised Lymphoma Therapy Design with
      Gemma 4 (Kaggle Hackathon Submission)"*
- [ ] Description includes: link to GitHub, link to HF Space, link to
      Kaggle writeup, list of open-source tools used.
- [ ] Captions / subtitles uploaded (.srt) — the live demo footage must be
      readable for judges who may have audio off.
- [ ] Cover image uploaded as the YouTube thumbnail.
- [ ] Embed link added to the Kaggle Writeup's media gallery.
- [ ] Final pre-submission test: open the YouTube link in a private
      browsing window with no Google login. Should play instantly.

---

## One-liner pitch (for the writeup)

*"IdiotypeForge collapses the design step of personalised lymphoma therapy
from six months to two hours, using ten open-source scientific tools
orchestrated by Gemma 4 — and a five-gate verification system that proves
every number in the dossier came from a real tool call, not a
hallucination. It runs on a laptop, with no cloud dependency. The
biology was right in 1985; the logistics were wrong; we can fix them now."*
