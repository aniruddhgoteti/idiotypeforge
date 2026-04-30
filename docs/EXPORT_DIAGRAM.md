# Exporting the architecture diagram

The editable source for the architecture diagram lives at
[`docs/architecture.excalidraw`](architecture.excalidraw). The README references
the rasterised PNG at `docs/architecture.png` (and an SVG fallback at
`docs/architecture.svg`), which need to be regenerated whenever the diagram
changes.

## Quick export — web UI (no install)

1. Open <https://excalidraw.com>.
2. Drag-and-drop `docs/architecture.excalidraw` into the canvas, **or** click
   the hamburger menu → **Open** → choose the file.
3. Tweak as needed (move boxes, change colours, add text).
4. Export:
   - **PNG** — File → Export image → choose **PNG** → uncheck "Background" if
     you want a transparent background → check "Embed scene" so the export can
     be re-imported as editable later → "Save to..." `docs/architecture.png`.
   - **SVG** — same flow, choose **SVG** → save to `docs/architecture.svg`.
5. Commit both the updated `.excalidraw` source and the regenerated
   `architecture.png` / `architecture.svg`.

Recommended export settings: scale **2×**, with background.

## Optional — VS Code extension

Install the [Excalidraw VS Code
extension](https://marketplace.visualstudio.com/items?itemName=pomdtr.excalidraw-editor)
and open `docs/architecture.excalidraw` directly inside the editor. Same export
options as the web UI.

## CLI export (advanced)

There is no first-party Excalidraw CLI yet, but
[`@excalidraw/excalidraw-cli`](https://github.com/excalidraw/excalidraw-cli) and
the `excalidraw_export` npm package can rasterise headlessly. For one-off
exports the web UI is faster.

## Where the PNG is referenced

- `README.md` — the "How it works under the hood" section embeds
  `docs/architecture.png`. If the file is missing, the README will show a
  broken-image icon — that's the cue to re-export.

## Tips for staying on-brand

- **Colour palette** (already baked into the source):
  - Agent (Gemma 4): amber `#fef3c7` / `#d97706`
  - Tools: blue `#dbeafe` / `#1e40af`
  - Verification gates: red `#fee2e2` / `#b91c1c`, with the ProvenanceGate in
    a slightly stronger shade (`#fecaca` / `#dc2626`) to draw the eye
  - Output / dossier: green `#dcfce7` / `#15803d`
  - Deliverables: purple `#ede9fe` / `#7c3aed`
- **Font**: Excalidraw's default "Hand-drawn" (Virgil) font — keep it. It
  signals "thoughtful design", not "auto-generated diagram".
- **Roughness**: keep at 1 for the hand-drawn look. Setting roughness to 0
  produces a sterile rectilinear diagram that loses the Excalidraw aesthetic.
- **Avoid emojis on production exports**: most browsers render the
  Excalidraw-canvas emojis fine, but in a video YouTube re-encoding can muddy
  them. If shipping for video, render as 4× scale and verify in the target
  player.
