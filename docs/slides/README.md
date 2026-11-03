# The presentation

A 30-minute talk on the project, following the course format: introduction (5 minutes),
implementation (5), errors and lessons (5), live demo (10) and questions (5).

| File | What it is |
|---|---|
| `presentation.pptx` | the slide deck (21 slides). Every slide has speaker notes: what to say, what to point at, backup answers and the transition to the next slide. |
| `presentation.pdf` | the same slides as a PDF, for viewing on GitHub or printing (without the notes) |
| `speaker_notes.md` | all the speaker notes as one readable script, for rehearsing |
| `build_deck.js` | builds `presentation.pptx` and `speaker_notes.md` (pptxgenjs) |
| `diagrams.md`, `render_diagrams.js`, `diagrams/` | the Mermaid sources of the slide diagrams and their rendered images |

## Presenting

Open `presentation.pptx` in PowerPoint (or Keynote or Google Slides) and use **Presenter View**: the
audience sees the slides, you see the notes and a timer. Each note starts with its time budget:

| Block | Slides | Time |
|---|---|---|
| Introduction: the problem, the question, the two approaches, the data | 1-6 | 5 min |
| Implementation: LogBERT, the LLM conditions, the rules, the pipeline, the results | 7-12 | 5 min |
| Errors and lessons: why each approach behaves as it does, what went wrong, lessons | 13-18 | 5 min |
| Live demo | 19 | 10 min |
| Conclusion and questions | 20-21 | 5 min |

Before the talk, prepare the demo on the presenting machine (see slide 19's notes):

```bash
make data preprocess
```

```bash
make restore OUT=outputs/bgl_cpu
```

```bash
python app/demo.py --config configs/bgl_cpu.yaml --share
```

Add your name on the title slide; its notes begin with "Hello, my name is [your name]".

## Rebuilding

The numbers on the slides come from the reported run: the charts read
`docs/figures/figure_data.json` (written by `scripts/make_figures.py`) and `results/`. After editing
`build_deck.js` or `diagrams.md`:

```bash
npm install
```

```bash
node render_diagrams.js && node build_deck.js
```

`render_diagrams.js` needs Playwright with Chromium (`npx playwright install chromium`). To refresh
the PDF, export `presentation.pptx` to PDF from PowerPoint, or run
`soffice --headless --convert-to pdf presentation.pptx`.
