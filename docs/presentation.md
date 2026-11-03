# Presentation plan (30 minutes)

Follows the course format: introduction (5 min), implementation (5), errors and lessons (5), live demo
(10), Q&A (5). If you present as a team, give each member one block of about 10 minutes. Figures marked
TBD come from the full run (see `docs/report.md`).

## 1. Task and methods - 5 min

**Slide 1 - Title and question.** *How does a BERT-style model that learns sequences of log-template
IDs compare with a rule-guided LLM that reads the log text?*

**Slide 2 - Why logs.** A raw BGL line with its ten fields, and the alert tag highlighted. Point out that
millions of lines are produced, only 7.3% are alerts, and nobody can read them by hand.

**Slide 3 - From lines to templates to windows.** Drain turns `generating core.2275` into
`generating core.<*>` (template ID = "log key"). Show 5-minute windows sliding by 1 minute, and the rule
that a window is anomalous if it contains any alert.

**Slide 4 - Approach A, LogBERT.** A small Transformer trained from scratch on normal key sequences,
with two tasks: masked-key prediction and the hypersphere loss. Detection: more than *r* masked keys
outside the top-*g* guesses means anomalous.

**Slide 5 - Approach B, LogRules-style.** Induction: an LLM writes keyword rules from labeled
templates. Verification keeps only rules that are never wrong on the training data. Deduction runs four
conditions: zero-shot, few-shot, rule-based, and keyword-only (no LLM).

## 2. Implementation - 5 min

**Slide 6 - Pipeline diagram.** `download -> preprocess -> {train/evaluate LogBERT | induce rules ->
4 conditions} -> compare -> analyze -> demo`. One YAML config; `make smoke` runs it on a laptop in a minute.

**Slide 7 - The fairness choices** (the core of the project):
- a chronological split, never random (Le & Zhang, 2022)
- one evaluation unit: identical test windows for every method, with metrics recomputed by one script
  that rejects mismatched windows
- alert tags removed before any model sees the text
- g and r tuned on validation, not on test

**Slide 8 - Key code** (show three short snippets):
- `logad/data/windows.py`: vectorised sliding windows, plus the test against LogBERT's original loop
- `logad/logbert/detector.py`: storing ranks so the whole (g, r) grid costs one forward pass
- `logad/rules/induction.py`: `verify_rules`, which turns LLM suggestions into trusted rules

**Slide 9 - Results table** (from `outputs/bgl/results/comparison.md`): precision, recall, F1, false
positives and inference time for A, A+ and B1-B4, plus the template-level oracle. TBD.

## 3. Errors, debugging and lessons - 5 min

**Slide 10 - Reading the reference code critically.** The LogBERT repository tunes its threshold on the
test set and silently skips short test sequences. Show what our numbers would look like with the same
shortcuts if you have time to run it (TBD), or simply state the risk.

**Slide 11 - Random masking misses things.** In the demo an inserted alert was sometimes never masked,
so LogBERT never looked at it. This led to the full-coverage extension (show A vs. A+).

**Slide 12 - Template purity.** Drain merges some alert and normal messages (`data <*> <*>`), so no
template-level method can exceed the oracle bound. Show the share of alerts in mixed templates (TBD).

**Slide 13 - Lessons.** Small samples mislead: on the 2k sample LogBERT flags almost everything. Write
tests against the original code. Cache every LLM call. Report the time spent, not just F1.

## 4. Live demo - 10 min

Start before the talk: `python app/demo.py --share` (or the last cell of the Colab notebook), and keep
the link on a slide so classmates can try it on their phones.

| Step | Input | What to point out |
|---|---|---|
| 1 | Paste the full line `KERNDTLB 1118536327 2005.06.11 R30-M0-N9-C:J16-U01 2005-06-11-17.32.07.581048 R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt` | The alert tag is removed before analysis (label leakage). Both approaches flag it. Show the rule that fired and LogBERT's rank for the line. |
| 2 | `instruction cache parity error corrected` | Contains "error" but is **normal**. Did the keyword rules learn not to use "error" alone? (Verification rejects rules that are wrong on training data.) |
| 3 | `ciod: LOGIN chdir(/p/gb1/stella/RAPTOR/2183) failed: Input/output error` vs. `ciod: LOGIN /p/gb1/x failed: No such file or directory` | Nearly identical text, different labels (alert vs. normal). A hard case for keywords. Does the LLM separate them? |
| 4 | Tab 2: pick a normal window, insert `rts panic! - stopping execution` at different positions | LogBERT's verdict changes with context; compare the "Without your line" count. |
| 5 | Ask the audience for a made-up line, e.g. `fan module 3 failed on node card R12-M0` | An unseen template: LogBERT sees `<unk>` (always a miss), while the LLM reasons about the text. |

Fallback if the network fails: run `python app/demo.py --no-live-llm` locally. Saved LLM decisions
still cover every template from the experiment.

## 5. Q&A - 5 min (prepared answers)

- **Why not compare with the published numbers?** The papers use different evaluation units and samples
  (DeepLog: 86.12 vs. 16.8 F1 on BGL). Only a shared setup answers the question.
- **Isn't giving the LLM labeled examples unfair to LogBERT?** LogBERT is semi-supervised by design (it
  uses only normal data); Approach B uses labels during induction. That is the trade-off being studied,
  and the zero-shot condition shows the LLM without any labels.
- **Why template level for the LLM?** Cost: a few hundred templates instead of millions of messages. The
  oracle row quantifies what is lost.
- **Did you reproduce LogRules fully?** No. Its CPO alignment stage (fine-tuning the 8B model) is out of
  scope; we reproduce induction and deduction.
- **How sensitive is LogBERT to g and r?** Show the validation F1 grid (`logbert_tuning_grid.csv`).
- **Could the two approaches be combined?** Check the agreement table: if they catch different alerts,
  a union or vote may help (future work).
