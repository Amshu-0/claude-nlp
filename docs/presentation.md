# Presentation plan (30 minutes)

Follows the course format: introduction (5 min), implementation (5), errors and lessons (5), live demo
(10), Q&A (5). If you present as a team, give each member one block of about 10 minutes. All figures
come from the full-BGL CPU run (`docs/report.md`, `results/`).

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

**Slide 9 - Results table** (from `results/comparison.md`, test windows, 495 of 5,742 anomalous):

| Method | P | R | F1 | FP | Time |
|---|---:|---:|---:|---:|---:|
| A LogBERT / A+ full coverage | 0.174 / 0.184 | 0.921 / 0.897 | 0.293 / 0.305 | 2,159 / 1,975 | 78 / 156 s |
| B1 LLM zero-shot | 0.108 | 1.000 | 0.194 | 4,103 | 47 min |
| B2 LLM few-shot | 0.323 | 0.941 | 0.481 | 976 | 53 min |
| B3 LLM rule-based | 0.489 | 0.943 | 0.644 | 488 | 57 min |
| B4 keyword rules, no LLM | 1.000 | 0.915 | 0.956 | 0 | 8 ms |

Say it in one line: the rules win, and they win without the LLM.

**Slide 10 - Why.** Three findings, one picture each:
- *Drift hurts LogBERT:* 361 of 472 test templates are new; 99.8% of its false-positive windows
  contain one.
- *The LLM reads the rules' sentences, not their keywords:* in only 20 of 420 citations does the cited
  rule's keyword occur in the template. Example: rule 4 ("Messages mentioning 'FAILED' …", keywords
  `lustre mount failed`, …) applied to `ciod: Error loading … Permission denied`.
- *Ranking keeps near-duplicates:* 12 of the top 30 rules are about "terminated"; `socket closed` sits at
  rank 32, just outside, and is the keyword rules' biggest miss.

## 3. Errors, debugging and lessons - 5 min

**Slide 11 - Reading the reference code critically.** The LogBERT repository tunes its threshold on the
test set and silently skips short test sequences. Choosing g and r on the test windows, as it does,
would raise our LogBERT F1 from 0.293 to 0.475; tuning on validation keeps the honest number.

**Slide 12 - Random masking misses things.** In the demo an inserted alert was sometimes never masked,
so LogBERT never looked at it. This led to the full-coverage extension (A vs. A+: F1 0.293 → 0.305).

**Slide 13 - Template purity and label drift.** Within the test period only 31 of 28,915 alerts sit in
mixed templates (oracle F1 0.996). Across periods it is worse: `ciod: LOGIN <*> failed: <*> <*>` is
`Permission denied` (normal) in training but `Input/output error` (816 alerts) in test. Drain masked
exactly the words that carry the label.

**Slide 14 - Lessons.** Small samples mislead: on the 2k sample LogBERT flags almost everything. Write
tests against the original code. Cache every LLM call (it saved the run when the machine restarted).
Check that an LLM applies the rules it cites. Report the time spent, not just F1.

## 4. Live demo - 10 min

Start before the talk with the reported run's model and decisions (after `make data preprocess` and
`make restore OUT=outputs/bgl_cpu`, see the README): `python app/demo.py --config configs/bgl_cpu.yaml
--share`. Keep the link on a slide so classmates can try it on their phones.

Each line is inserted into the same normal test window, which already contains messages the
rule-based LLM flags falsely. Its window verdict therefore reads "anomalous" before you add anything,
so read the "Your line: …" part of each verdict.

| Step | Input | What to point out |
|---|---|---|
| 1 | Paste the full line `KERNDTLB 1118536327 2005.06.11 R30-M0-N9-C:J16-U01 2005-06-11-17.32.07.581048 R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt` | The alert tag is removed before analysis (label leakage). The keyword rules flag it (rule 24). LogBERT never saw this key in a normal window (`<unk>`, rank 176), so it counts as a miss, but one miss does not flag the window (it needs more than r = 8). |
| 2 | `instruction cache parity error corrected` | Contains "error" but is **normal**. The keyword rules leave it alone (verification rejects rules that are wrong on training data); the zero-shot LLM called it abnormal in our run (33,547 false-alarm messages). |
| 3 | `ciod: LOGIN chdir(/p/gb1/stella/RAPTOR/2183) failed: Input/output error` vs. `ciod: LOGIN /p/gb1/x failed: No such file or directory` | Nearly identical text, different labels (alert vs. normal). Drain maps the alert to `ciod: LOGIN <*> failed: <*> <*>`, so the model never sees "Input/output error". In our run the few-shot and rule-based LLM got both lines backwards. |
| 4 | Tab 2: pick a normal window, insert `rts panic! - stopping execution` at different positions | LogBERT's verdict changes with context; compare the "Without your line" count. |
| 5 | Ask the audience for a made-up line, e.g. `fan module 3 failed on node card R12-M0` | An unseen template: LogBERT sees `<unk>` (always a miss), while the LLM reasons about the text (live, so the model server must be running; otherwise only the keyword rules judge it). |

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
- **How sensitive is LogBERT to g and r?** Very: only 16 of 231 (g, r) settings come within 0.02 of the
  best validation F1 (`results/logbert/logbert_tuning_grid.csv`).
- **Is k = 30 rules cherry-picked?** It is LogRules' value, fixed in the proposal. On validation k = 45
  is best, and it would have given test F1 0.980 (`results/analysis/keyword_rule_sensitivity.csv`).
- **Why is the LLM worse than its own rules?** It follows the rules' sentences, which are broader than
  the verified keywords, so it fires on templates no keyword matches: 5 real alerts, but 23 false alarms.
- **Could the two approaches be combined?** They do miss different windows (18 only LogBERT, 15 only the
  keyword rules), but "either" keeps LogBERT's 2,159 false positives (F1 0.301) and "both" scores 0.939,
  below the rules alone. Only the rule-based LLM gains from requiring LogBERT to agree (0.644 → 0.776).
