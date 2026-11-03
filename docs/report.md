# Comparing BERT-Based and Rule-Guided Language Models for System Log Anomaly Detection

**Technical report (draft)** · CSCI 390 Natural Language Processing and Applications · Dr. Qi Li

> **Status of this draft.** Approach A results come from the full-BGL run on a 4-core CPU
> (`configs/bgl_cpu.yaml`). Approach B is running; its rows are marked **TBD** until then, to be
> copied from `results/comparison.md` and `results/analysis/analysis.md`.

## 1. Problem formulation

Large computer systems write millions of log messages; most record routine events, a few signal
failures. We study **window-level log anomaly detection**: given all messages a system printed in a
5-minute window, decide whether the window contains at least one alert. Anomalous is the positive class.

**Research question.** How does a BERT-style model that learns patterns in *sequences of log-template
IDs* (LogBERT; Guo et al., 2021) compare with a *rule-guided large language model that reads the log
text itself* (inspired by LogRules; Huang et al., 2025) in identifying anomalous system events?

The two papers cannot be compared through their published numbers: they use different evaluation units
and data samples (DeepLog, for example, reaches 86.12 F1 on BGL in the LogBERT paper but 16.8 F1 in the
LogRules paper). We therefore re-run both approaches **under one shared setup**: the same dataset, the
same chronological split and the same test windows, scored by the same code.

## 2. Dataset and preprocessing

**BGL** (Oliner & Stearley, 2007), from Loghub (Zhu et al., 2023): 4,747,963 messages from the
BlueGene/L supercomputer at Lawrence Livermore National Laboratory, of which 348,460 (7.3%) carry an
alert tag. Each line has ten fields; the first is the **alert tag** (`-` for normal messages, otherwise
a category such as `KERNDTLB` or `APPREAD`).

**Data source.** Zenodo, Loghub's official host, was unreachable from our development environment.
We therefore used a Git LFS copy of `BGL.log` from a public GitHub repository and verified that it is
Loghub's file. Its SHA-256 (`666130b1…a5d2`) matches the copy's LFS record. It has exactly 4,747,963
lines and 348,460 alert lines, and it contains all 2,000 lines of Loghub's official `BGL_2k` sample.

**Preventing label leakage.** The alert tag is the ground truth. `logad/data/bgl.py` splits it off
into a label column while reading the file. Every model reads only the message *content* (or Drain
templates built from it), so no method can see the label. A test checks that tags never appear in the
model-visible text.

**Template extraction.** Drain (He et al., 2017) groups messages printed by the same statement and
replaces variable parts with `<*>` (e.g. `generating core.2275` becomes `generating core.<*>`). We use the
`drain3` implementation with the LogBERT repository's BGL settings: depth 3, similarity threshold 0.3,
and masking of hexadecimal numbers, IPv4 addresses and integers. drain3's depth semantics match
logparser's Drain, which LogBERT uses: depth 3 groups messages by length only. Clusters whose final
template text is identical share one ID, as in logparser. On full BGL, Drain forms 2,028 clusters
that end with 923 distinct templates; preprocessing takes 199 s on 4 CPU cores. On Loghub's
2,000-line sample we obtain 100 templates against 120 in Loghub's hand-labelled ground truth. Some
over-merging is visible (e.g. `data <*> <*>` mixes storage-interrupt alerts with normal messages).
Ten templates are mixed in this way, covering 2.74% of all messages.

**Chronological split.** The timeline is cut into the first 60% (training), the next 20% (validation)
and the last 20% (test). We never shuffle: Le & Zhang (2022) showed that random splits inflate
semi-supervised detectors such as LogBERT. Both approaches share this split.

**Shared evaluation unit.** As in LogBERT, messages are grouped into **5-minute windows that slide by
1 minute**, built separately inside each split so no window mixes periods. A window is anomalous if it
contains at least one alert. This is LogBERT's convention (`max(label_data[start:end])` in its
`sliding_window`), and our vectorised implementation is tested against a line-by-line port of that
function.

| Split | Period (UTC) | Messages | Alerts | Windows | Anomalous windows | Templates (unseen in training) |
|---|---|---:|---:|---:|---:|---:|
| Train | 2005-06-03 to 2005-10-10 | 3,569,472 | 296,569 | 21,112 | 2,083 | 442 (0) |
| Validation | 2005-10-10 to 2005-11-22 | 696,160 | 22,976 | 10,478 | 446 | 375 (167) |
| Test | 2005-11-22 to 2006-01-04 | 482,331 | 28,915 | 5,742 | 495 | 472 (361) |

**The log changes over time.** 79.6% of test messages use a template never seen in training. The two
largest are periodic diagnostics, `<*> total interrupts. …` and `<*> microseconds spent in the rbs
signal handler …`, with 135,092 test messages each and none before October 2005. As a result, 71.1% of
*normal* test windows contain at least one unseen template, against 9.6% of normal validation windows.
This is the "new log events" problem Le & Zhang (2022) describe for chronological splits. It hits
LogBERT hardest, because an unseen key is always a miss, while an LLM can read that such a message is
benign. Only 172 distinct keys occur in normal training windows, which bounds LogBERT's vocabulary.

## 3. Methods

### 3.1 Approach A: LogBERT (reproduction)

LogBERT is not a pretrained language model. It is a small Transformer encoder trained from scratch on
sequences of log keys (template IDs) from **normal training windows only**.

| Component | Setting |
|---|---|
| Input | `[<dist>, k1, ..., kn]`; windows longer than 511 keys are split into chunks |
| Encoder | 2 pre-norm Transformer layers, hidden size 256, 4 heads, GELU feed-forward of size 1024, dropout 0.1, sinusoidal positions |
| Vocabulary | keys seen in normal training windows; unseen keys become `<unk>` |
| Objective | MLKP: cross-entropy on masked keys (50% masked; 80/10/10 replacement) **+ 0.1 × VHM**: MSE between each `<dist>` vector and the hypersphere centre, re-estimated every epoch |
| Optimisation | AdamW, learning rate 1e-3 with 500 warm-up steps, batch 64, gradient clipping 1.0, early stopping (patience 5) on a held-out tail of 10% of normal training windows |
| Detection | mask keys and rank the true key among the model's predictions; a chunk is anomalous if **more than r** masked keys fall outside the **top-g**; a window is anomalous if any chunk is |
| Tuning | g ∈ {1, ..., 20} and r ∈ {0, ..., 20} chosen by F1 on the **validation** windows |

Because only the rank of each true key is stored, the whole (g, r) grid is evaluated from a single
forward pass. Unseen (`<unk>`) keys can never be predicted and therefore always count as misses.

**How this reproduction relates to the reference code.** We checked every point below against the
source of the LogBERT repository (`HelenGuohx/logbert`, branch `main`, read on 2026-11-03).

We match its BGL options for the window (5 minutes, step 1), mask ratio (0.5), `min_len` (10),
sequence length (512), hidden size (256), attention heads (4), learning rate (1e-3), dev fraction (0.1)
and the hypersphere loss (`nn.MSELoss`, added as `loss + 0.1 * hyper_loss`). The differences:

| Aspect | LogBERT repository (file) | This project | Why |
|---|---|---|---|
| Train/test selection | normal windows shuffled (`sample(frac=1, random_state=12)`); 40% train, the other 60% plus every abnormal window test (`BGL/data_process.py`) | chronological 60/20/20 | Le & Zhang (2022); deployment realism |
| Threshold selection | `find_best_threshold(test_normal_results, test_abnormal_results, …)` searches the **test** results (`bert_pytorch/predict_log.py`) | g and r tuned on validation, applied once to test | avoids test-set tuning |
| Short test windows | `fixed_window` returns nothing below `min_len`, and `generate_test` skips it (`dataset/sample.py`, `predict_log.py`) | every window is evaluated | both approaches must score the same windows |
| Long windows | truncated with `line[:seq_len]` (`dataset/sample.py`) | split into chunks | messages after position 512 would otherwise go unchecked |
| Decision rule | `undetected_tokens > masked_tokens * seq_threshold` (`predict_log.py`) | undetected keys > r | the paper's formulation, as in the proposal |
| Encoder depth | `layers = 4` (`BGL/logbert.py`) | 2 layers | the paper's description, as in the proposal |
| Training budget | batch 32, up to 200 epochs, early-stopping patience 10 (`BGL/logbert.py`) | batch 64, up to 50 epochs, patience 5 | compute: the full run trains on a CPU |
| Hypersphere centre | mean over the train and validation loaders each epoch (`bert_pytorch/train_log.py`) | mean over (a sample of) training sequences each epoch | keeps the validation period untouched |

**The reported run.** We trained on 4 CPU cores with `configs/bgl_cpu.yaml`, which keeps the
architecture above. The 35,035 normal training chunks contain only 12,394 distinct sequences, so the
run de-duplicated them: 11,615 training and 820 dev sequences, a vocabulary of 172 keys and 1,670,320
parameters. The centre was estimated from 4,000 sequences and training ran at most 6 epochs. It took
60 minutes, and the dev loss fell from 1.59 to 1.02 (epoch 5, kept). A GPU run with
`configs/bgl.yaml` (up to 50 epochs, no de-duplication) may score differently.

**Extension (full-coverage masking).** LogBERT masks a random 50% of the keys once, so a single
anomalous message is inspected with probability of only about 0.5. Our `full_coverage` mode splits the
positions into ⌈1/mask ratio⌉ disjoint groups and masks each in turn, so every key is checked exactly
once, at twice the inference cost. It is reported as a separate method, `logbert_full`.

### 3.2 Approach B: Rule-guided LLM (partial adaptation of LogRules)

LogRules works in three stages: induction, alignment and deduction. We implement **induction** and
**deduction** and leave out the CPO alignment (fine-tuning) stage.

**Induction.** The training pool contains every template seen in the training period, labeled by the
majority label of its training messages. The generator LLM sees batches of 20 labeled templates,
balanced between normal and abnormal, over 3 shuffled rounds. For each batch it proposes up to 8 rules
as JSON: a sentence, keyword phrases copied from the logs, and a label (`prompts/rule_induction.txt`).
Duplicate proposals are merged.

The generator is **Qwen2.5-7B-Instruct**, the same open model as the classifier. LogRules used
GPT-4o-mini, which was not available to us (no API key), and the proposal allows a comparable model in
that case. This is a limitation worth discussing: the classifier applies rules written by a model of
its own size, so a stronger generator might induce better rules. Verification on the training set
guards against the worst of them either way.

**Verification and ranking.** Every candidate rule is applied to the *whole* training pool by keyword
matching. A rule is kept only if it is never wrong on the templates it matches (precision 1.0) and
matches at least one. Kept rules are ranked by usage: the training templates (then messages) they
cover correctly. The top 30 are used (`outputs/bgl/rules/rules.md`): **TBD** candidates gave **TBD**
kept rules.

**Deduction: four conditions.** The classifier is **Qwen2.5-7B-Instruct** (greedy decoding). Each
*unique template* in the validation and test periods is classified once, and its label is copied to
every message with that template. A window is flagged if any of its messages is.

1. **Zero-shot:** the template alone (`prompts/zero_shot.txt`).
2. **Few-shot (case-based):** plus 8 fixed, label-balanced training templates (`prompts/few_shot.txt`).
3. **Rule-based:** plus the 30 ranked rules; the model must apply the first matching rule and cite
   its number (`prompts/rule_based.txt`).
4. **Keyword baseline (no LLM):** the same 30 rules as a decision list of plain string matches. The
   first rule that fires decides; otherwise the template is normal. This tests whether the LLM adds
   anything beyond the rules.

Answers are parsed as JSON, with a word-level fallback. Unusable answers default to *normal* and are
counted (**TBD** per condition). Every prompt and raw answer is stored in a JSONL cache
(`outputs/llm_cache/`), which makes reruns free and the experiment auditable.

## 4. Evaluation

**Metrics.** Precision, recall, F1 and false positives on the shared test windows; message-level F1 as
a secondary result for Approach B (the LogRules setting); and inference time. For LogBERT this is the
wall-clock time to score the test windows. For Approach B it is the summed latency of classifying the
test templates, using the original latency for cached answers. `scripts/compare.py` recomputes all
metrics from the saved predictions and refuses methods that did not score exactly the shared windows.

**Main results (test period, window level).** The test set has 5,742 windows, of which 495 (8.6%) are
anomalous. Inference times were measured on 4 CPU cores.

| Method | Precision | Recall | F1 | FP | Msg-level F1 | Inference time |
|---|---:|---:|---:|---:|---:|---:|
| A: LogBERT (g = 12, r = 3) | 0.174 | 0.921 | 0.293 | 2,159 | - | 78 s |
| A+: LogBERT, full-coverage masking (g = 8, r = 8) | 0.184 | 0.897 | 0.305 | 1,975 | - | 156 s |
| B1: LLM zero-shot | TBD | TBD | TBD | TBD | TBD | TBD |
| B2: LLM few-shot | TBD | TBD | TBD | TBD | TBD | TBD |
| B3: LLM rule-based | TBD | TBD | TBD | TBD | TBD | TBD |
| B4: Keyword rules (no LLM) | TBD | TBD | TBD | TBD | TBD | TBD |
| Template-level oracle (upper bound for B) | 1.000 | 0.992 | 0.996 | 0 | 0.999 | - |

**Analysis** (`results/analysis/analysis.md`).

- **LogBERT's false positives come from the drift.** 99.8% of its 2,159 false-positive windows contain
  a template never seen in training. It flags only 0.3% of normal test windows without such a template,
  but 57.7% of those with one. Recall stays high: it misses 39 of the 495 anomalous windows.
- **Tuning on test would hide part of this.** Choosing g and r on the test windows, as the reference code
  does, raises test F1 from 0.293 to 0.475 (g = 6, r = 20, precision 0.337). That setting sits at the
  edge of our grid, so the inflated figure is a lower bound. The validation-tuned number is the honest one.
- **The operating point is fragile.** On validation, only 16 of 231 (g, r) settings come within 0.02 of
  the best F1 (0.599); at r = 3, F1 ranges from 0.261 to 0.599 across g. Validation also under-represents
  the drift: 9.6% of its normal windows contain an unseen template, against 71.1% in test.
- **Full coverage helps a little.** Checking every key raises precision from 0.174 to 0.184 (F1 0.293 →
  0.305) at twice the inference time. Both variants catch the same 442 anomalous windows and share
  1,925 false positives.
- **What LogBERT misses.** Window recall is 0.95 to 0.99 for the large alert categories (`APPSEV`,
  `KERNRTSP`, `KERNMC`, `MASABNORM`), but 0.39 for `KERNSOCK` (`<*> <*> <*> socket closed`, 14 missed
  windows) and 0.14 for `LINKIAP` (`MidplaneSwitchController::receiveTrain() iap failed …`).
- **Template purity barely limits Approach B.** Only 3 of 472 test templates are mixed, holding 258
  messages and 31 of the 28,915 test alerts, so a template-level oracle reaches F1 0.996.
- **What the rules add** (TBD). Compare B3 with B4: if they are close, the rules, not the LLM, do the
  work; if B3 is clearly better, the LLM generalises the rules to templates no keyword matches.
- **Agreement between the approaches** (TBD). Which anomalous windows only A, only B, both or neither
  catch, to tell whether the approaches are complementary.
- **Cost.** LogBERT scores the 5,742 test windows in 78 s (156 s with full coverage). The LLM's cost grows
  with the number of unique templates (472 in the test period), not messages, but every template that
  appears after deployment needs another LLM call.

## 5. Reflection: errors and difficulties

- **Matching LogBERT's windowing exactly.** The original `sliding_window` loop has edge cases: it stops
  after the first window that reaches the last message, and it drops empty and duplicate windows. We
  replaced it with a binary-search version and wrote a test that compares both on random bursty
  timelines.
- **Reading the reference code critically.** Three choices in the LogBERT repository would have inflated
  our numbers: tuning the threshold on test, dropping short test sequences, and shuffled
  training/test selection. Documenting and changing them was a large part of the reproduction.
- **Random masking leaves messages unchecked.** Inspecting single windows in the demo showed that an
  inserted alert is often simply not masked, which motivated the full-coverage extension.
- **Template granularity.** Drain with a 0.3 similarity threshold merges some alert and normal messages
  into one template (`data <*> <*>` on the sample). This caps every template-level method, so we added
  the purity analysis and the oracle bound.
- **Engineering bugs caught by tests.** NumPy floats in the training history made checkpoints
  unloadable with the safe `torch.load(weights_only=True)`. A categorical pandas column broke
  `fillna(0)` under pandas 3. Integer table columns rejected string labels in the demo.
- **Infrastructure.** The development environment could not reach Hugging Face, Zenodo or the OpenAI API.
  We therefore designed for offline testing: stub LLMs in unit tests, the 2,000-line sample for smoke
  runs, and a response cache so the expensive LLM steps run once on Colab.
- **Small samples mislead.** On the 2,000-line sample, which is spread over seven months, most windows
  hold one to three messages and LogBERT flags almost everything (test precision 0.13 at recall 1.0).
  The sample is only a pipeline check, not evidence about either method.
- **TBD:** difficulties from the full run (training time, LLM output format errors, rule quality, etc.).

## 6. Reproducibility

All settings live in `configs/bgl.yaml`; every run saves its resolved configuration, and seeds are fixed.
`make all` (or the Colab notebook) reproduces every number above, and `make smoke` checks the pipeline in
about a minute on a CPU. The trained LogBERT checkpoint, the induced rules, all prompts and the LLM
response cache are kept with the results.

## References

- Guo, H., Yuan, S., & Wu, X. (2021). LogBERT: Log anomaly detection via BERT. arXiv:2103.04475.
- He, P., Zhu, J., Zheng, Z., & Lyu, M. R. (2017). Drain: An online log parsing approach with fixed depth tree. ICWS 2017, 33-40.
- Huang, X., Zhang, T., & Zhao, W. (2025). LogRules: Enhancing log analysis capability of large language models through rules. Findings of NAACL 2025, 452-470.
- Le, V.-H., & Zhang, H. (2022). Log-based anomaly detection with deep learning: How far are we? ICSE 2022. arXiv:2202.04301.
- Oliner, A., & Stearley, J. (2007). What supercomputers say: A study of five system logs. DSN 2007, 575-584.
- Zhu, J., He, S., He, P., Liu, J., & Lyu, M. R. (2023). Loghub: A large collection of system log datasets for AI-driven log analytics. ISSRE 2023, 355-366.
