# Comparing BERT-Based and Rule-Guided Language Models for System Log Anomaly Detection

**Technical report** · CSCI 390 Natural Language Processing and Applications · Dr. Qi Li

> **The reported run.** Every number in this report comes from one run on the full BGL log on a
> 4-core CPU without a GPU (`configs/bgl_cpu.yaml`). LogBERT was trained on the CPU, and Approach B
> used Qwen2.5-7B-Instruct (4-bit) through a local llama.cpp server. The tables are copied from
> `results/comparison.md` and `results/analysis/analysis.md`.

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
majority label of its training messages: 442 templates, 31 of them abnormal. The generator LLM sees
batches of 20 labeled templates, half normal and half abnormal (the rare abnormal ones are cycled).
For each batch it proposes up to 8 rules as JSON: a sentence, keyword phrases copied from the logs,
and a label (`prompts/rule_induction.txt`). Duplicate proposals are merged. The default configuration
makes 3 shuffled passes over the pool; to save CPU time, the reported run made one pass (42 batches).

The generator is **Qwen2.5-7B-Instruct**, the same open model as the classifier. LogRules used
GPT-4o-mini, which was not available to us (no API key), and the proposal allows a comparable model in
that case. This is a limitation worth discussing: the classifier applies rules written by a model of
its own size, so a stronger generator might induce better rules. Verification on the training set
guards against the worst of them, though only for their keywords (Section 4).

**Verification and ranking.** Every candidate rule is applied to the *whole* training pool by keyword
matching. A rule is kept only if it is never wrong on the templates it matches (precision 1.0) and
matches at least one. Kept rules are ranked by usage: the training templates (then messages) they
cover correctly. The top 30 are used. In the reported run, the 42 batches gave 160 distinct candidate
rules, of which 99 survived verification (80 abnormal, 19 normal). The top 30 hold 22 abnormal and 8
normal rules (`results/rules/rules.md`).

**Deduction: four conditions.** The classifier is **Qwen2.5-7B-Instruct** (greedy decoding). Each
*unique template* is classified once, and its label is copied to every message with that template. A
window is flagged if any of its messages is. To save CPU time, the reported run classified only the 472
test templates with the LLM; the keyword baseline, which costs nothing, was also scored on validation.

1. **Zero-shot:** the template alone (`prompts/zero_shot.txt`).
2. **Few-shot (case-based):** plus 8 fixed, label-balanced training templates (`prompts/few_shot.txt`).
3. **Rule-based:** plus the 30 ranked rules; the model must apply the first matching rule and cite
   its number (`prompts/rule_based.txt`).
4. **Keyword baseline (no LLM):** the same 30 rules as a decision list of plain string matches. The
   first rule that fires decides; otherwise the template is normal. This tests whether the LLM adds
   anything beyond the rules.

Answers are parsed as JSON, with a word-level fallback. Unusable answers would default to *normal* and
be counted; in the reported run all 1,416 classification answers were usable. Every prompt and raw
answer is stored in a JSONL cache, which makes reruns free and the experiment auditable. The cache of
the reported run is committed (`results/llm_cache/`) and replays all of Approach B without a model.

## 4. Evaluation

**Metrics.** Precision, recall, F1 and false positives on the shared test windows; message-level F1 as
a secondary result for Approach B (the LogRules setting); and inference time. For LogBERT this is the
wall-clock time to score the test windows. For Approach B it is the summed latency of classifying the
test templates, using the original latency for cached answers. `scripts/compare.py` recomputes all
metrics from the saved predictions and refuses methods that did not score exactly the shared windows.

**Main results (test period, window level).** The test set has 5,742 windows, of which 495 (8.6%) are
anomalous. Inference times were measured on 4 CPU cores. For B1-B3 they are the summed latencies of
the 472 test templates (6.0, 6.8 and 7.2 s per template).

| Method | Precision | Recall | F1 | FP | Msg-level F1 | Inference time |
|---|---:|---:|---:|---:|---:|---:|
| A: LogBERT (g = 12, r = 3) | 0.174 | 0.921 | 0.293 | 2,159 | - | 78 s |
| A+: LogBERT, full-coverage masking (g = 8, r = 8) | 0.184 | 0.897 | 0.305 | 1,975 | - | 156 s |
| B1: LLM zero-shot | 0.108 | 1.000 | 0.194 | 4,103 | 0.180 | 47 min |
| B2: LLM few-shot | 0.323 | 0.941 | 0.481 | 976 | 0.444 | 53 min |
| B3: LLM rule-based | 0.489 | 0.943 | 0.644 | 488 | 0.491 | 57 min |
| B4: Keyword rules (no LLM) | 1.000 | 0.915 | **0.956** | 0 | 0.984 | 8 ms |
| Template-level oracle (upper bound for B) | 1.000 | 0.992 | 0.996 | 0 | 0.999 | - |

**Analysis** (`results/analysis/analysis.md`).

*Approach A.*

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

*Approach B.*

- **On its own, the LLM flags anything that sounds like an error.** Zero-shot, Qwen labels 388 of the
  472 test templates abnormal. It catches every anomalous window but also flags 78% of the normal ones
  (precision 0.108). Its false alarms read like errors but were never alerts on BGL: `<*> total
  interrupts. …` (135,092 messages), `instruction cache parity error corrected` (33,547) and `ciod:
  Received signal …` (27,510).
- **Guidance helps, and rules help more than examples.** Eight labeled training examples cut the
  abnormal templates to 122 (F1 0.481). The 30 ranked rules cut them to 37 (F1 0.644, 488
  false-positive windows). This is the comparison LogRules makes in favour of rules over case-based
  prompting.
- **The rules, not the LLM, do the work.** Applied as plain keyword matching, the same 30 rules (B4)
  reach F1 0.956 with no false positive at all (message-level F1 0.984), in 8 ms instead of 57
  minutes. On the test templates the LLM never labels normal what the keyword rules label abnormal; it
  only adds 28 abnormal labels of its own. Five of them are real alert templates that no keyword
  matches (`MidplaneSwitchController::receiveTrain() iap failed …`, `Power Good signal deactivated …`,
  `L<*> major internal error`, `… bll_clear_port failed …` and `rts assertion failed …`). They are
  worth 14 more anomalous windows (recall 0.915 → 0.943). The other 23 are false alarms covering 57,309
  normal messages.
- **The LLM follows the rules' sentences, not their keywords.** B3 cites a rule in 420 of its 472
  answers, but the cited rule's keywords occur in the template in only 20 of them, and only 12
  citations name the rule the keyword decision list would apply. All 23 false-positive templates cite
  an abnormal rule none of whose keywords occurs in the template; 12 of them cite rule 4. Rule 4 reads
  "Messages mentioning 'FAILED', 'terminated for reason', 'kernel panic', and 'machine check
  interrupt' are indicative of system failures", but its verified keywords are `lustre mount failed`,
  `rts: kernel terminated for reason`, `kernel panic` and `machine check interrupt`. The model applies
  the sentence to `ciod: Error loading … Permission denied` (13,952 messages) and to `ciod: LOGIN <*>
  failed: No such file or directory` (3,623). Verification guarantees the keywords, not the sentence
  the LLM reads. Rule 1, "Describes normal system operations or status updates.", serves as the
  catch-all for 349 normal answers.
- **The top-30 cut-off costs the keyword rules recall.** Ranking by usage fills the top 30 with
  near-duplicates: 12 of the 30 rules are variants of "terminated", "panic" or "exited abnormally"
  (ranks 8-10, 13-15, 20-22 and 27-29). This pushes the `socket closed` rule to rank 32 and the rules
  for `receiveTrain() iap failed` to ranks 63 and 64; these two templates occur in most of B4's 42
  missed windows. The keyword baseline's F1 for different numbers of rules:

  | Rules used (top k) | 10 | 20 | 30 (reported) | 35 | 45 | 60 | 99 (all) |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Validation F1 | 0.820 | 0.831 | 0.837 | 0.885 | 0.888 | 0.870 | 0.870 |
  | Test F1 | 0.942 | 0.942 | 0.956 | 0.980 | 0.980 | 0.980 | 0.988 |

  Test precision stays 1.000 for every k; validation precision falls to 0.942 (22 false-positive
  windows) from k = 60. Choosing k on validation (best F1 at k = 45) would have given test F1 0.980.
  We report k = 30, the value fixed in the proposal. Selecting rules for the templates they add rather
  than for raw usage would avoid the problem.
- **Template purity barely limits Approach B, but some labels change over time.** Only 3 of 472 test
  templates are mixed, holding 258 messages and 31 of the 28,915 test alerts, so a template-level
  oracle reaches F1 0.996. Between periods, however, two templates change label. `ciod: LOGIN <*>
  failed: <*> <*>` merges `chdir(…) failed: Permission denied` (472 normal messages in training) with
  `chdir(…) failed: Input/output error` (816 `APPOUT` alerts in test): Drain masked exactly the two
  words that carry the label. B2-B4 label it normal, which accounts for 816 of the 913 alert messages
  B4 misses. It causes only 2 missed windows, because most windows with these messages also contain
  alerts that are caught. `data <*> <*>` goes the other way: 75% alerts in training, 1 of 79 messages
  in test.

*The two approaches together.*

- **They miss different windows.** LogBERT catches 18 anomalous windows the keyword rules miss (half of
  them contain `socket closed`, which no top-30 rule covers). The rules catch 15 that LogBERT misses (8
  contain `ciod: Error reading message prefix on CioStream socket …`). Both miss 24, mostly windows
  with `socket closed` (14) and `iap failed` (6). Against the rule-based LLM the split is 10 only
  LogBERT, 21 only B3 and 18 neither.
- **But they do not combine well.** Flagging a window when either LogBERT or the keyword rules flag it
  raises recall to 0.952 but keeps all 2,159 of LogBERT's false positives (F1 0.301). Flagging it only
  when both do gives F1 0.939, below the rules alone. The rule-based LLM is the exception: requiring
  LogBERT to agree removes 280 of its 488 false-positive windows (F1 0.644 → 0.776), because the two
  raise false alarms for different reasons (new templates, misapplied rules).
- **New templates hurt LogBERT, not the rules.** 361 of the 472 test templates never occur in
  training, yet only 20 test templates contain an alert. LogBERT counts every unseen key as a miss, so
  new benign templates become false alarms. The keyword decision list labels a template normal unless
  a rule fires, which is the right default for almost all of them.
- **Cost.** LogBERT trains in 60 minutes and scores the 5,742 test windows in 78 s (156 s with full
  coverage). Rule induction took 50 minutes (42 LLM calls), after which the keyword rules classify every
  test template in 8 ms. The LLM conditions took 47-57 minutes for 472 templates. Their cost grows with
  the number of unique templates, not messages, but every template that appears after deployment needs
  another LLM call.

## 5. Conclusion

Under one shared, chronological setup on BGL, the rule-guided approach clearly beats the LogBERT
reproduction: F1 0.644 for the rule-based LLM and 0.956 for the same rules as keyword matching, against
0.293 for LogBERT. The main reason is that the log changes over time. LogBERT sees only template IDs, so
it flags windows simply because their templates are new; Approach B reads the text of each template
and can tell a new benign message from an alert.

The LLM itself is not what makes Approach B work, though. Zero-shot it flags most templates. The
induced rules improve it a lot, but it applies them loosely, and it is beaten by matching their
verified keywords literally, which takes milliseconds instead of an hour. On BGL, where alert messages
are few and distinctively worded, the value of LogRules-style induction lies in the verified rules
more than in the model that applies them. The two approaches miss different alerts, but LogBERT's false
alarms make it a poor partner for the keyword rules.

These findings come from one dataset, one chronological split and one run of each method under a CPU
budget: LogBERT trained for at most 6 epochs, induction made a single pass, and one 4-bit 7B model both
wrote and applied the rules. A stronger rule generator, rules whose sentences are verified as well as
their keywords, a rule selection that rewards coverage, or a dataset whose anomalies are less lexical
could change the balance.

## 6. Reflection: errors and difficulties

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
- **Infrastructure.** At first the development environment could reach neither Zenodo, Hugging Face nor
  the OpenAI API. We therefore designed for offline testing: stub LLMs in unit tests, the 2,000-line
  sample for smoke runs, and a response cache. We verified a mirror of the dataset (Section 2), and once
  Hugging Face became reachable we ran everything in the same environment on CPU. The Colab notebook
  remains the route to a GPU.
- **Small samples mislead.** On the 2,000-line sample, which is spread over seven months, most windows
  hold one to three messages and LogBERT flags almost everything (test precision 0.13 at recall 1.0).
  The sample is only a pipeline check, not evidence about either method.
- **Running a 7B model without a GPU or an API key.** We served the official 4-bit GGUF build of
  Qwen2.5-7B-Instruct with llama.cpp (through llama-cpp-python) on the same 4 CPU cores. At 6-7 s per
  template, the three LLM conditions took 2.6 hours, and induction (42 calls of up to 1,024 output
  tokens) took 50 minutes. Three choices kept this affordable: classifying each unique template once
  (472 calls instead of 482,331), putting the log line last in every prompt so the server can reuse the
  cached prompt prefix (about 20% faster per call), and running the LLM only after LogBERT's training.
  While both shared the CPU, a single call took 80-105 s.
- **Long jobs on a short-lived machine.** The cloud container was restarted twice while the session was
  idle; the first restart killed the model server in the middle of the zero-shot condition. The
  response cache made this harmless: the rerun took the 689 answers already stored from the cache and
  computed only the rest.
- **Answer format was no problem; rule quality was.** All 1,416 classification answers were usable. The
  induced rules were the weak point. The same 7B model wrote vague sentences ("Describes normal system
  operations or status updates."), many near-duplicates, and keywords such as `pro` or
  `<*>:<*>eeeeeee` that pass verification only because no training template contradicts them.
  Verification removed 61 of 160 candidates, but it checks keywords, not the sentences the LLM reads
  (Section 4).

## 7. Reproducibility

All settings live in `configs/bgl.yaml`. The reported run used `configs/bgl_cpu.yaml`, which extends it
for a CPU: de-duplicated training sequences, at most 6 LogBERT epochs, one induction pass, test
templates only, and a local llama.cpp server. Every run saves its resolved configuration, and seeds are
fixed. `make all CONFIG=configs/bgl_cpu.yaml`, with the server from that file running, reproduces the
run; `make smoke` checks the pipeline in about a minute. The trained LogBERT
checkpoint, the induced rules (and all candidates), the per-template decisions, all prompts and the LLM
response cache are committed under `results/`. With the cache, Approach B replays without a model
(`results/README.md`).

## References

- Guo, H., Yuan, S., & Wu, X. (2021). LogBERT: Log anomaly detection via BERT. arXiv:2103.04475.
- He, P., Zhu, J., Zheng, Z., & Lyu, M. R. (2017). Drain: An online log parsing approach with fixed depth tree. ICWS 2017, 33-40.
- Huang, X., Zhang, T., & Zhao, W. (2025). LogRules: Enhancing log analysis capability of large language models through rules. Findings of NAACL 2025, 452-470.
- Le, V.-H., & Zhang, H. (2022). Log-based anomaly detection with deep learning: How far are we? ICSE 2022. arXiv:2202.04301.
- Oliner, A., & Stearley, J. (2007). What supercomputers say: A study of five system logs. DSN 2007, 575-584.
- Zhu, J., He, S., He, P., Liu, J., & Lyu, M. R. (2023). Loghub: A large collection of system log datasets for AI-driven log analytics. ISSRE 2023, 355-366.
