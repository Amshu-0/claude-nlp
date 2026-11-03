# Speaker notes

The notes of `presentation.pptx`, slide by slide, to read and rehearse anywhere. In PowerPoint
they appear under each slide in Presenter View. This file is written by `build_deck.js`: edit
the notes there and rebuild.

## 1. Finding failures in supercomputer logs

**TIME:** about 20 seconds

**SAY:**
Hello, my name is [your name]. My project asks a practical question: how should we find failures in the logs of a supercomputer?

I compare two very different approaches from recent papers. The first is LogBERT, a BERT-style model that learns patterns in sequences of log messages. The second follows a 2025 paper called LogRules: a large language model reads the text of the messages and is guided by rules it wrote itself. I ran both on the same data under the same rules, and the result surprised me.

**POINT AT:** the log lines on the right. They are real lines from the dataset; the two orange ones are alerts.

**NEXT:** Let me start with what one of these lines contains.

## 2. What a log line contains

**TIME:** about 1 minute

**SAY:**
This is one real line from our dataset, BGL: the system log of the BlueGene/L supercomputer at Lawrence Livermore National Laboratory. Each line has ten fields; I show the important ones: a time stamp, the hardware node that wrote it, the software component, a severity level, and the free-text message, here 'data TLB error interrupt'.

The first field is special. It is the alert tag: system administrators marked every message that should raise an alert with a category, here KERNDTLB, and normal messages get a dash. That tag is our ground truth. So the first thing the pipeline does is cut it off. If a model could see it, it would 'detect' every alert without learning anything. This is called label leakage, and a test in the code checks that it cannot happen.

The data set is big: 4.7 million lines over seven months, 348,460 of them alerts, which is 7.3 percent.

**POINT AT:** the orange chip (the label, removed first) and the blue chip (what the models read).

**IF ASKED:**
- The ten fields: alert tag, epoch time, date, node, full time, node again, type, component, level, content.
- There are 41 alert categories. The most common, KERNDTLB, has 152,734 messages.

**NEXT:** Why is finding the alerts hard?

## 3. Why this is hard

**TIME:** about 1 minute

**SAY:**
Three things make this hard. First, alerts are rare: only 7.3 percent of the messages. A detector that always says 'normal' is 92.7 percent accurate and completely useless. That is why I report precision, recall and F1, never accuracy.

Second, the log changes over time. Software is updated and new kinds of messages appear. In our test period, almost 80 percent of the messages use a message type the models never saw during training.

Third, words can mislead. 'Instruction cache parity error corrected' contains the word error but is normal, because the hardware fixed itself. 'LOGIN failed: Input/output error' looks routine, but it was an alert.

So the research question is: how does a BERT-style model that learns sequences of log-template IDs compare with a rule-guided language model that reads the text, at finding these anomalous events?

**POINT AT:** the three numbers, then the question at the bottom.

**NEXT:** Here are the two approaches side by side.

## 4. Two approaches, one fair setup

**TIME:** about 1 minute

**SAY:**
On the left, Approach A: LogBERT, from 2021. It never reads words. It sees each message only as an ID number and learns which sequences of IDs are normal. It trains only on windows without alerts, so it never sees a failure. It is a small Transformer, 1.7 million parameters, trained from scratch.

On the right, Approach B, inspired by LogRules from 2025. It reads the words of each message type. A large language model, Qwen 2.5 with 7 billion parameters, writes rules from labelled training examples, and the rules guide the classification. Nothing is trained here.

The two papers report numbers on different data samples and evaluation units, so their published numbers cannot be compared. The heart of this project is the shared setup at the bottom: the same data, a chronological split, the same 5,742 test windows, and one script that scores both.

**IF ASKED:**
- How different the papers' setups are: DeepLog scores F1 86 on BGL in the LogBERT paper and 16.8 in the LogRules paper.

**NEXT:** First, how both approaches see the data.

## 5. From log lines to templates and windows

**TIME:** about 1 minute

**SAY:**
Messages that differ only in their numbers, like 'generating core.304' and 'generating core.2276', come from the same line of source code. The Drain algorithm finds this shared shape, the template, and replaces the parts that vary with a star placeholder. 4.7 million messages become 923 templates. Each template gets an ID, called a log key, which plays the role of a word in a vocabulary. LogBERT reads these IDs; the language model reads the template text.

Then the timeline is cut into five-minute windows that slide by one minute, as in the LogBERT paper. A window is anomalous if it contains at least one alert, so one alert makes several overlapping windows anomalous, as the diagram shows. Every method must give a verdict for every one of the 5,742 test windows.

**POINT AT:** the template examples, then the orange alert in the window diagram.

**IF ASKED:**
- Drain settings come from the LogBERT repository: depth 3, similarity 0.3, masking of numbers, hex values and IP addresses.
- Drain over-merges a little: on the 2,000-line sample it finds 100 templates against 120 in the hand-made ground truth. The LOGIN example shows the cost: the words that decide the label become stars.

**NEXT:** How we split the data turned out to be decisive.

## 6. The log changes over time

**TIME:** about 40 seconds

**SAY:**
We split by time: the first 60 percent of the timeline for training, the next 20 percent for validation, the last 20 percent for testing. Never at random, because a random split lets a model learn from the future and inflates its scores.

The chart shows the consequence. In the test period, 79.6 percent of all messages use a template that never appears in training, and 71 percent of the normal test windows contain at least one such template. Keep this in mind: it explains most of the results.

**POINT AT:** the dark blue Test bar, then the light blue one next to it.

**IF ASKED:**
- Validation is much milder: 8.9 percent of its messages and 9.6 percent of its normal windows. So settings tuned on validation do not see the full drift.
- The two new diagnostics are '… total interrupts …' and '… microseconds spent in the rbs signal handler …'.

**NEXT:** Now the two approaches in more detail, starting with LogBERT.

## 7. LogBERT learns “normal” by filling in blanks

**TIME:** about 1 minute

**SAY:**
LogBERT borrows BERT's training trick: fill in the blanks. In a normal window it hides half of the log keys and learns to predict them from the keys around them, like guessing a missing word in a sentence. This is called masked log key prediction.

Under the hood, each key becomes a vector of 256 numbers, an embedding. Two Transformer layers with self-attention let every position look at every other position in the window. A second loss, the hypersphere loss, pulls a summary vector of every normal sequence towards one centre point, so normal windows cluster together.

It trains only on normal windows: 19,029 of them, which contain 11,615 distinct sequences once duplicates are removed. Training took one hour on four CPU cores.

**POINT AT:** the hidden keys in the diagram, then the two losses.

**IF ASKED:**
- Total loss: masked-key cross-entropy plus 0.1 times the hypersphere loss, as in the LogBERT repository.
- 2 layers, 4 heads, hidden size 256, feed-forward 1024, maximum length 512; the best dev loss (1.02) came at epoch 5.
- The hidden keys are replaced 80% of the time by a mask token, 10% by a random key and 10% kept, exactly as in BERT.

**NEXT:** How does it decide that a window is anomalous?

## 8. LogBERT flags windows it cannot predict

**TIME:** about 50 seconds

**SAY:**
At test time LogBERT hides keys again and ranks the true key among its 176 guesses. If the true key is not among its top g guesses, it is a miss, and a window with more than r misses is anomalous. One detail matters a lot: a template it never saw in training becomes the 'unknown' token and is always a miss.

g and r are chosen on the validation period: g equals 12 and r equals 3. The heat map shows how fragile that choice is: only 16 of 231 settings come close to the best. The original LogBERT code chooses its threshold on the test set; doing that here would raise F1 from 0.29 to 0.48, which would not be honest.

I also added an extension, A-plus, that checks every key once instead of a random half.

**POINT AT:** the 'unk always missed' box, then the dot and the outlined cells in the heat map.

**IF ASKED:**
- Ranks are stored once, so all 231 settings are evaluated with one pass of the model.
- Full coverage: F1 0.293 → 0.305 and 2,159 → 1,975 false alarms, at twice the inference time (78 s → 156 s).

**NEXT:** Approach B works completely differently.

## 9. One LLM, four kinds of guidance

**TIME:** about 50 seconds

**SAY:**
Approach B uses Qwen 2.5 7B Instruct, an open language model. With no GPU and no API key, I ran a 4-bit version through llama.cpp on the CPU, with greedy decoding so the answers are reproducible.

I test four conditions. B1, zero-shot: just the template and a question. B2, few-shot: plus eight labelled training examples. B3, rule-based: plus 30 rules that were induced and verified on the training data; the model must cite the rule it used. And B4, the control experiment: the same 30 rules applied as plain keyword matching, with no language model at all. If B3 is not better than B4, the rules do the work, not the model.

Each of the 472 test templates is classified once, and its label is copied to all of its messages.

**POINT AT:** the four cards from left to right, then the JSON answer.

**IF ASKED:**
- The model reads about 23 tokens per second and writes about 5 on 4 cores. Ending every prompt with the log line lets the server reuse the shared beginning: about 20% faster.
- All 1,416 classification answers were valid; none fell back to the default label.
- Classifying templates instead of messages costs almost nothing here: the best a template-level method could reach is F1 0.996.

**NEXT:** Where do the rules come from?

## 10. The LLM writes rules; the data checks them

**TIME:** about 50 seconds

**SAY:**
The language model writes the rules. It sees the 442 labelled training templates in batches of 20, half normal and half abnormal, and proposes rules with keyword phrases copied from the logs. That gave 160 candidate rules.

Then the data checks them. Each rule is applied to every training template by keyword matching, and only rules that are never wrong are kept: 99 of the 160. They are ranked by how many templates they cover, and the top 30 are used, as in the LogRules paper.

On the left is a real rule. The weak spots are already visible: vague sentences, near-duplicates, and odd keywords. Most importantly, verification checks the keywords, not the sentence the model will read. That will matter in a moment.

**POINT AT:** the funnel 160 → 99 → 30, then the rule's sentence versus its keywords.

**IF ASKED:**
- LogRules used GPT-4o-mini to write rules. I used the same Qwen model for both roles because no API key was available; the proposal allowed a comparable model.
- Rule induction took 50 minutes for 42 batches. Keywords shorter than three letters or digits are dropped automatically.
- LogRules' third stage, fine-tuning a small model to follow rules (CPO), is not reproduced.

**NEXT:** Briefly, how all this is implemented.

## 11. One pipeline, tested end to end

**TIME:** about 30 seconds

**SAY:**
Everything is one Python package with one command per step, driven by a single configuration file. The diagram is the actual pipeline: preprocessing, then the two approaches in parallel, then one comparison script that refuses any method that did not score exactly the same windows.

There are 148 automated tests and continuous integration on every push. And every result, including the trained model, the rules and every language-model answer, is committed to the repository, so all numbers can be checked, and Approach B can even be replayed without the model.

**IF ASKED:**
- The code is in src/logad with five subpackages; scripts/ has one entry point per step; docs/guide.md explains every file.
- The comparison script recomputes all metrics from the saved predictions and rejects a method that skipped or added windows.

**NEXT:** So, the results.

## 12. Results: the rules win, without the LLM

**TIME:** about 1 minute

**SAY:**
Here is the main table on the 5,742 test windows. Every method finds about 90 percent or more of the anomalous windows: recall is high everywhere. What separates them is precision: how many of their alarms are real.

LogBERT has an F1 of 0.29: it raises 2,159 false alarms. The language model alone, zero-shot, is even worse: it flags almost everything. Examples help, to 0.48; rules help more, to 0.64.

But the best method by far is B4, the same rules as plain keyword matching: F1 0.956 with zero false alarms, in 8 milliseconds instead of an hour. So the answer to the research question has two parts: the rule-guided approach beats LogBERT, but the rules, not the language model, do the work. The next slides explain why.

**POINT AT:** the B4 row, then the longest orange bar.

**IF ASKED:**
- Message-level F1, the LogRules metric: B4 0.984, B3 0.491, B2 0.444, B1 0.180.
- One-off costs: 60 minutes to train LogBERT, 50 minutes to induce the rules.
- Accuracy would hide this: LogBERT is 62% accurate, and never flagging anything would be 91% accurate.

**NEXT:** Why does LogBERT do so badly?

## 13. LogBERT flags whatever is new

**TIME:** about 1 minute

**SAY:**
To find out, I ran the trained model again over the whole test period and recorded every guess. For templates it knows, it misses only 2 percent of the keys: it really did learn normal behaviour. A template it never saw is always a miss, by construction.

And here is the key fact: 97 percent of the alert keys belong to templates LogBERT never saw, because alert messages almost never occur in normal training windows. So LogBERT finds alerts mainly because they are new, not because it understands them.

But in the test period, 80 percent of the normal messages are new as well. So 58 percent of the normal windows that contain a new template get flagged, against 0.3 percent of the others, and 99.8 percent of its false alarms contain a new template. LogBERT has become a new-message detector.

**POINT AT:** the two bars at 100%, then the 97%.

**IF ASKED:**
- Counts: 28,094 of 28,915 alert keys are new to LogBERT; 360,557 normal keys are new as well.
- The figure comes from scripts/make_figures.py, which masks every key of the test period once and records the rank of the true key.

**NEXT:** What does the language model do instead?

## 14. Guidance narrows the LLM

**TIME:** about 45 seconds

**SAY:**
Of the 472 test templates, only 20 contain alerts. Zero-shot, the model calls 388 templates abnormal: anything that sounds like an error. Its idea of 'abnormal' is not the administrators' idea of an alert.

Eight examples cut that to 122, and the rules to 37. The keyword rules flag only 9, and all 9 are correct. So guidance clearly helps the language model, which matches the LogRules paper. But the model still flags far more than the rules themselves do.

**POINT AT:** the blue part of each bar: the false alarms.

**IF ASKED:**
- Zero-shot false alarms include '… total interrupts …' (135,092 messages), 'instruction cache parity error corrected' (33,547) and 'ciod: Received signal …' (27,510).

**NEXT:** Why is the model worse than its own rules?

## 15. The LLM follows rule wording, not keywords

**TIME:** about 1 minute

**SAY:**
I checked every rule the model cited. It cites a rule in 420 answers, but in only 20 of them does that rule's keyword actually occur in the log.

Here is an example. For 'ciod: Received signal', 27,510 normal messages, the model answers abnormal because of rule 8, 'contains terminated'. The word 'terminated' is not in the message.

The model follows the rules' sentences, which are broad, not their verified keywords. That has one benefit: it catches 5 alert templates that no keyword matches, worth 14 windows. But it also creates 23 false-alarm templates covering 57,000 normal messages, 12 of them through one vague rule, rule 4. The keyword rules have zero false alarms.

**POINT AT:** the message, then the word 'terminated' in the answer.

**IF ASKED:**
- Only 12 of the 420 citations name the rule the keyword matcher would actually apply.
- The model never labels normal a template the keyword rules call abnormal; it only adds abnormal labels.
- Rule 1, 'Describes normal system operations or status updates', serves as its catch-all for 349 normal answers.

**NEXT:** Two more traps.

## 16. Two traps: rule ranking and label drift

**TIME:** about 45 seconds

**SAY:**
First, rule ranking. Ranking by coverage fills the top 30 with near-duplicates, and the rule for 'socket closed' lands at rank 32, just outside. That template causes most of the keyword rules' misses. Choosing the number of rules on validation would give 45 rules and a test F1 of 0.98. I report 30 because the proposal fixed it in advance.

Second, label drift. In training, 'LOGIN chdir failed: Permission denied' is normal. In the test period, 'LOGIN chdir failed: Input/output error' is an alert, 816 messages. Drain masked exactly the words that decide the label, so both become the same template, and every method that learned from the training data calls it normal.

**POINT AT:** the jump of both lines at k = 35, then the two LOGIN lines.

**IF ASKED:**
- The LOGIN template accounts for 816 of the 913 alert messages the keyword rules miss, but only 2 missed windows, because those windows contain other alerts that are caught.
- A better rule selection would reward the new templates a rule covers, not raw usage.

**NEXT:** What else went wrong during the project?

## 17. What went wrong, and how we fixed it

**TIME:** about 1 minute

**SAY:**
Several things went wrong, and fixing them was a large part of the work. First, reading the reference code critically: the LogBERT repository tunes its threshold on the test set, shuffles windows before splitting, skips short test windows and truncates long ones. Each of these would have inflated our numbers, so I changed them and documented why.

Second, resources: there was no GPU and no OpenAI key. I ran the language model in 4-bit on four CPU cores, and putting the log line at the end of every prompt let the server reuse its work, which made each call 20 percent faster.

Third, the cloud machine restarted while I was away, once in the middle of the language-model run. Because every answer is cached, the rerun reused 689 stored answers and only computed the rest.

And the dataset's official host was unreachable, so I verified a mirror with a checksum and line counts.

**IF ASKED:**
- Engineering bugs the tests caught: NumPy numbers in the training history made checkpoints unloadable with safe loading; a pandas category column broke fillna; the demo's table rejected string labels in integer columns.
- Windowing: LogBERT's sliding-window loop has edge cases; our vectorised version is tested against a port of the original on random timelines.

**NEXT:** What I take away from this.

## 18. Lessons and next steps

**TIME:** about 30 seconds

**SAY:**
Four lessons. The evaluation setup can change the answer more than the choice of model. Always check what a model relies on: LogBERT turned out to detect newness. Check that a language model actually does what it says it does. And always run the simple baseline: here it won.

Next steps would be selecting rules for what they add, checking rule sentences and not only keywords, training LogBERT longer on a GPU, and testing on other log datasets.

**IF ASKED:**
- Combining LogBERT with the keyword rules does not help: OR keeps LogBERT's false alarms (F1 0.301), AND scores 0.939, below the rules alone. Only B3 gains from requiring LogBERT to agree.

**NEXT:** Now let me show you the system live.

## 19. Live demo

**TIME:** about 10 minutes (the course's demo block)

**SAY:**
BEFORE THE TALK: on the presenting machine run 'make data preprocess', then 'make restore OUT=outputs/bgl_cpu', then start 'python app/demo.py --config configs/bgl_cpu.yaml --share' and keep the page open. The share link lets the audience try it on their phones.

Step 1. Paste the full KERNDTLB line. Show that the alert tag is removed first and shown as the ground truth. The keyword rules flag the line with rule 24, and so does the rule-based language model. LogBERT misses this key, because it never saw it in a normal window, but one miss is not enough to flag the window: it needs more than three.

Step 2. Paste 'instruction cache parity error corrected'. It contains the word error but is normal. The keyword rules and the rule-based model leave it alone. Mention that zero-shot, the model called it abnormal.

Step 3. Paste the LOGIN Input/output error line. It is a real alert, but it maps to the template that was normal in training, so the guided methods miss it: label drift, live.

Step 4. Open the second tab, choose a normal window and insert 'rts panic! - stopping execution' at different positions. Watch how LogBERT's verdict and the 'without your line' count change.

Step 5, if time allows. Ask the audience for a made-up line, for example 'fan module 3 failed on node card R12-M0'. Nobody has seen this template: LogBERT reads it as unknown, and the keyword rules match nothing. Explain what each method sees.

IF SOMETHING FAILS: without the model server, the demo still runs on the saved decisions (add --no-live-llm); for new lines, only the keyword rules and LogBERT answer.

**NEXT:** Back to the slides for the conclusion.

## 20. Conclusion

**TIME:** about 45 seconds

**SAY:**
To answer the research question: under one fair, chronological setup, the rule-guided approach clearly beats LogBERT: 0.64 for the rule-based model and 0.96 for the same rules as keywords, against 0.29.

The main reason is that the log changes over time. LogBERT sees only IDs, so it flags anything new; the rule-guided approach reads the words and can tell a harmless new message from an alert.

But the verified rules, not the language model, do the work: the model applies them loosely, and plain keyword matching is better and about 400,000 times faster.

These results come from one dataset and one run under a CPU budget, so a stronger rule generator or a dataset whose failures are less tied to specific words could change the balance.

**NEXT:** Thank you. I'm happy to take questions.

## 21. Thank you. Questions?

**TIME:** the remaining time, about 4 minutes

**SAY:**
Thank you for listening. I'm happy to take questions.

**IF ASKED:**
- Why not compare the papers' numbers? Different data samples and evaluation units: DeepLog scores F1 86 on BGL in one paper and 16.8 in the other. Only a shared setup answers the question.
- Is it unfair that Approach B uses labels? That is the trade-off studied: LogBERT is semi-supervised by design, and the zero-shot condition B1, without labels, is the weakest method.
- Why is LogBERT worse than in its paper? Its code shuffles windows into training and test and picks the threshold on test; a chronological split with validation tuning is much harder. Tuning on test would give 0.475 here.
- Why classify templates, not messages? 472 LLM calls instead of 482,331; the best possible template-level F1 is 0.996, so little is lost.
- Did you reproduce LogRules fully? Induction and deduction, not the CPO fine-tuning stage; and one 4-bit 7B model wrote and applied the rules (LogRules used GPT-4o-mini to write them).
- Is k = 30 cherry-picked? It was fixed in the proposal before the run; validation would pick 45, which gives 0.980.
- How sensitive is LogBERT to g and r? Very: only 16 of 231 settings come within 0.02 of the best validation F1.
- Could the approaches be combined? They miss different windows, but OR keeps LogBERT's false alarms (0.301) and AND gives 0.939, below the rules alone.
