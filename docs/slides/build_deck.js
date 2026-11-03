// Build presentation.pptx: the 30-minute talk, with speaker notes on every slide.
//
//   npm install                  # pptxgenjs and mermaid, pinned in package.json
//   node render_diagrams.js      # only after editing diagrams.md
//   node build_deck.js
//
// The talk follows the course format: introduction (5 min), implementation (5),
// errors and lessons (5), live demo (10), questions (5). The chart series are
// read from docs/figures/figure_data.json (written by scripts/make_figures.py
// from the reported run); the other numbers are quoted from results/ and
// docs/report.md.
const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

const HERE = __dirname;
const ROOT = path.join(HERE, "..", "..");
const FIGURES = path.join(ROOT, "docs", "figures");
const DIAGRAMS = path.join(HERE, "diagrams");
const OUTPUT = path.join(HERE, "presentation.pptx");
const DATA = JSON.parse(fs.readFileSync(path.join(FIGURES, "figure_data.json"), "utf8"));

// ------------------------------------------------------------------ theme

// Navy for the frame, blue for "normal" and Approach A, orange for "alert" and
// Approach B: the same meaning the figures give these colours.
const THEME = {
  name: "Log Anomaly Detection",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F2A", // text
    lt1: "FFFFFF", // background
    dk2: "13233F", // navy: dark slides, emphasis
    lt2: "F2F4F7", // light grey: cards
    accent1: "2A78D6", // blue: normal, Approach A
    accent2: "EB6834", // orange: alert, Approach B
    accent3: "1BAF7A",
    accent4: "EDA100",
    accent5: "6250D6",
    accent6: "5A6070", // grey: secondary text
    hlink: "2A78D6",
    folHlink: "6250D6",
  },
};
const HEX = THEME.colors; // for the options that only take hex values
const GRID = "E1E0D9"; // chart gridlines
const MONO = "Courier New"; // log lines and code

const W = 13.333; // LAYOUT_WIDE
const MX = 0.6; // left and right margin
const CW = W - 2 * MX; // content width
const TOP = 1.5; // where content starts below the title

const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Finding failures in supercomputer logs";
pres.subject = "LogBERT versus a rule-guided LLM on BGL";
pres.author = "CSCI 390 project";
const C = pres.SchemeColor;

// --------------------------------------------------------------- layouts

// Dark title layout: the opening and closing slides.
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.7, w: 6.4, h: 2.0,
      fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.9, w: 6.2, h: 1.1,
      fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "course", type: "body", x: 0.7, y: 5.55, w: 6.2, h: 0.9,
      fontSize: 14, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});

// Light content layout: a title on top, everything else composed per slide.
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.38, w: 11.2, h: 0.85,
      fontSize: 36, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Log anomaly detection on BGL  ·  CSCI 390", options: { x: MX, y: 7.02, w: 6, h: 0.3,
      fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.23, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

// Dark content layout: the demo and the conclusion.
pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.38, w: 11.2, h: 0.85,
      fontSize: 36, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Log anomaly detection on BGL  ·  CSCI 390", options: { x: MX, y: 7.02, w: 6, h: 0.3,
      fontSize: 10, color: C.background2, margin: 0 } } },
  ],
  slideNumber: { x: 12.23, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
});

// ---------------------------------------------------------- building blocks

// Width and height of a PNG file, read from its header.
function pngSize(file) {
  const header = fs.readFileSync(file).subarray(16, 24);
  return { w: header.readUInt32BE(0), h: header.readUInt32BE(4) };
}

// Place an image inside a box, keeping its aspect ratio; `align` positions it
// horizontally ("left", "center", "right") and vertically at the top.
function addImageFit(slide, file, box, altText, align = "center") {
  const { w: pw, h: ph } = pngSize(file);
  const scale = Math.min(box.w / pw, box.h / ph);
  const w = pw * scale;
  const h = ph * scale;
  const x = align === "left" ? box.x : align === "right" ? box.x + box.w - w : box.x + (box.w - w) / 2;
  slide.addImage({ path: file, x, y: box.y, w, h, altText });
  return { x, y: box.y, w, h };
}

// A rounded card: the deck's container for grouped content.
function card(slide, x, y, w, h, fill = C.background2, name = "card") {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0.5 }, rectRadius: 0.08, objectName: name,
  });
}

// The approach badge: a circle with "A" (blue) or "B" (orange), the deck's motif.
function badge(slide, letter, x = 12.15, y = 0.5, d = 0.6) {
  slide.addText(letter, {
    shape: pres.shapes.OVAL, x, y, w: d, h: d, fill: { color: letter === "A" ? C.accent1 : C.accent2 },
    color: C.background1, bold: true, fontSize: 22, align: "center", valign: "middle", margin: 0,
    isTextBox: true, objectName: `badge ${letter}`,
  });
}

// A big number with a short label below it.
function stat(slide, { x, y, w, value, label, color = C.text2, size = 48, labelSize = 15, labelH = 0.95 }) {
  slide.addText(value, { x, y, w, h: size / 60, fontSize: size, bold: true, color, margin: 0,
    valign: "bottom", isTextBox: true, objectName: "stat value" });
  slide.addText(label, { x, y: y + size / 60 + 0.08, w, h: labelH, fontSize: labelSize, color: C.text1,
    margin: 0, valign: "top", isTextBox: true, objectName: "stat label" });
}

// The sliding windows as native shapes: six 5-minute windows starting a minute
// apart, one alert at 10:04, and a time axis. Windows holding the alert are
// orange, the last one is normal.
function windowsDiagram(slide, { x, y, w }) {
  const minute = w / 10;
  const barH = 0.32;
  const gap = 0.07;
  const axisY = y + 6 * (barH + gap) + 0.05;
  const alertX = x + 4 * minute;
  // The alert line goes first so the bars (drawn see-through) sit on top of it.
  slide.addShape(pres.shapes.LINE, { x: alertX, y: y - 0.12, w: 0, h: axisY - y + 0.12,
    line: { color: C.accent2, width: 2.5 }, objectName: "alert time" });
  for (let i = 0; i < 6; i += 1) {
    const anomalous = i <= 4; // windows starting at 10:00 ... 10:04 contain 10:04
    slide.addText(`Window ${i + 1}: ${anomalous ? "anomalous" : "normal"}`, {
      shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.05, x: x + i * minute, y: y + i * (barH + gap),
      w: 5 * minute, h: barH, fill: { color: anomalous ? "FDEEE7" : C.background2, transparency: 15 },
      line: { color: anomalous ? C.accent2 : "C3C2B7", width: 1 }, fontSize: 12, color: C.text1,
      align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `window ${i + 1}`,
    });
  }
  slide.addShape(pres.shapes.LINE, { x, y: axisY, w, h: 0, line: { color: "C3C2B7", width: 1 }, objectName: "time axis" });
  for (let m = 0; m <= 10; m += 2) {
    slide.addText(`10:${String(m).padStart(2, "0")}`, { x: x + m * minute - 0.4, y: axisY + 0.05, w: 0.8, h: 0.3,
      fontSize: 12, color: C.accent6, align: "center", margin: 0, isTextBox: true });
  }
  slide.addText("one alert at 10:04", { x: alertX + 0.08, y: y - 0.45, w: 2.2, h: 0.3, fontSize: 12, bold: true,
    color: C.accent2, margin: 0, isTextBox: true });
  return axisY + 0.4; // bottom of the drawing
}

// A card with a bold heading and body text.
function textCard(slide, { x, y, w, h, head, body, headColor = C.text2, fill = C.background2, size = 15 }) {
  card(slide, x, y, w, h, fill);
  slide.addText(
    [
      { text: head, options: { bold: true, fontSize: size + 3, color: headColor, breakLine: true } },
      { text: body, options: { fontSize: size, color: C.text1 } },
    ],
    { x: x + 0.22, y: y + 0.18, w: w - 0.44, h: h - 0.36, valign: "top", margin: 0,
      paraSpaceAfter: 6, isTextBox: true, objectName: "card text" },
  );
}

// A dark panel of monospace lines; lines marked { alert: true } are orange.
function logPanel(slide, lines, { x, y, w, h, size = 12, fill = C.text2, transparency = 0 }) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, fill: { color: fill, transparency }, line: { color: fill, width: 0.5, transparency },
    rectRadius: 0.08, objectName: "log panel",
  });
  const runs = lines.map((line, i) => ({
    text: line.text,
    options: {
      color: line.alert ? C.accent2 : line.dim ? C.background2 : C.background1,
      bold: Boolean(line.alert || line.bold),
      fontFace: line.mono === false ? undefined : MONO,
      fontSize: line.size || size,
      breakLine: i < lines.length - 1,
    },
  }));
  slide.addText(runs, { x: x + 0.22, y: y + 0.15, w: w - 0.44, h: h - 0.3, valign: "middle", margin: 0,
    paraSpaceAfter: 4, isTextBox: true, objectName: "log lines" });
}

// Bulleted list with consistent spacing.
function bullets(slide, items, box, size = 16) {
  slide.addText(
    items.map((item, i) => ({ text: item, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { ...box, fontSize: size, color: C.text1, valign: "top", margin: 0, paraSpaceAfter: 10,
      isTextBox: true, objectName: "bullets" },
  );
}

// Shared chart styling; returns a fresh object every call (pptxgenjs mutates options).
function chartOptions(extra) {
  return {
    catAxisLabelColor: HEX.accent6, valAxisLabelColor: HEX.accent6,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 13, valAxisLabelFontSize: 12,
    valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" },
    valAxisLineShow: false, catAxisLineShow: true,
    dataLabelFontFace: "+mn-lt", dataLabelFontSize: 12, dataLabelColor: HEX.dk1,
    legendFontFace: "+mn-lt", legendFontSize: 13, legendColor: HEX.dk1,
    titleFontFace: "+mn-lt", titleFontSize: 15, titleColor: HEX.dk1, showTitle: true,
    ...extra,
  };
}

// Every slide's title and notes, collected for speaker_notes.md.
const SCRIPT = [];
let currentTitle = "";

// Put the title in the slide's title placeholder and remember it.
function titled(slide, text) {
  slide.addText(text, { placeholder: "title" });
  currentTitle = text;
}

// Speaker notes: the timing, what to say, what to point at, backup facts and
// the transition, as separate paragraphs. The same text goes to speaker_notes.md.
function notes(slide, { time, say, point, asked, next }) {
  const parts = [`TIME: ${time}`, "", "SAY:", ...say.flatMap((p) => [p, ""])];
  if (point) parts.push(`POINT AT: ${point}`, "");
  if (asked) parts.push("IF ASKED:", ...asked.map((a) => `- ${a}`), "");
  if (next) parts.push(`NEXT: ${next}`);
  const text = parts.join("\n").trim();
  slide.addNotes(text);
  SCRIPT.push({ title: currentTitle, text });
}

// Write the speaker notes of every slide as Markdown, for rehearsing.
function writeScript(file) {
  const lines = [
    "# Speaker notes",
    "",
    "The notes of `presentation.pptx`, slide by slide, to read and rehearse anywhere. In PowerPoint",
    "they appear under each slide in Presenter View. This file is written by `build_deck.js`: edit",
    "the notes there and rebuild.",
    "",
  ];
  SCRIPT.forEach(({ title, text }, i) => {
    lines.push(`## ${i + 1}. ${title}`, "");
    for (const line of text.split("\n")) {
      // Section labels in bold; everything else as written.
      lines.push(/^(TIME|SAY|POINT AT|IF ASKED|NEXT):/.test(line) ? line.replace(/^([A-Z ]+):/, "**$1:**") : line);
    }
    lines.push("");
  });
  fs.writeFileSync(file, lines.join("\n").replace(/\n{3,}/g, "\n\n"));
}

// ------------------------------------------------------------------ data

const METHOD_ROWS = [
  ["logbert", "A   LogBERT", "A"],
  ["logbert_full", "A+  LogBERT, full coverage", "A"],
  ["llm_zero_shot", "B1  LLM zero-shot", "B"],
  ["llm_few_shot", "B2  LLM few-shot", "B"],
  ["llm_rule_based", "B3  LLM rule-based", "B"],
  ["keyword_rules", "B4  Keyword rules, no LLM", "B"],
];
// Exact test metrics per method from results/<method>.json (figure_data.json
// rounds them to four decimals, which would round twice in the table).
const RESULTS = Object.fromEntries(
  METHOD_ROWS.map(([key]) => {
    const result = JSON.parse(fs.readFileSync(path.join(ROOT, "results", `${key}.json`), "utf8"));
    return [key, { ...result.test, inference_seconds: result.inference_seconds }];
  }),
);
const fmt3 = (x) => Number(x).toFixed(3);
const fmtInt = (x) => Math.round(x).toLocaleString("en-US");
function duration(seconds) {
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 600) return `${Math.round(seconds)} s`;
  return `${Math.round(seconds / 60)} min`;
}

// =================================================================== slides

// 1 ---------------------------------------------------------------- title
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
  titled(s, "Finding failures in supercomputer logs");
  s.addText("LogBERT versus a rule-guided LLM, compared under one fair setup", { placeholder: "body" });
  s.addText(
    [
      { text: "CSCI 390  ·  Natural Language Processing and Applications  ·  Dr. Qi Li", options: { breakLine: true } },
      { text: "github.com/Amshu-0/claude-nlp" },
    ],
    { placeholder: "course" },
  );
  s.addText("Seven real lines from the BGL log. Orange: alerts.", {
    x: 7.2, y: 1.75, w: 5.53, h: 0.35, fontSize: 13, color: C.background2, margin: 0, isTextBox: true,
  });
  logPanel(
    s,
    [
      { text: "-        instruction cache parity error corrected" },
      { text: "-        generating core.304" },
      { text: "-        1347195 double-hummer alignment exceptions" },
      { text: "KERNDTLB data TLB error interrupt", alert: true },
      { text: "-        ciod: LOGIN chdir(…) failed: Permission denied" },
      { text: "APPOUT   ciod: LOGIN chdir(…) failed: Input/output error", alert: true },
      { text: "-        10732 total interrupts. 0 critical input …" },
    ],
    { x: 7.2, y: 2.2, w: 5.53, h: 3.0, size: 10, fill: C.background1, transparency: 90 },
  );
  notes(s, {
    time: "about 20 seconds",
    say: [
      "Hello, my name is [your name]. My project asks a practical question: how should we find failures in the logs of a supercomputer?",
      "I compare two very different approaches from recent papers. The first is LogBERT, a BERT-style model that learns patterns in sequences of log messages. The second follows a 2025 paper called LogRules: a large language model reads the text of the messages and is guided by rules it wrote itself. I ran both on the same data under the same rules, and the result surprised me.",
    ],
    point: "the log lines on the right. They are real lines from the dataset; the two orange ones are alerts.",
    next: "Let me start with what one of these lines contains.",
  });
}

// 2 --------------------------------------------------------- a log line
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  titled(s, "What a log line contains");
  // The fields of one real line as chips, with what each field is below it.
  const fields = [
    { text: "KERNDTLB", label: "alert tag", fill: C.accent2, color: C.background1 },
    { text: "1118536327", label: "time" },
    { text: "R30-M0-N9-C:J16-U01", label: "node" },
    { text: "RAS", label: "type" },
    { text: "KERNEL", label: "component" },
    { text: "FATAL", label: "level" },
    { text: "data TLB error interrupt", label: "content", fill: C.accent1, color: C.background1 },
  ];
  const charW = 0.112; // inches per character of 13.5 pt Courier New
  const widths = fields.map((f) => Math.max(f.text.length * charW + 0.32, 0.95));
  const gap = (CW - widths.reduce((a, b) => a + b, 0)) / (fields.length - 1);
  let x = MX;
  fields.forEach((f, i) => {
    s.addText(f.text, {
      shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06, x, y: 1.7, w: widths[i], h: 0.62,
      fill: { color: f.fill || C.background2 }, line: { color: f.fill || C.background2, width: 0.5 },
      fontFace: MONO, fontSize: 13.5, bold: true, color: f.color || C.text1, align: "center",
      valign: "middle", margin: 0, isTextBox: true, objectName: `field ${f.label}`,
    });
    s.addText(f.label, { x, y: 2.4, w: widths[i], h: 0.35, fontSize: 13, color: C.accent6,
      align: "center", margin: 0, isTextBox: true });
    x += widths[i] + gap;
  });
  textCard(s, {
    x: MX, y: 3.05, w: 5.9, h: 1.65, headColor: C.accent2,
    head: "The alert tag is the label",
    body: "Administrators marked every alert with one of 41 categories; '-' means normal. The pipeline cuts the tag off before any model reads the line.",
  });
  textCard(s, {
    x: MX + 6.23, y: 3.05, w: 5.9, h: 1.65, headColor: C.accent1,
    head: "The content is what the models read",
    body: "LogBERT sees it as a template ID; the language model reads its words.",
  });
  const third = CW / 3;
  stat(s, { x: MX, y: 5.0, w: third - 0.2, value: "4,747,963", label: "log lines, June 2005 to January 2006" });
  stat(s, { x: MX + third, y: 5.0, w: third - 0.2, value: "348,460", label: "alerts: 7.3% of all lines", color: C.accent2 });
  stat(s, { x: MX + 2 * third, y: 5.0, w: third - 0.2, value: "BlueGene/L", label: "supercomputer, Lawrence Livermore National Laboratory" });
  notes(s, {
    time: "about 1 minute",
    say: [
      "This is one real line from our dataset, BGL: the system log of the BlueGene/L supercomputer at Lawrence Livermore National Laboratory. Each line has ten fields; I show the important ones: a time stamp, the hardware node that wrote it, the software component, a severity level, and the free-text message, here 'data TLB error interrupt'.",
      "The first field is special. It is the alert tag: system administrators marked every message that should raise an alert with a category, here KERNDTLB, and normal messages get a dash. That tag is our ground truth. So the first thing the pipeline does is cut it off. If a model could see it, it would 'detect' every alert without learning anything. This is called label leakage, and a test in the code checks that it cannot happen.",
      "The data set is big: 4.7 million lines over seven months, 348,460 of them alerts, which is 7.3 percent.",
    ],
    point: "the orange chip (the label, removed first) and the blue chip (what the models read).",
    asked: [
      "The ten fields: alert tag, epoch time, date, node, full time, node again, type, component, level, content.",
      "There are 41 alert categories. The most common, KERNDTLB, has 152,734 messages.",
    ],
    next: "Why is finding the alerts hard?",
  });
}

// 3 ------------------------------------------------------ why it is hard
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  titled(s, "Why this is hard");
  const w = (CW - 0.5) / 3;
  const cards = [
    { value: "7.3%", color: C.accent2, head: "Alerts are rare",
      body: "Always answering 'normal' is 92.7% accurate and useless. So we measure precision, recall and F1." },
    { value: "79.6%", color: C.accent1, head: "The log changes",
      body: "of the test messages use a template never seen in training." },
    { value: "error ≠ alert", color: C.text2, head: "Words mislead",
      body: "'instruction cache parity error corrected' is normal. 'LOGIN … failed: Input/output error' was an alert." },
  ];
  cards.forEach((c, i) => {
    const x = MX + i * (w + 0.25);
    card(s, x, TOP, w, 2.95);
    s.addText(c.value, { x: x + 0.25, y: TOP + 0.2, w: w - 0.5, h: 0.95, fontSize: c.value.length > 6 ? 36 : 48,
      bold: true, color: c.color, margin: 0, valign: "bottom", isTextBox: true });
    s.addText(
      [
        { text: c.head, options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
        { text: c.body, options: { fontSize: 15, color: C.text1 } },
      ],
      { x: x + 0.25, y: TOP + 1.25, w: w - 0.5, h: 1.6, valign: "top", margin: 0, paraSpaceAfter: 6, isTextBox: true },
    );
  });
  card(s, MX, 4.75, CW, 1.95, C.text2, "research question");
  s.addText(
    [
      { text: "Research question", options: { bold: true, fontSize: 15, color: C.accent2, breakLine: true } },
      { text: "How does a BERT-style model that learns sequences of log-template IDs compare with a rule-guided large language model that reads the log text itself, in identifying anomalous system events?",
        options: { fontSize: 20, color: C.background1 } },
    ],
    { x: MX + 0.35, y: 4.9, w: CW - 0.7, h: 1.65, valign: "middle", margin: 0, paraSpaceAfter: 6, isTextBox: true },
  );
  notes(s, {
    time: "about 1 minute",
    say: [
      "Three things make this hard. First, alerts are rare: only 7.3 percent of the messages. A detector that always says 'normal' is 92.7 percent accurate and completely useless. That is why I report precision, recall and F1, never accuracy.",
      "Second, the log changes over time. Software is updated and new kinds of messages appear. In our test period, almost 80 percent of the messages use a message type the models never saw during training.",
      "Third, words can mislead. 'Instruction cache parity error corrected' contains the word error but is normal, because the hardware fixed itself. 'LOGIN failed: Input/output error' looks routine, but it was an alert.",
      "So the research question is: how does a BERT-style model that learns sequences of log-template IDs compare with a rule-guided language model that reads the text, at finding these anomalous events?",
    ],
    point: "the three numbers, then the question at the bottom.",
    next: "Here are the two approaches side by side.",
  });
}

// 4 ------------------------------------------------------ two approaches
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  titled(s, "Two approaches, one fair setup");
  const w = (CW - 0.3) / 2;
  const sides = [
    { letter: "A", title: "LogBERT", sub: "Guo et al., 2021 · reproduced", rows: [
      ["Reads", "sequences of template IDs in a 5-minute window; never the words"],
      ["Learns from", "normal windows only; it never sees an alert"],
      ["Model", "a small Transformer, 1.67 M parameters, trained from scratch"],
      ["Flags", "a window whose hidden keys it cannot predict"],
    ] },
    { letter: "B", title: "Rule-guided LLM", sub: "LogRules, Huang et al., 2025 · adapted", rows: [
      ["Reads", "the words of each template; never the sequence"],
      ["Learns from", "labelled training templates, turned into verified rules"],
      ["Model", "Qwen2.5-7B-Instruct, 7.6 B parameters, used as it is"],
      ["Flags", "templates the rules (or the LLM applying them) call abnormal"],
    ] },
  ];
  sides.forEach((side, i) => {
    const x = MX + i * (w + 0.3);
    card(s, x, TOP, w, 3.75);
    badge(s, side.letter, x + 0.25, TOP + 0.25, 0.58);
    s.addText(
      [
        { text: side.title, options: { bold: true, fontSize: 22, color: C.text2, breakLine: true } },
        { text: side.sub, options: { fontSize: 13, color: C.accent6 } },
      ],
      { x: x + 1.0, y: TOP + 0.2, w: w - 1.2, h: 0.75, margin: 0, valign: "middle", isTextBox: true },
    );
    side.rows.forEach(([label, text], r) => {
      const y = TOP + 1.15 + r * 0.62;
      s.addText(label, { x: x + 0.25, y, w: 1.45, h: 0.55, fontSize: 14, bold: true, color: side.letter === "A" ? C.accent1 : C.accent2,
        margin: 0, valign: "top", isTextBox: true });
      s.addText(text, { x: x + 1.7, y, w: w - 1.95, h: 0.55, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true });
    });
  });
  card(s, MX, 5.5, CW, 1.2, C.text2, "shared setup");
  s.addText(
    [
      { text: "Shared setup", options: { bold: true, color: C.accent2, breakLine: true } },
      { text: "same BGL data  ·  chronological 60 / 20 / 20 split  ·  the same 5,742 test windows  ·  one scoring script  ·  label removed first",
        options: { color: C.background1 } },
    ],
    { x: MX + 0.35, y: 5.5, w: CW - 0.7, h: 1.2, fontSize: 16, valign: "middle", margin: 0, paraSpaceAfter: 4, isTextBox: true },
  );
  notes(s, {
    time: "about 1 minute",
    say: [
      "On the left, Approach A: LogBERT, from 2021. It never reads words. It sees each message only as an ID number and learns which sequences of IDs are normal. It trains only on windows without alerts, so it never sees a failure. It is a small Transformer, 1.7 million parameters, trained from scratch.",
      "On the right, Approach B, inspired by LogRules from 2025. It reads the words of each message type. A large language model, Qwen 2.5 with 7 billion parameters, writes rules from labelled training examples, and the rules guide the classification. Nothing is trained here.",
      "The two papers report numbers on different data samples and evaluation units, so their published numbers cannot be compared. The heart of this project is the shared setup at the bottom: the same data, a chronological split, the same 5,742 test windows, and one script that scores both.",
    ],
    asked: [
      "How different the papers' setups are: DeepLog scores F1 86 on BGL in the LogBERT paper and 16.8 in the LogRules paper.",
    ],
    next: "First, how both approaches see the data.",
  });
}

// 5 ------------------------------------------------ templates and windows
pres.addSection({ title: "Implementation" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
  titled(s, "From log lines to templates and windows");
  s.addText("Drain groups messages into templates", { x: MX, y: TOP, w: 5.7, h: 0.4, fontSize: 18, bold: true,
    color: C.text2, margin: 0, isTextBox: true });
  logPanel(
    s,
    [
      { text: "generating core.304", dim: true },
      { text: "generating core.2276", dim: true },
      { text: "generating core.5089", dim: true },
      { text: "→ generating core.<*>", bold: true },
      { text: " " },
      { text: "ciod: LOGIN chdir(/p/gb1/…/2183)", dim: true },
      { text: "      failed: Input/output error", dim: true },
      { text: "→ ciod: LOGIN <*> failed: <*> <*>", bold: true },
    ],
    { x: MX, y: TOP + 0.5, w: 5.7, h: 3.05, size: 13 },
  );
  stat(s, { x: MX, y: 5.2, w: 5.7, value: "4.7 M → 923", size: 40, labelSize: 14,
    label: "messages become 923 templates; each template ID is a 'log key', like a word in a vocabulary" });
  s.addText("Windows: 5 minutes, sliding by 1 minute", { x: 6.75, y: TOP, w: 6.0, h: 0.4, fontSize: 18, bold: true,
    color: C.text2, margin: 0, isTextBox: true });
  const bottom = windowsDiagram(s, { x: 6.75, y: TOP + 1.0, w: 5.98 });
  bullets(
    s,
    [
      "A window is anomalous if it contains at least one alert.",
      "One alert makes up to five overlapping windows anomalous.",
      "Test period: 5,742 windows, 495 of them anomalous (8.6%).",
    ],
    { x: 6.75, y: bottom + 0.2, w: 5.98, h: 6.85 - bottom - 0.2 },
    15,
  );
  notes(s, {
    time: "about 1 minute",
    say: [
      "Messages that differ only in their numbers, like 'generating core.304' and 'generating core.2276', come from the same line of source code. The Drain algorithm finds this shared shape, the template, and replaces the parts that vary with a star placeholder. 4.7 million messages become 923 templates. Each template gets an ID, called a log key, which plays the role of a word in a vocabulary. LogBERT reads these IDs; the language model reads the template text.",
      "Then the timeline is cut into five-minute windows that slide by one minute, as in the LogBERT paper. A window is anomalous if it contains at least one alert, so one alert makes several overlapping windows anomalous, as the diagram shows. Every method must give a verdict for every one of the 5,742 test windows.",
    ],
    point: "the template examples, then the orange alert in the window diagram.",
    asked: [
      "Drain settings come from the LogBERT repository: depth 3, similarity 0.3, masking of numbers, hex values and IP addresses.",
      "Drain over-merges a little: on the 2,000-line sample it finds 100 templates against 120 in the hand-made ground truth. The LOGIN example shows the cost: the words that decide the label become stars.",
    ],
    next: "How we split the data turned out to be decisive.",
  });
}

// 6 ---------------------------------------------------------- drift
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
  titled(s, "The log changes over time");
  const unseen = DATA.unseen_templates.percent_by_split;
  const panels = DATA.window_scores.panels;
  const normalWithUnseen = panels["Normal windows with a template never seen in training"].windows;
  const normalWithout = panels["Normal windows, every template seen in training"].windows;
  const testWindowsShare = (100 * normalWithUnseen) / (normalWithUnseen + normalWithout);
  const VAL_NORMAL_WINDOWS_SHARE = 9.6; // docs/report.md, Section 2 ("9.6% of normal validation windows")
  s.addChart(
    pres.charts.BAR,
    [
      { name: "Messages whose template never appears in training", labels: ["Training", "Validation", "Test"],
        values: [0, unseen.val, unseen.test] },
      { name: "Normal windows that contain such a template", labels: ["Training", "Validation", "Test"],
        values: [0, VAL_NORMAL_WINDOWS_SHARE, testWindowsShare] },
    ],
    chartOptions({
      x: MX, y: TOP, w: 7.2, h: 5.25, barDir: "col", barGrouping: "clustered", barGapWidthPct: 70,
      chartColors: [HEX.accent1, "9EC5F4"], showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: '0.0"%"', valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25,
      valAxisLabelFormatCode: '0"%"', showLegend: true, legendPos: "b",
      title: "Never-seen templates, by period",
    }),
  );
  const x = 8.15;
  const w = W - MX - x;
  s.addText(
    [
      { text: "Split by time, never shuffled", options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
      { text: "Train on the first 60% of the timeline (June to October 2005), tune on the next 20%, test on the last 20% (November 2005 to January 2006). A random split would let a model learn from the future.",
        options: { fontSize: 15, color: C.text1 } },
    ],
    { x, y: TOP, w, h: 2.2, valign: "top", margin: 0, paraSpaceAfter: 6, isTextBox: true },
  );
  s.addTable(
    [
      [{ text: "Period", options: { bold: true } }, { text: "Messages", options: { bold: true, align: "right" } },
        { text: "Windows", options: { bold: true, align: "right" } }],
      ["Training", { text: "3,569,472", options: { align: "right" } }, { text: "21,112", options: { align: "right" } }],
      ["Validation", { text: "696,160", options: { align: "right" } }, { text: "10,478", options: { align: "right" } }],
      ["Test", { text: "482,331", options: { align: "right" } }, { text: "5,742", options: { align: "right" } }],
    ],
    { x, y: TOP + 2.35, w, colW: [1.5, w - 2.75, 1.25], rowH: 0.38, fontSize: 14, color: C.text1,
      border: { type: "solid", pt: 0.75, color: GRID }, valign: "middle", margin: [0, 0.08, 0, 0.08] },
  );
  textCard(s, { x, y: 5.4, w, h: 1.35, size: 14, head: "New in December",
    body: "Two periodic diagnostics appear: 135,092 test messages each, never seen before October 2005." });
  notes(s, {
    time: "about 40 seconds",
    say: [
      "We split by time: the first 60 percent of the timeline for training, the next 20 percent for validation, the last 20 percent for testing. Never at random, because a random split lets a model learn from the future and inflates its scores.",
      "The chart shows the consequence. In the test period, 79.6 percent of all messages use a template that never appears in training, and 71 percent of the normal test windows contain at least one such template. Keep this in mind: it explains most of the results.",
    ],
    point: "the dark blue Test bar, then the light blue one next to it.",
    asked: [
      "Validation is much milder: 8.9 percent of its messages and 9.6 percent of its normal windows. So settings tuned on validation do not see the full drift.",
      "The two new diagnostics are '… total interrupts …' and '… microseconds spent in the rbs signal handler …'.",
    ],
    next: "Now the two approaches in more detail, starting with LogBERT.",
  });
}

// 7 --------------------------------------------------- LogBERT training
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
  titled(s, "LogBERT learns “normal” by filling in blanks");
  badge(s, "A");
  addImageFit(s, path.join(DIAGRAMS, "logbert-train.png"), { x: MX, y: TOP + 0.05, w: CW, h: 2.65 },
    "LogBERT training: hide half the keys of a normal window, predict them, and pull the sequence summary towards a centre");
  const w = (CW - 0.5) / 3;
  const cards = [
    { head: "Normal windows only", body: "19,029 normal training windows, 11,615 distinct sequences. LogBERT never sees an alert." },
    { head: "Two losses", body: "Predict the hidden keys (MLKP) and pull each sequence's summary vector towards one centre (VHM): loss = MLKP + 0.1 × VHM." },
    { head: "A small Transformer", body: "256-number embeddings, 2 self-attention layers with 4 heads: 1.67 M parameters. 60 minutes on 4 CPU cores." },
  ];
  cards.forEach((c, i) => textCard(s, { x: MX + i * (w + 0.25), y: 4.5, w, h: 2.2, head: c.head, body: c.body,
    headColor: C.accent1 }));
  notes(s, {
    time: "about 1 minute",
    say: [
      "LogBERT borrows BERT's training trick: fill in the blanks. In a normal window it hides half of the log keys and learns to predict them from the keys around them, like guessing a missing word in a sentence. This is called masked log key prediction.",
      "Under the hood, each key becomes a vector of 256 numbers, an embedding. Two Transformer layers with self-attention let every position look at every other position in the window. A second loss, the hypersphere loss, pulls a summary vector of every normal sequence towards one centre point, so normal windows cluster together.",
      "It trains only on normal windows: 19,029 of them, which contain 11,615 distinct sequences once duplicates are removed. Training took one hour on four CPU cores.",
    ],
    point: "the hidden keys in the diagram, then the two losses.",
    asked: [
      "Total loss: masked-key cross-entropy plus 0.1 times the hypersphere loss, as in the LogBERT repository.",
      "2 layers, 4 heads, hidden size 256, feed-forward 1024, maximum length 512; the best dev loss (1.02) came at epoch 5.",
      "The hidden keys are replaced 80% of the time by a mask token, 10% by a random key and 10% kept, exactly as in BERT.",
    ],
    next: "How does it decide that a window is anomalous?",
  });
}

// 8 -------------------------------------------------- LogBERT detection
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
  titled(s, "LogBERT flags windows it cannot predict");
  badge(s, "A");
  const diagram = addImageFit(s, path.join(DIAGRAMS, "logbert-detect.png"), { x: MX, y: TOP, w: CW, h: 1.75 },
    "LogBERT detection: rank each hidden key; a key outside the top g guesses is missed; more than r misses make the window anomalous");
  const y = diagram.y + diagram.h + 0.3;
  const heat = addImageFit(s, path.join(FIGURES, "04_logbert_tuning_grid.png"), { x: MX, y, w: 6.3, h: 6.85 - y },
    "Validation F1 for every (g, r) pair; the chosen pair g = 12, r = 3 is marked", "left");
  bullets(
    s,
    [
      "g and r are chosen on the validation period: g = 12, r = 3 (validation F1 0.599).",
      "The choice is fragile: only 16 of 231 pairs come within 0.02 of the best.",
      "The original code tunes on the test set. Here that would report F1 0.475 instead of 0.293.",
      "Our extension A+ checks every key once instead of a random half: F1 0.305.",
    ],
    { x: heat.x + heat.w + 0.45, y: y + 0.1, w: W - MX - (heat.x + heat.w + 0.45), h: 6.85 - y },
    16,
  );
  notes(s, {
    time: "about 50 seconds",
    say: [
      "At test time LogBERT hides keys again and ranks the true key among its 176 guesses. If the true key is not among its top g guesses, it is a miss, and a window with more than r misses is anomalous. One detail matters a lot: a template it never saw in training becomes the 'unknown' token and is always a miss.",
      "g and r are chosen on the validation period: g equals 12 and r equals 3. The heat map shows how fragile that choice is: only 16 of 231 settings come close to the best. The original LogBERT code chooses its threshold on the test set; doing that here would raise F1 from 0.29 to 0.48, which would not be honest.",
      "I also added an extension, A-plus, that checks every key once instead of a random half.",
    ],
    point: "the 'unk always missed' box, then the dot and the outlined cells in the heat map.",
    asked: [
      "Ranks are stored once, so all 231 settings are evaluated with one pass of the model.",
      "Full coverage: F1 0.293 → 0.305 and 2,159 → 1,975 false alarms, at twice the inference time (78 s → 156 s).",
    ],
    next: "Approach B works completely differently.",
  });
}

// 9 ---------------------------------------------- Approach B conditions
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
  titled(s, "One LLM, four kinds of guidance");
  badge(s, "B");
  const w = (CW - 0.75) / 4;
  const conditions = [
    ["B1  Zero-shot", "The template and a question: abnormal or normal?"],
    ["B2  Few-shot", "Plus 8 labelled training templates as examples."],
    ["B3  Rule-based", "Plus the 30 verified rules. The model must cite the rule it applied."],
    ["B4  Keyword rules", "The same 30 rules as plain string matching. No LLM: the control experiment."],
  ];
  conditions.forEach(([head, body], i) => textCard(s, { x: MX + i * (w + 0.25), y: TOP, w, h: 2.0, head, body,
    headColor: C.accent2, size: 15, fill: i === 3 ? "FDEEE7" : C.background2 }));
  s.addText("The model", { x: MX, y: 3.9, w: 6.0, h: 0.4, fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true });
  bullets(
    s,
    [
      "Qwen2.5-7B-Instruct, 7.6 B parameters, 4-bit (4.7 GB)",
      "llama.cpp on 4 CPU cores, temperature 0: reproducible",
      "6 to 7 seconds per template; every answer cached",
      "Each of the 472 test templates classified once; the label is copied to all its messages",
    ],
    { x: MX, y: 4.45, w: 6.0, h: 2.4 },
    16,
  );
  s.addText("A real B3 answer", { x: 7.0, y: 3.9, w: 5.73, h: 0.4, fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true });
  logPanel(
    s,
    [
      { text: "Log: <*> TLB error interrupt", dim: true },
      { text: '{"label": "abnormal", "rule": 24,', bold: true },
      { text: ' "reason": "TLB error interrupt is', bold: true },
      { text: '            typically abnormal"}', bold: true },
    ],
    { x: 7.0, y: 4.45, w: 5.73, h: 2.2, size: 14 },
  );
  notes(s, {
    time: "about 50 seconds",
    say: [
      "Approach B uses Qwen 2.5 7B Instruct, an open language model. With no GPU and no API key, I ran a 4-bit version through llama.cpp on the CPU, with greedy decoding so the answers are reproducible.",
      "I test four conditions. B1, zero-shot: just the template and a question. B2, few-shot: plus eight labelled training examples. B3, rule-based: plus 30 rules that were induced and verified on the training data; the model must cite the rule it used. And B4, the control experiment: the same 30 rules applied as plain keyword matching, with no language model at all. If B3 is not better than B4, the rules do the work, not the model.",
      "Each of the 472 test templates is classified once, and its label is copied to all of its messages.",
    ],
    point: "the four cards from left to right, then the JSON answer.",
    asked: [
      "The model reads about 23 tokens per second and writes about 5 on 4 cores. Ending every prompt with the log line lets the server reuse the shared beginning: about 20% faster.",
      "All 1,416 classification answers were valid; none fell back to the default label.",
      "Classifying templates instead of messages costs almost nothing here: the best a template-level method could reach is F1 0.996.",
    ],
    next: "Where do the rules come from?",
  });
}

// 10 --------------------------------------------------------- the rules
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
  titled(s, "The LLM writes rules; the data checks them");
  badge(s, "B");
  const diagram = addImageFit(s, path.join(DIAGRAMS, "logrules.png"), { x: MX, y: TOP, w: CW, h: 2.4 },
    "Rule induction: 442 labelled templates, 160 candidate rules, 99 never wrong on training data, top 30 used by B3 and B4");
  const y = Math.max(diagram.y + diagram.h + 0.3, 4.15);
  const w = (CW - 0.3) / 2;
  card(s, MX, y, w, 6.75 - y);
  s.addText(
    [
      { text: "A verified rule (rank 4 of 99)", options: { bold: true, fontSize: 18, color: C.accent2, breakLine: true } },
      { text: "“Messages mentioning 'FAILED', 'terminated for reason', 'kernel panic', and 'machine check interrupt' are indicative of system failures.”",
        options: { italic: true, fontSize: 15, color: C.text1, breakLine: true } },
      { text: "Keywords: lustre mount failed · rts: kernel terminated for reason · kernel panic · machine check interrupt",
        options: { fontSize: 14, color: C.text1, breakLine: true } },
      { text: "Correct on 9 of 9 matching training templates.", options: { fontSize: 14, color: C.accent6 } },
    ],
    { x: MX + 0.25, y: y + 0.15, w: w - 0.5, h: 6.75 - y - 0.3, valign: "top", margin: 0, paraSpaceAfter: 8, isTextBox: true },
  );
  card(s, MX + w + 0.3, y, w, 6.75 - y);
  s.addText("Weak spots, visible already", { x: MX + w + 0.55, y: y + 0.15, w: w - 0.5, h: 0.4, fontSize: 18, bold: true,
    color: C.text2, margin: 0, isTextBox: true });
  bullets(
    s,
    [
      "Vague sentences: “Indicates a system error or failure.”",
      "Near-duplicates: rules 8, 9 and 10 all key on ‘terminated’.",
      "Odd keywords that happen to pass: ‘pro’, ‘<*>:<*>eeeeeee’.",
      "Verification checks the keywords, not the sentence the model reads.",
    ],
    { x: MX + w + 0.55, y: y + 0.65, w: w - 0.5, h: 6.75 - y - 0.8 },
    14,
  );
  notes(s, {
    time: "about 50 seconds",
    say: [
      "The language model writes the rules. It sees the 442 labelled training templates in batches of 20, half normal and half abnormal, and proposes rules with keyword phrases copied from the logs. That gave 160 candidate rules.",
      "Then the data checks them. Each rule is applied to every training template by keyword matching, and only rules that are never wrong are kept: 99 of the 160. They are ranked by how many templates they cover, and the top 30 are used, as in the LogRules paper.",
      "On the left is a real rule. The weak spots are already visible: vague sentences, near-duplicates, and odd keywords. Most importantly, verification checks the keywords, not the sentence the model will read. That will matter in a moment.",
    ],
    point: "the funnel 160 → 99 → 30, then the rule's sentence versus its keywords.",
    asked: [
      "LogRules used GPT-4o-mini to write rules. I used the same Qwen model for both roles because no API key was available; the proposal allowed a comparable model.",
      "Rule induction took 50 minutes for 42 batches. Keywords shorter than three letters or digits are dropped automatically.",
      "LogRules' third stage, fine-tuning a small model to follow rules (CPO), is not reproduced.",
    ],
    next: "Briefly, how all this is implemented.",
  });
}

// 11 ---------------------------------------------------- implementation
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
  titled(s, "One pipeline, tested end to end");
  const diagram = addImageFit(s, path.join(DIAGRAMS, "run-flow.png"), { x: MX, y: TOP, w: CW, h: 2.6 },
    "The pipeline: preprocess, then LogBERT and rule induction in parallel, one comparison on the same windows, analysis, results");
  const y = Math.max(diagram.y + diagram.h + 0.45, 4.35);
  const w = (CW - 0.75) / 4;
  const facts = [
    ["28", "Python modules in one package, logad: data, logbert, llm, rules, evaluation"],
    ["1", "YAML configuration for the whole experiment; any value can be overridden"],
    ["148", "automated tests, offline, about 10 s; CI on every push"],
    ["1,458", "LLM calls, all cached and committed: Approach B replays without a model"],
  ];
  facts.forEach(([value, label], i) => {
    const x = MX + i * (w + 0.25);
    card(s, x, y, w, 6.75 - y);
    stat(s, { x: x + 0.25, y: y + 0.2, w: w - 0.5, value, label, size: 44, labelSize: 14 });
  });
  notes(s, {
    time: "about 30 seconds",
    say: [
      "Everything is one Python package with one command per step, driven by a single configuration file. The diagram is the actual pipeline: preprocessing, then the two approaches in parallel, then one comparison script that refuses any method that did not score exactly the same windows.",
      "There are 148 automated tests and continuous integration on every push. And every result, including the trained model, the rules and every language-model answer, is committed to the repository, so all numbers can be checked, and Approach B can even be replayed without the model.",
    ],
    asked: [
      "The code is in src/logad with five subpackages; scripts/ has one entry point per step; docs/guide.md explains every file.",
      "The comparison script recomputes all metrics from the saved predictions and rejects a method that skipped or added windows.",
    ],
    next: "So, the results.",
  });
}

// 12 ----------------------------------------------------------- results
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
  titled(s, "Results: the rules win, without the LLM");
  const header = ["Method", "Precision", "Recall", "F1", "False alarms", "Time"].map((text, i) => ({
    text, options: { bold: true, color: C.background1, fill: { color: C.text2 }, align: i === 0 ? "left" : "right" },
  }));
  const rows = METHOD_ROWS.map(([key, label]) => {
    const r = RESULTS[key];
    const best = key === "keyword_rules";
    const cell = (text, align = "right") => ({ text, options: { align, bold: best, fill: { color: best ? "FDEEE7" : C.background1 } } });
    return [cell(label, "left"), cell(fmt3(r.precision)), cell(fmt3(r.recall)), cell(fmt3(r.f1)),
      cell(fmtInt(r.fp)), cell(duration(r.inference_seconds))];
  });
  s.addTable([header, ...rows], {
    x: MX, y: TOP, w: 7.35, colW: [2.55, 1.0, 0.82, 0.73, 1.35, 0.9], rowH: 0.5, fontSize: 14, color: C.text1,
    border: { type: "solid", pt: 0.75, color: GRID }, valign: "middle", margin: [0, 0.08, 0, 0.08],
  });
  s.addText(
    "Test period: 5,742 windows, 495 anomalous (8.6%). Time: a verdict for every test window on 4 CPU cores. The best any template-level method could reach: F1 0.996. In the chart, blue is Approach A and orange is Approach B.",
    { x: MX, y: 5.2, w: 7.35, h: 0.9, fontSize: 13, color: C.accent6, margin: 0, valign: "top", isTextBox: true },
  );
  const order = [...METHOD_ROWS].reverse(); // a bar chart lists its first category at the bottom
  s.addChart(
    pres.charts.BAR,
    [{ name: "F1 on the test windows", labels: order.map(([, label]) => label.replace(/\s+/g, " ").replace(", no LLM", "")),
      values: order.map(([key]) => RESULTS[key].f1) }],
    chartOptions({
      x: 8.25, y: TOP - 0.05, w: W - MX - 8.25, h: 5.35, barDir: "bar", barGapWidthPct: 55,
      chartColors: order.map(([, , approach]) => (approach === "A" ? HEX.accent1 : HEX.accent2)),
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0,
      valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00", showLegend: false,
      catAxisLabelFontSize: 12, title: "F1 on the test windows",
    }),
  );
  notes(s, {
    time: "about 1 minute",
    say: [
      "Here is the main table on the 5,742 test windows. Every method finds about 90 percent or more of the anomalous windows: recall is high everywhere. What separates them is precision: how many of their alarms are real.",
      "LogBERT has an F1 of 0.29: it raises 2,159 false alarms. The language model alone, zero-shot, is even worse: it flags almost everything. Examples help, to 0.48; rules help more, to 0.64.",
      "But the best method by far is B4, the same rules as plain keyword matching: F1 0.956 with zero false alarms, in 8 milliseconds instead of an hour. So the answer to the research question has two parts: the rule-guided approach beats LogBERT, but the rules, not the language model, do the work. The next slides explain why.",
    ],
    point: "the B4 row, then the longest orange bar.",
    asked: [
      "Message-level F1, the LogRules metric: B4 0.984, B3 0.491, B2 0.444, B1 0.180.",
      "One-off costs: 60 minutes to train LogBERT, 50 minutes to induce the rules.",
      "Accuracy would hide this: LogBERT is 62% accurate, and never flagging anything would be 91% accurate.",
    ],
    next: "Why does LogBERT do so badly?",
  });
}

// 13 -------------------------------------------- why LogBERT loses
pres.addSection({ title: "Errors and lessons" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Errors and lessons" });
  titled(s, "LogBERT flags whatever is new");
  badge(s, "A");
  const groups = DATA.key_misses.groups;
  s.addChart(
    pres.charts.BAR,
    [
      { name: "Normal messages", labels: ["Template seen in normal training", "Template new to LogBERT"],
        values: [groups.known_normal.percent_missed, groups.new_normal.percent_missed] },
      { name: "Alert messages", labels: ["Template seen in normal training", "Template new to LogBERT"],
        values: [groups.known_alert.percent_missed, groups.new_alert.percent_missed] },
    ],
    chartOptions({
      x: MX, y: TOP, w: 6.75, h: 5.3, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
      chartColors: [HEX.accent1, HEX.accent2], showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: '0.0"%"', valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25,
      valAxisLabelFormatCode: '0"%"', showLegend: true, legendPos: "b",
      title: `Keys LogBERT fails to predict (true key not in its top ${DATA.key_misses.g})`,
    }),
  );
  const alertKeys = groups.known_alert.keys + groups.new_alert.keys;
  const newAlertShare = Math.round((100 * groups.new_alert.keys) / alertKeys);
  const x = 7.75;
  const w = W - MX - x;
  stat(s, { x, y: TOP, w, value: `${newAlertShare}%`, color: C.accent2, size: 44, labelSize: 15,
    label: "of the alert keys come from templates LogBERT never saw: it detects newness, not failure" });
  stat(s, { x, y: TOP + 1.85, w, value: "57.7% vs 0.3%", color: C.accent1, size: 36, labelSize: 15,
    label: "of normal test windows flagged, with and without a never-seen template" });
  stat(s, { x, y: TOP + 3.6, w, value: "99.8%", color: C.text2, size: 44, labelSize: 15,
    label: "of its 2,159 false alarms contain a never-seen template" });
  notes(s, {
    time: "about 1 minute",
    say: [
      "To find out, I ran the trained model again over the whole test period and recorded every guess. For templates it knows, it misses only 2 percent of the keys: it really did learn normal behaviour. A template it never saw is always a miss, by construction.",
      "And here is the key fact: 97 percent of the alert keys belong to templates LogBERT never saw, because alert messages almost never occur in normal training windows. So LogBERT finds alerts mainly because they are new, not because it understands them.",
      "But in the test period, 80 percent of the normal messages are new as well. So 58 percent of the normal windows that contain a new template get flagged, against 0.3 percent of the others, and 99.8 percent of its false alarms contain a new template. LogBERT has become a new-message detector.",
    ],
    point: "the two bars at 100%, then the 97%.",
    asked: [
      "Counts: 28,094 of 28,915 alert keys are new to LogBERT; 360,557 normal keys are new as well.",
      "The figure comes from scripts/make_figures.py, which masks every key of the test period once and records the rank of the true key.",
    ],
    next: "What does the language model do instead?",
  });
}

// 14 ------------------------------------------- what guidance does to the LLM
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Errors and lessons" });
  titled(s, "Guidance narrows the LLM");
  badge(s, "B");
  const conds = DATA.flagged_templates.conditions;
  const names = ["B4 keyword rules", "B3 rule-based", "B2 few-shot", "B1 zero-shot"]; // bottom to top
  const labels = names.map((n) => `${n}: ${conds[n].with_alerts + conds[n].normal} flagged`);
  s.addChart(
    pres.charts.BAR,
    [
      { name: "Template contains alerts (correct)", labels, values: names.map((n) => conds[n].with_alerts) },
      { name: "Template is normal (false alarm)", labels, values: names.map((n) => conds[n].normal) },
    ],
    chartOptions({
      x: MX, y: TOP, w: 7.85, h: 5.3, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 45,
      chartColors: [HEX.accent2, HEX.accent1], showValue: false, valAxisMinVal: 0, valAxisMaxVal: 400,
      valAxisMajorUnit: 100, showLegend: true, legendPos: "b", catAxisLabelFontSize: 13,
      title: "Test templates labelled abnormal (of 472)",
    }),
  );
  const x = 8.8;
  const w = W - MX - x;
  stat(s, { x, y: TOP, w, value: `${DATA.flagged_templates.templates_with_alerts}`, size: 54, color: C.accent2,
    label: "of the 472 test templates contain alerts" });
  bullets(
    s,
    [
      "Zero-shot, the LLM calls anything error-like abnormal: 388 templates.",
      "Examples cut this to 122, rules to 37: guidance helps, as LogRules found.",
      "The keyword rules flag 9 templates, and all 9 contain alerts.",
    ],
    { x, y: TOP + 2.0, w, h: 3.2 },
    16,
  );
  notes(s, {
    time: "about 45 seconds",
    say: [
      "Of the 472 test templates, only 20 contain alerts. Zero-shot, the model calls 388 templates abnormal: anything that sounds like an error. Its idea of 'abnormal' is not the administrators' idea of an alert.",
      "Eight examples cut that to 122, and the rules to 37. The keyword rules flag only 9, and all 9 are correct. So guidance clearly helps the language model, which matches the LogRules paper. But the model still flags far more than the rules themselves do.",
    ],
    point: "the blue part of each bar: the false alarms.",
    asked: [
      "Zero-shot false alarms include '… total interrupts …' (135,092 messages), 'instruction cache parity error corrected' (33,547) and 'ciod: Received signal …' (27,510).",
    ],
    next: "Why is the model worse than its own rules?",
  });
}

// 15 ------------------------------------------- the LLM and the rules
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Errors and lessons" });
  titled(s, "The LLM follows rule wording, not keywords");
  badge(s, "B");
  stat(s, { x: MX, y: TOP + 0.1, w: 4.4, value: "20 of 420", size: 60, color: C.accent2, labelSize: 16,
    label: "rule citations where the cited rule's keyword actually occurs in the log" });
  logPanel(
    s,
    [
      { text: "Log (normal: 27,510 messages)", dim: true, mono: false, size: 13 },
      { text: "ciod: Received signal <*>, code=<*>,", bold: true },
      { text: "      errno=<*>, address=<*>", bold: true },
      { text: "The LLM (B3) answers", dim: true, mono: false, size: 13 },
      { text: '{"label": "abnormal", "rule": 8,', alert: true },
      { text: ' "reason": "contains \'terminated\'"}', alert: true },
      { text: "Rule 8 keywords: terminated · panic! - stopping execution · exited abnormally. None occurs in the log.", dim: true, mono: false, size: 13 },
    ],
    { x: 5.4, y: TOP, w: W - MX - 5.4, h: 3.35, size: 13.5 },
  );
  const w = (CW - 0.5) / 3;
  const facts = [
    ["+5 alert templates", "that no keyword matches, caught by reading the sentences: 14 more windows"],
    ["+23 false alarms", "templates covering 57,309 normal messages; 12 of them cite the broad rule 4"],
    ["0 false alarms", "for the same rules as plain keyword matching (B4)"],
  ];
  facts.forEach(([head, body], i) => textCard(s, { x: MX + i * (w + 0.25), y: 5.1, w, h: 1.65, head, body, size: 14,
    headColor: i === 2 ? C.accent1 : C.accent2 }));
  notes(s, {
    time: "about 1 minute",
    say: [
      "I checked every rule the model cited. It cites a rule in 420 answers, but in only 20 of them does that rule's keyword actually occur in the log.",
      "Here is an example. For 'ciod: Received signal', 27,510 normal messages, the model answers abnormal because of rule 8, 'contains terminated'. The word 'terminated' is not in the message.",
      "The model follows the rules' sentences, which are broad, not their verified keywords. That has one benefit: it catches 5 alert templates that no keyword matches, worth 14 windows. But it also creates 23 false-alarm templates covering 57,000 normal messages, 12 of them through one vague rule, rule 4. The keyword rules have zero false alarms.",
    ],
    point: "the message, then the word 'terminated' in the answer.",
    asked: [
      "Only 12 of the 420 citations name the rule the keyword matcher would actually apply.",
      "The model never labels normal a template the keyword rules call abnormal; it only adds abnormal labels.",
      "Rule 1, 'Describes normal system operations or status updates', serves as its catch-all for 349 normal answers.",
    ],
    next: "Two more traps.",
  });
}

// 16 ------------------------------------------- rule ranking and label drift
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Errors and lessons" });
  titled(s, "Two traps: rule ranking and label drift");
  const sens = DATA.rule_sensitivity;
  const ks = sens.k.map((k) => String(k));
  s.addChart(
    pres.charts.LINE,
    [
      { name: "Validation F1", labels: ks, values: sens.val_f1 },
      { name: "Test F1", labels: ks, values: sens.test_f1 },
    ],
    chartOptions({
      x: MX, y: TOP, w: 6.4, h: 4.3, chartColors: [HEX.accent1, HEX.accent2], lineSize: 2,
      lineDataSymbol: "circle", lineDataSymbolSize: 7, valAxisMinVal: 0.6, valAxisMaxVal: 1.0,
      valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", showLegend: true, legendPos: "b",
      title: "Keyword rules: F1 by number of rules used (k)",
    }),
  );
  s.addText(
    "k = 30, fixed in the proposal: test F1 0.956. k = 45, best on validation: 0.980. Near-duplicates fill the top 30, and the 'socket closed' rule sits at rank 32.",
    { x: MX, y: 5.95, w: 6.4, h: 0.85, fontSize: 13, color: C.accent6, margin: 0, valign: "top", isTextBox: true },
  );
  const x = 7.35;
  const w = W - MX - x;
  s.addText("Label drift: one template, two labels", { x, y: TOP, w, h: 0.4, fontSize: 18, bold: true,
    color: C.text2, margin: 0, isTextBox: true });
  s.addText("Training: normal, 472 messages", { x, y: TOP + 0.55, w, h: 0.32, fontSize: 14, bold: true,
    color: C.accent1, margin: 0, isTextBox: true });
  logPanel(s, [{ text: "ciod: LOGIN chdir(…) failed:" }, { text: "      Permission denied" }],
    { x, y: TOP + 0.9, w, h: 0.95, size: 13.5 });
  s.addText("Test: alert, 816 messages", { x, y: TOP + 2.0, w, h: 0.32, fontSize: 14, bold: true,
    color: C.accent2, margin: 0, isTextBox: true });
  logPanel(s, [{ text: "ciod: LOGIN chdir(…) failed:", alert: true }, { text: "      Input/output error", alert: true }],
    { x, y: TOP + 2.35, w, h: 0.95, size: 13.5 });
  textCard(s, { x, y: TOP + 3.5, w, h: 1.75, size: 14, head: "Drain: ciod: LOGIN <*> failed: <*> <*>",
    body: "The words that decide the label were masked, so every method that learned from training calls it normal." });
  notes(s, {
    time: "about 45 seconds",
    say: [
      "First, rule ranking. Ranking by coverage fills the top 30 with near-duplicates, and the rule for 'socket closed' lands at rank 32, just outside. That template causes most of the keyword rules' misses. Choosing the number of rules on validation would give 45 rules and a test F1 of 0.98. I report 30 because the proposal fixed it in advance.",
      "Second, label drift. In training, 'LOGIN chdir failed: Permission denied' is normal. In the test period, 'LOGIN chdir failed: Input/output error' is an alert, 816 messages. Drain masked exactly the words that decide the label, so both become the same template, and every method that learned from the training data calls it normal.",
    ],
    point: "the jump of both lines at k = 35, then the two LOGIN lines.",
    asked: [
      "The LOGIN template accounts for 816 of the 913 alert messages the keyword rules miss, but only 2 missed windows, because those windows contain other alerts that are caught.",
      "A better rule selection would reward the new templates a rule covers, not raw usage.",
    ],
    next: "What else went wrong during the project?",
  });
}

// 17 ------------------------------------------- difficulties
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Errors and lessons" });
  titled(s, "What went wrong, and how we fixed it");
  const items = [
    ["Reference code tunes on test", "We tune g and r on validation. Tuning on test would report F1 0.475 instead of 0.293."],
    ["Shortcuts in the original pipeline", "It shuffles windows, skips short test windows and truncates long ones. We split by time, score every window and chunk long ones."],
    ["No GPU, no API key", "Qwen in 4-bit through llama.cpp on 4 CPU cores. Ending every prompt with the log line made calls 20% faster."],
    ["The machine restarted mid-run", "Every LLM answer is cached, so the rerun reused 689 stored answers and computed only the rest."],
    ["Dataset host unreachable", "We verified a mirror: checksum, 4,747,963 lines, 348,460 alerts, all 2,000 sample lines present."],
    ["Small samples mislead", "On the 2,000-line sample LogBERT flags almost everything. Only the full log gives meaningful numbers."],
  ];
  const w = (CW - 0.5) / 3;
  const h = 2.5;
  items.forEach(([head, body], i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    textCard(s, { x: MX + col * (w + 0.25), y: TOP + row * (h + 0.25), w, h, head, body, size: 16 });
  });
  notes(s, {
    time: "about 1 minute",
    say: [
      "Several things went wrong, and fixing them was a large part of the work. First, reading the reference code critically: the LogBERT repository tunes its threshold on the test set, shuffles windows before splitting, skips short test windows and truncates long ones. Each of these would have inflated our numbers, so I changed them and documented why.",
      "Second, resources: there was no GPU and no OpenAI key. I ran the language model in 4-bit on four CPU cores, and putting the log line at the end of every prompt let the server reuse its work, which made each call 20 percent faster.",
      "Third, the cloud machine restarted while I was away, once in the middle of the language-model run. Because every answer is cached, the rerun reused 689 stored answers and only computed the rest.",
      "And the dataset's official host was unreachable, so I verified a mirror with a checksum and line counts.",
    ],
    asked: [
      "Engineering bugs the tests caught: NumPy numbers in the training history made checkpoints unloadable with safe loading; a pandas category column broke fillna; the demo's table rejected string labels in integer columns.",
      "Windowing: LogBERT's sliding-window loop has edge cases; our vectorised version is tested against a port of the original on random timelines.",
    ],
    next: "What I take away from this.",
  });
}

// 18 ------------------------------------------- lessons
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Errors and lessons" });
  titled(s, "Lessons and next steps");
  const w = (CW - 0.3) / 2;
  card(s, MX, TOP, w, 5.25);
  s.addText("Lessons", { x: MX + 0.3, y: TOP + 0.2, w: w - 0.6, h: 0.45, fontSize: 20, bold: true, color: C.text2,
    margin: 0, isTextBox: true });
  s.addText(
    [
      "The evaluation setup can change the answer more than the model.",
      "Check what a model relies on: LogBERT turned out to detect newness.",
      "Check that an LLM does what it says it does: 20 of 420 citations held.",
      "Always run the simple baseline: here it won.",
    ].map((t, i, all) => ({ text: t, options: { bullet: { type: "number" }, breakLine: i < all.length - 1 } })),
    { x: MX + 0.3, y: TOP + 0.85, w: w - 0.6, h: 4.2, fontSize: 19, color: C.text1, valign: "top", margin: 0,
      paraSpaceAfter: 18, isTextBox: true },
  );
  card(s, MX + w + 0.3, TOP, w, 5.25, C.text2);
  s.addText("Next steps", { x: MX + w + 0.6, y: TOP + 0.2, w: w - 0.6, h: 0.45, fontSize: 20, bold: true,
    color: C.background1, margin: 0, isTextBox: true });
  s.addText(
    [
      "Select rules for the new templates they cover, not raw usage.",
      "Check rule sentences too, or show the LLM only the keywords.",
      "Train LogBERT longer on a GPU, and try other Loghub datasets (Thunderbird, Spirit).",
      "Combine A and B3: requiring both raises B3's F1 from 0.644 to 0.776.",
    ].map((t, i, all) => ({ text: t, options: { bullet: true, breakLine: i < all.length - 1 } })),
    { x: MX + w + 0.6, y: TOP + 0.85, w: w - 0.6, h: 4.2, fontSize: 19, color: C.background1, valign: "top",
      margin: 0, paraSpaceAfter: 18, isTextBox: true },
  );
  notes(s, {
    time: "about 30 seconds",
    say: [
      "Four lessons. The evaluation setup can change the answer more than the choice of model. Always check what a model relies on: LogBERT turned out to detect newness. Check that a language model actually does what it says it does. And always run the simple baseline: here it won.",
      "Next steps would be selecting rules for what they add, checking rule sentences and not only keywords, training LogBERT longer on a GPU, and testing on other log datasets.",
    ],
    asked: [
      "Combining LogBERT with the keyword rules does not help: OR keeps LogBERT's false alarms (F1 0.301), AND scores 0.939, below the rules alone. Only B3 gains from requiring LogBERT to agree.",
    ],
    next: "Now let me show you the system live.",
  });
}

// 19 ---------------------------------------------------------- demo
pres.addSection({ title: "Live demo" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Live demo" });
  titled(s, "Live demo");
  logPanel(s, [{ text: "python app/demo.py --config configs/bgl_cpu.yaml --share" }],
    { x: MX, y: TOP - 0.05, w: CW, h: 0.6, size: 15, fill: C.background1, transparency: 88 });
  const steps = [
    ["KERNDTLB 1118536327 2005.06.11 R30-M0-N9-C:J16-U01 … FATAL data TLB error interrupt",
      "The tag is removed first. Keyword rule 24 and the LLM flag it; LogBERT misses the key, but needs more than 3 misses."],
    ["instruction cache parity error corrected",
      "Says 'error' but is normal: the rules leave it alone. Zero-shot, the LLM called it abnormal."],
    ["ciod: LOGIN chdir(/p/gb1/stella/RAPTOR/2183) failed: Input/output error",
      "An alert hidden by label drift: it maps to a template that was normal in training."],
    ["Tab 2: insert  rts panic! - stopping execution  into a normal window",
      "Move it around and watch LogBERT's missed-key count and the 'without your line' comparison."],
  ];
  steps.forEach(([line, what], i) => {
    const y = TOP + 0.85 + i * 1.12;
    s.addText(String(i + 1), { shape: pres.shapes.OVAL, x: MX, y: y + 0.22, w: 0.55, h: 0.55,
      fill: { color: C.accent2 }, color: C.background1, bold: true, fontSize: 20, align: "center",
      valign: "middle", margin: 0, isTextBox: true });
    s.addText(
      [
        { text: line, options: { fontFace: MONO, fontSize: 14, bold: true, color: C.background1, breakLine: true } },
        { text: what, options: { fontSize: 15, color: C.background2 } },
      ],
      { x: MX + 0.85, y, w: CW - 0.85, h: 1.0, valign: "middle", margin: 0, paraSpaceAfter: 4, isTextBox: true },
    );
  });
  notes(s, {
    time: "about 10 minutes (the course's demo block)",
    say: [
      "BEFORE THE TALK: on the presenting machine run 'make data preprocess', then 'make restore OUT=outputs/bgl_cpu', then start 'python app/demo.py --config configs/bgl_cpu.yaml --share' and keep the page open. The share link lets the audience try it on their phones.",
      "Step 1. Paste the full KERNDTLB line. Show that the alert tag is removed first and shown as the ground truth. The keyword rules flag the line with rule 24, and so does the rule-based language model. LogBERT misses this key, because it never saw it in a normal window, but one miss is not enough to flag the window: it needs more than three.",
      "Step 2. Paste 'instruction cache parity error corrected'. It contains the word error but is normal. The keyword rules and the rule-based model leave it alone. Mention that zero-shot, the model called it abnormal.",
      "Step 3. Paste the LOGIN Input/output error line. It is a real alert, but it maps to the template that was normal in training, so the guided methods miss it: label drift, live.",
      "Step 4. Open the second tab, choose a normal window and insert 'rts panic! - stopping execution' at different positions. Watch how LogBERT's verdict and the 'without your line' count change.",
      "Step 5, if time allows. Ask the audience for a made-up line, for example 'fan module 3 failed on node card R12-M0'. Nobody has seen this template: LogBERT reads it as unknown, and the keyword rules match nothing. Explain what each method sees.",
      "IF SOMETHING FAILS: without the model server, the demo still runs on the saved decisions (add --no-live-llm); for new lines, only the keyword rules and LogBERT answer.",
    ],
    next: "Back to the slides for the conclusion.",
  });
}

// 20 -------------------------------------------------------- conclusion
pres.addSection({ title: "Conclusion" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
  titled(s, "Conclusion");
  const w = (CW - 0.5) / 3;
  const points = [
    ["0.956", C.accent2, "F1 of the verified rules applied as keywords, against 0.293 for LogBERT, under one fair, chronological setup."],
    ["79.6%", C.accent1, "of test messages are new. LogBERT, reading only IDs, flags newness; reading the words tells a harmless new message from an alert."],
    ["20 of 420", C.accent2, "The verified rules do the work: the LLM applies them loosely, and keyword matching is better and 400,000× faster."],
  ];
  points.forEach(([value, color, body], i) => {
    const x = MX + i * (w + 0.25);
    card(s, x, TOP, w, 3.3, C.background1, "conclusion card");
    stat(s, { x: x + 0.3, y: TOP + 0.3, w: w - 0.6, value, color, size: 44, labelSize: 17,
      labelH: 2.0, label: body });
  });
  s.addText(
    "Limits: one dataset, one chronological split, one run per method on a CPU budget, and one 4-bit 7B model that both wrote and applied the rules. A stronger rule generator, verified rule sentences, or a dataset whose failures are less tied to particular words could change the balance.",
    { x: MX, y: 5.15, w: CW, h: 1.2, fontSize: 16, color: C.background2, margin: 0, valign: "top", isTextBox: true },
  );
  notes(s, {
    time: "about 45 seconds",
    say: [
      "To answer the research question: under one fair, chronological setup, the rule-guided approach clearly beats LogBERT: 0.64 for the rule-based model and 0.96 for the same rules as keywords, against 0.29.",
      "The main reason is that the log changes over time. LogBERT sees only IDs, so it flags anything new; the rule-guided approach reads the words and can tell a harmless new message from an alert.",
      "But the verified rules, not the language model, do the work: the model applies them loosely, and plain keyword matching is better and about 400,000 times faster.",
      "These results come from one dataset and one run under a CPU budget, so a stronger rule generator or a dataset whose failures are less tied to specific words could change the balance.",
    ],
    next: "Thank you. I'm happy to take questions.",
  });
}

// 21 ---------------------------------------------------------- questions
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Conclusion" });
  titled(s, "Thank you. Questions?");
  s.addText("Code, results and a step-by-step guide to every part of the project", { placeholder: "body" });
  s.addText(
    [
      { text: "github.com/Amshu-0/claude-nlp", options: { breakLine: true } },
      { text: "docs/guide.md  ·  docs/report.md  ·  results/" },
    ],
    { placeholder: "course" },
  );
  card(s, 7.45, 2.1, 5.28, 3.3, C.background1, "recap");
  s.addText(
    [
      { text: "In one sentence", options: { bold: true, fontSize: 16, color: C.accent2, breakLine: true } },
      { text: "Reading the words beats reading only the IDs when the log changes, and verified keyword rules beat the LLM that wrote them.",
        options: { fontSize: 20, color: C.text1 } },
    ],
    { x: 7.75, y: 2.3, w: 4.7, h: 2.9, valign: "middle", margin: 0, paraSpaceAfter: 10, isTextBox: true },
  );
  notes(s, {
    time: "the remaining time, about 4 minutes",
    say: ["Thank you for listening. I'm happy to take questions."],
    asked: [
      "Why not compare the papers' numbers? Different data samples and evaluation units: DeepLog scores F1 86 on BGL in one paper and 16.8 in the other. Only a shared setup answers the question.",
      "Is it unfair that Approach B uses labels? That is the trade-off studied: LogBERT is semi-supervised by design, and the zero-shot condition B1, without labels, is the weakest method.",
      "Why is LogBERT worse than in its paper? Its code shuffles windows into training and test and picks the threshold on test; a chronological split with validation tuning is much harder. Tuning on test would give 0.475 here.",
      "Why classify templates, not messages? 472 LLM calls instead of 482,331; the best possible template-level F1 is 0.996, so little is lost.",
      "Did you reproduce LogRules fully? Induction and deduction, not the CPO fine-tuning stage; and one 4-bit 7B model wrote and applied the rules (LogRules used GPT-4o-mini to write them).",
      "Is k = 30 cherry-picked? It was fixed in the proposal before the run; validation would pick 45, which gives 0.980.",
      "How sensitive is LogBERT to g and r? Very: only 16 of 231 settings come within 0.02 of the best validation F1.",
      "Could the approaches be combined? They miss different windows, but OR keeps LogBERT's false alarms (0.301) and AND gives 0.939, below the rules alone.",
    ],
  });
}

// ------------------------------------------------------------ write + theme

// Write THEME's colours and name into the deck's theme part. pptxgenjs writes
// the two theme fonts but Office's default colours, so scheme colours would
// otherwise resolve to Office blue and orange.
async function applyTheme(file, theme) {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const part = "ppt/theme/theme1.xml";
  const slots = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
  const scheme = `<a:clrScheme name="${theme.name}">${slots
    .map((slot) => `<a:${slot}><a:srgbClr val="${theme.colors[slot]}"/></a:${slot}>`)
    .join("")}</a:clrScheme>`;
  const xml = (await zip.file(part).async("string"))
    .replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, () => scheme)
    .replace(/(<a:(?:theme|fontScheme)\b[^>]*?\bname=")[^"]*"/g, (_, head) => `${head}${theme.name}"`);
  if (!xml.includes(scheme)) throw new Error(`${part} has no colour scheme to replace`);
  zip.file(part, xml);
  // A scheme colour passed to a hex-only option ends up as an invalid colour value.
  for (const name of Object.keys(zip.files).filter((n) => n.endsWith(".xml"))) {
    const bad = (await zip.file(name).async("string")).match(/<a:srgbClr val="((?![0-9A-Fa-f]{6}")[^"]*)"/);
    if (bad) throw new Error(`${name}: invalid colour "${bad[1]}"; pass a six-digit hex value there`);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

pres
  .writeFile({ fileName: OUTPUT })
  .then(() => applyTheme(OUTPUT, THEME))
  .then(() => writeScript(path.join(HERE, "speaker_notes.md")))
  .then(() => console.log(`wrote ${path.relative(ROOT, OUTPUT)} and docs/slides/speaker_notes.md`))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
