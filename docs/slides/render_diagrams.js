// Render the slide diagrams in diagrams.md to transparent PNGs in diagrams/.
//
//   npm install                   # mermaid and pptxgenjs, pinned in package.json
//   npx playwright install chromium   # once, if Playwright has no browser yet
//   node render_diagrams.js
//
// Every ```mermaid block preceded by a "<!-- diagram: name -->" comment becomes
// diagrams/<name>.png. Mermaid runs in a headless Chromium (Playwright), with a
// theme that matches the deck: Calibri-like text, light fills, grey lines.
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const HERE = __dirname;
const OUT = path.join(HERE, "diagrams");

// Mermaid settings shared by every diagram. "Carlito" is metrically identical
// to Calibri, the deck's body font, and is used where Calibri is not installed.
const MERMAID_CONFIG = {
  startOnLoad: false,
  theme: "base",
  securityLevel: "strict",
  flowchart: { htmlLabels: true, curve: "basis", padding: 16, nodeSpacing: 40, rankSpacing: 46 },
  gantt: { barHeight: 26, barGap: 8, fontSize: 16, sectionFontSize: 16, leftPadding: 110 },
  themeVariables: {
    fontFamily: "Calibri, Carlito, sans-serif",
    fontSize: "18px",
    primaryColor: "#e7f1fb",
    primaryBorderColor: "#2a78d6",
    primaryTextColor: "#0b0b0b",
    lineColor: "#52514e",
    textColor: "#0b0b0b",
    edgeLabelBackground: "#ffffff",
    // Gantt: windows in light blue, the alert in the deck's orange.
    taskBkgColor: "#e7f1fb",
    taskBorderColor: "#2a78d6",
    taskTextColor: "#0b0b0b",
    taskTextDarkColor: "#0b0b0b",
    taskTextOutsideColor: "#0b0b0b",
    critBkgColor: "#eb6834",
    critBorderColor: "#eb6834",
    sectionBkgColor: "#ffffff",
    sectionBkgColor2: "#ffffff",
    altSectionBkgColor: "#ffffff",
    gridColor: "#e1e0d9",
    todayLineColor: "transparent",
  },
};

// The named diagrams of a Markdown file, as [{ name, code }].
function namedDiagrams(markdown) {
  const pattern = /<!-- diagram: ([\w-]+) -->\s*\n```mermaid\n([\s\S]*?)```/g;
  return [...markdown.matchAll(pattern)].map((match) => ({ name: match[1], code: match[2] }));
}

// Render each diagram in the browser and screenshot it at three times its size,
// so it stays sharp when the slide is projected.
async function main() {
  const diagrams = namedDiagrams(fs.readFileSync(path.join(HERE, "diagrams.md"), "utf8"));
  fs.mkdirSync(OUT, { recursive: true });

  const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: 3 });
  await page.setContent('<html><body style="margin:0;background:transparent"><div id="c"></div></body></html>');
  await page.addScriptTag({ path: require.resolve("mermaid/dist/mermaid.min.js") });
  await page.evaluate((config) => mermaid.initialize(config), MERMAID_CONFIG);

  for (const { name, code } of diagrams) {
    const svg = await page.evaluate(async ([id, source]) => (await mermaid.render(id, source)).svg, [`d-${name}`, code]);
    await page.evaluate((markup) => { document.getElementById("c").innerHTML = markup; }, svg);
    const element = await page.$("#c svg");
    await element.screenshot({ path: path.join(OUT, `${name}.png`), omitBackground: true });
    console.log(`wrote diagrams/${name}.png`);
  }
  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
