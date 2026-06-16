// Builds the JCIM manuscript .docx from manuscript.json using the docx library.
const fs = require("fs");
const path = require("path");
const ROOT = path.join(__dirname, "..");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        ImageRun, AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
        PageBreak, VerticalAlign, LineNumberRestartFormat } = require("docx");

const D = JSON.parse(fs.readFileSync(path.join(ROOT, "manuscript.json"), "utf8"));
const CONTENT = 9360;

const R = (runs, base = {}) => runs.map(r => new TextRun({
  text: r.t, bold: r.b || base.bold || false, italics: r.i || base.italics || false,
  superScript: r.sup || false, subScript: r.sub || false, font: "Times New Roman", size: base.size || 24,
}));

const para = (runs, opts = {}) => new Paragraph({
  children: R(runs, opts.run || {}),
  alignment: opts.align,
  spacing: opts.spacing || { line: 480, lineRule: "auto", after: 0 },
  ...(opts.extra || {}),
});

const center = AlignmentType.CENTER, just = AlignmentType.JUSTIFIED;
const children = [];

children.push(new Paragraph({ alignment: center, spacing: { before: 240, after: 240 },
  children: [new TextRun({ text: D.title, bold: true, font: "Times New Roman", size: 32 })] }));
children.push(new Paragraph({ alignment: center, spacing: { after: 120 },
  children: [new TextRun({ text: D.authors, font: "Times New Roman", size: 24 })] }));
children.push(new Paragraph({ alignment: center, spacing: { after: 60 },
  children: [new TextRun({ text: D.affiliation, italics: true, font: "Times New Roman", size: 22 })] }));
children.push(new Paragraph({ alignment: center, spacing: { after: 240 },
  children: [new TextRun({ text: D.corresponding, font: "Times New Roman", size: 22 })] }));

children.push(new Paragraph({ spacing: { before: 120, after: 120 },
  children: [new TextRun({ text: "ABSTRACT", bold: true, font: "Times New Roman", size: 24 })] }));
children.push(para([{ t: D.abstract }], { align: just }));
children.push(new Paragraph({ spacing: { before: 160, after: 120 },
  children: [new TextRun({ text: "KEYWORDS", bold: true, font: "Times New Roman", size: 24 })] }));
children.push(new Paragraph({ spacing: { after: 0 }, alignment: just,
  children: [new TextRun({ text: D.keywords, font: "Times New Roman", size: 24 })] }));
children.push(new Paragraph({ children: [new PageBreak()] }));

const border = { style: BorderStyle.SINGLE, size: 2, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

for (const b of D.blocks) {
  if (b.type === "h1") {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: R(b.runs) }));
  } else if (b.type === "h2") {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: R(b.runs) }));
  } else if (b.type === "p") {
    children.push(para(b.runs, { align: just }));
  } else if (b.type === "equation") {
    children.push(new Paragraph({ alignment: center,
      spacing: { before: 120, after: 120, line: 240, lineRule: "auto" },
      children: R(b.runs) }));
  } else if (b.type === "caption") {
    children.push(new Paragraph({ alignment: just, spacing: { before: 80, after: 240 },
      children: b.runs.map(r => new TextRun({ text: r.t, bold: r.b, italics: r.i,
        superScript: r.sup, subScript: r.sub, font: "Times New Roman", size: 20 })) }));
  } else if (b.type === "figure") {
    children.push(new Paragraph({ alignment: center, spacing: { before: 240, after: 40 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(b.path),
        transformation: { width: b.w, height: b.h },
        altText: { title: "Figure", description: "Figure", name: "Figure" } })] }));
  } else if (b.type === "table") {
    const ncol = b.header.length;
    const cw = Math.floor(CONTENT / ncol);
    const widths = Array(ncol).fill(cw); widths[ncol - 1] = CONTENT - cw * (ncol - 1);
    const mkCell = (runs, i, head) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      shading: head ? { fill: "D9E2F3", type: ShadingType.CLEAR } : undefined,
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        spacing: { line: 240, after: 0 },
        children: runs.map(r => new TextRun({ text: r.t, bold: head || r.b, italics: r.i,
          superScript: r.sup, font: "Times New Roman", size: 22 })) })],
    });
    const rows = [new TableRow({ tableHeader: true,
      children: b.header.map((c, i) => mkCell(c, i, true)) })];
    for (const row of b.rows) rows.push(new TableRow({ children: row.map((c, i) => mkCell(c, i, false)) }));
    children.push(new Table({ width: { size: CONTENT, type: WidthType.DXA }, columnWidths: widths, rows }));
    children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  }
}

children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("References")] }));
D.references.forEach((ref, idx) => {
  children.push(new Paragraph({
    alignment: just, spacing: { line: 276, lineRule: "auto", after: 80 },
    indent: { left: 360, hanging: 360 },
    children: [new TextRun({ text: (idx + 1) + ". ", font: "Times New Roman", size: 22 }),
      ...ref.map(r => new TextRun({ text: r.t, italics: r.i, font: "Times New Roman", size: 22 }))],
  }));
});

children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(new Paragraph({ alignment: center, spacing: { before: 240, after: 240 },
  children: [new TextRun({ text: "For Table of Contents Use Only", bold: true, font: "Times New Roman", size: 24 })] }));
children.push(new Paragraph({ alignment: center, spacing: { after: 120 },
  children: [new TextRun({ text: D.title, bold: true, font: "Times New Roman", size: 22 })] }));
children.push(new Paragraph({ alignment: center, spacing: { after: 240 },
  children: [new TextRun({ text: D.authors, font: "Times New Roman", size: 22 })] }));
children.push(new Paragraph({ alignment: center,
  children: [new ImageRun({ type: "png", data: fs.readFileSync(D.toc_graphic),
    transformation: { width: 312, height: 168 },
    altText: { title: "TOC", description: "Table of contents graphic", name: "TOC" } })] }));

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman", color: "000000" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, italics: true, font: "Times New Roman", color: "000000" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
      lineNumbers: { countBy: 1, start: 1, restart: LineNumberRestartFormat.CONTINUOUS } },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(ROOT, "paper", "manuscript_JCIM.docx"), buf);
  console.log("manuscript_JCIM.docx written,", buf.length, "bytes");
});
