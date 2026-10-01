"""
generate_dataset_pdf.py
-----------------------
Creates a professional PDF document that describes the dataset
(data/text_classification_dataset.csv): purpose, creation method, categories,
statistics, sample rows, labelling rules and limitations.

Usage:
    python src/generate_dataset_pdf.py
"""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "text_classification_dataset.csv"
PDF_PATH = ROOT / "docs" / "Dataset_Documentation.pdf"

NAVY = colors.HexColor("#1F3A5F")
TEAL = colors.HexColor("#1B7F79")
LIGHT = colors.HexColor("#EAF2F8")
GREY = colors.HexColor("#555555")

CATEGORY_INFO = [
    ("Complaint", "The user reports a problem or expresses dissatisfaction and expects a fix, refund or action.",
     "The product stopped working after two days. This is a waste of money."),
    ("Inquiry", "The user asks a question or requests information (price, timing, steps, availability).",
     "Do you offer a student discount on the annual plan?"),
    ("Feedback", "The user shares an opinion, praise or a suggestion without demanding urgent action.",
     "It would be nice to have a dark mode option in the app."),
    ("Other", "Greetings, announcements, acknowledgements, spam and anything outside the three classes above.",
     "Good morning everyone."),
]


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=26, leading=30,
                                textColor=NAVY, alignment=0, spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=12, leading=16,
                                   textColor=GREY, spaceAfter=4),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=15, leading=19,
                             textColor=NAVY, spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=10.5, leading=15, spaceAfter=6),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=9, leading=12),
        "cellb": ParagraphStyle("cellb", parent=base["Normal"], fontSize=9, leading=12,
                                textColor=colors.white, fontName="Helvetica-Bold"),
        "center": ParagraphStyle("center", parent=base["Normal"], fontSize=9, alignment=TA_CENTER, textColor=GREY),
    }


def make_table(rows, col_widths, styles):
    """rows[0] is the header; all cells are wrapped in Paragraphs so long text wraps."""
    data = [[Paragraph(str(c), styles["cellb"]) for c in rows[0]]]
    for row in rows[1:]:
        data.append([Paragraph(str(c), styles["cell"]) for c in row])
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C4CE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.2 * cm, "Varolline Internship Program | Smart Text Classifier | Dataset Documentation")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    df = pd.read_csv(CSV_PATH)
    df["words"] = df["text"].str.split().str.len()
    df["chars"] = df["text"].str.len()
    styles = build_styles()
    story = []

    # ---------------- Title block ----------------
    story.append(Paragraph("VAROLLINE INTERNSHIP PROGRAM | AI ENGINEER TASK", ParagraphStyle(
        "eyebrow", parent=styles["subtitle"], textColor=TEAL, fontName="Helvetica-Bold", fontSize=9)))
    story.append(Paragraph("Smart Text Classifier<br/>Dataset Documentation", styles["title"]))
    story.append(Paragraph("Labelled short-text dataset for Complaint, Inquiry, Feedback and Other classification",
                           styles["subtitle"]))
    story.append(Paragraph("Prepared by: Muhammad Faizan Farooq | October 2026 | Version 1.0", styles["subtitle"]))
    story.append(Spacer(1, 10))

    # ---------------- 1. Overview ----------------
    story.append(Paragraph("1. Dataset Overview", styles["h1"]))
    story.append(Paragraph(
        f"This dataset contains <b>{len(df):,} short English text messages</b>, each labelled with one of four "
        "categories. It is used to train and evaluate a Logistic Regression classifier (TF-IDF features) that "
        "predicts the category of a customer-style message. The file is stored as "
        "<font name='Courier'>data/text_classification_dataset.csv</font> with three columns.", styles["body"]))
    story.append(make_table([
        ["Column", "Type", "Description"],
        ["id", "Integer", "Unique row number (1 to N)."],
        ["text", "String", "The short message written by a user."],
        ["category", "String", "Target label: Complaint, Inquiry, Feedback or Other."],
    ], [3 * cm, 3 * cm, 11 * cm], styles))

    # ---------------- 2. Source and creation ----------------
    story.append(Paragraph("2. Data Source and Creation Method", styles["h1"]))
    story.append(Paragraph(
        "No public dataset was copied. The dataset was <b>created specifically for this task</b> with a reproducible "
        "Python script (<font name='Courier'>src/generate_dataset.py</font>, random seed 42). The process was:",
        styles["body"]))
    for line in [
        "<b>1. Hand-written samples:</b> 12 realistic messages per category, including borderline sentences "
        "(for example an unhappy message that is really a question).",
        "<b>2. Template generation:</b> many additional messages produced from sentence templates combined with "
        "varied slot values such as products, cities, issues, features, days and times.",
        "<b>3. Noise injection:</b> about 15% of generated samples were lower-cased or had their final punctuation "
        "removed, to imitate messy real-world input.",
        "<b>4. Cleaning and balancing:</b> duplicates were removed, each class was balanced to the same size, "
        "and rows were shuffled.",
    ]:
        story.append(Paragraph(line, styles["body"]))

    # ---------------- 3. Categories ----------------
    story.append(KeepTogether([
        Paragraph("3. Category Definitions", styles["h1"]),
        make_table(
            [["Category", "Definition", "Example"]] + [[a, b, c] for a, b, c in CATEGORY_INFO],
            [2.6 * cm, 7.4 * cm, 7 * cm], styles),
    ]))

    # ---------------- 4. Statistics ----------------
    story.append(Paragraph("4. Dataset Statistics", styles["h1"]))
    counts = df["category"].value_counts()
    rows = [["Category", "Samples", "Share", "Avg. words", "Min words", "Max words"]]
    for cat in ["Complaint", "Inquiry", "Feedback", "Other"]:
        sub = df[df["category"] == cat]
        rows.append([cat, counts[cat], f"{counts[cat] / len(df):.0%}", f"{sub['words'].mean():.1f}",
                     sub["words"].min(), sub["words"].max()])
    rows.append(["<b>Total</b>", f"<b>{len(df)}</b>", "<b>100%</b>", f"<b>{df['words'].mean():.1f}</b>",
                 f"<b>{df['words'].min()}</b>", f"<b>{df['words'].max()}</b>"])
    story.append(make_table(rows, [3.2 * cm, 2.6 * cm, 2.4 * cm, 3 * cm, 2.9 * cm, 2.9 * cm], styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The classes are perfectly balanced, so accuracy is a fair metric alongside macro F1-score. "
        f"There are <b>{int(df['text'].isna().sum())}</b> missing values and "
        f"<b>{int(df['text'].duplicated().sum())}</b> duplicate texts.", styles["body"]))

    # ---------------- 5. Sample rows ----------------
    story.append(Paragraph("5. Sample Rows", styles["h1"]))
    samples = df.groupby("category", group_keys=False).apply(lambda g: g.head(4), include_groups=False)
    samples = df.loc[samples.index]
    rows = [["ID", "Text", "Category"]]
    for _, r in samples.iterrows():
        rows.append([r["id"], r["text"], r["category"]])
    story.append(make_table(rows, [1.4 * cm, 12.2 * cm, 3.4 * cm], styles))

    # ---------------- 6. Labelling rules ----------------
    story.append(Paragraph("6. Labelling Guidelines", styles["h1"]))
    for line in [
        "A message asking for something to be fixed, refunded or escalated is a <b>Complaint</b>, even if polite.",
        "A message whose main purpose is to ask for information is an <b>Inquiry</b>, even if the tone is negative.",
        "Praise, opinions and improvement ideas that do not demand urgent action are <b>Feedback</b>.",
        "Greetings, announcements, acknowledgements, spam and random text are <b>Other</b>.",
    ]:
        story.append(Paragraph("&bull; " + line, styles["body"]))

    # ---------------- 7. Limitations ----------------
    story.append(Paragraph("7. Limitations of the Dataset", styles["h1"]))
    for line in [
        "Most samples are template-based, so vocabulary and sentence patterns are less diverse than real user text.",
        "Only English is covered; Roman Urdu or mixed-language messages are not included.",
        "Each message has exactly one label; real messages can contain several intents at once.",
        "High scores on this dataset may be optimistic; the model should be re-tested on genuine user messages.",
    ]:
        story.append(Paragraph("&bull; " + line, styles["body"]))

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Smart Text Classifier - Dataset Documentation",
                            author="Muhammad Faizan Farooq")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    main()
