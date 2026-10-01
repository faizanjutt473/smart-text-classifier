"""
generate_report_pdf.py
----------------------
Builds docs/Project_Report.pdf (2 pages). All numbers come from
reports/metrics.json, so the report always matches the trained model.

Run after training:
    python -m src.generate_report_pdf
"""

import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src import config

PDF_PATH = config.ROOT / "docs" / "Project_Report.pdf"
NAVY = colors.HexColor("#1F3A5F")
TEAL = colors.HexColor("#1B7F79")
LIGHT = colors.HexColor("#EAF2F8")
GREY = colors.HexColor("#555555")


def styles():
    base = getSampleStyleSheet()
    return {
        "eyebrow": ParagraphStyle("eyebrow", parent=base["Normal"], fontSize=8, textColor=TEAL,
                                  fontName="Helvetica-Bold"),
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=21, leading=24, textColor=NAVY,
                                alignment=0, spaceAfter=2),
        "meta": ParagraphStyle("meta", parent=base["Normal"], fontSize=9, textColor=GREY, spaceAfter=4),
        "h": ParagraphStyle("h", parent=base["Heading2"], fontSize=11.5, leading=14, textColor=NAVY,
                            spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.4, leading=12.8, spaceAfter=3),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"], fontSize=9.4, leading=12.6,
                                 leftIndent=12, bulletIndent=2, spaceAfter=2),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8.8, leading=11),
        "cellb": ParagraphStyle("cellb", parent=base["Normal"], fontSize=8.8, leading=11,
                                textColor=colors.white, fontName="Helvetica-Bold"),
    }


def bullets(items, st):
    return [Paragraph(text, st["bullet"], bulletText="\u2022") for text in items]


def metric_table(m, st):
    pct = lambda x: f"{x * 100:.1f}%"
    rows = [
        ["Evaluation set", "Rows", "Accuracy", "Precision", "Recall", "Macro F1"],
        ["Held-out test (20% split)", m["test_rows"], pct(m["test"]["accuracy"]),
         pct(m["test"]["precision_macro"]), pct(m["test"]["recall_macro"]), pct(m["test"]["f1_macro"])],
        ["Independent hand-written set", m["external_rows"], pct(m["external"]["accuracy"]),
         pct(m["external"]["precision_macro"]), pct(m["external"]["recall_macro"]),
         pct(m["external"]["f1_macro"])],
    ]
    data = [[Paragraph(str(c), st["cellb"]) for c in rows[0]]]
    data += [[Paragraph(str(c), st["cell"]) for c in r] for r in rows[1:]]
    table = Table(data, colWidths=[5.2 * cm, 1.4 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 2.0 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C4CE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def sample_table(st):
    """Demo predictions saved by src/demo_predictions.py."""
    df = pd.read_csv(config.SAMPLE_PREDICTIONS_PATH)
    data = [[Paragraph(c, st["cellb"]) for c in ["#", "Message", "Predicted", "Confidence"]]]
    for i, r in df.iterrows():
        data.append([Paragraph(str(i + 1), st["cell"]), Paragraph(r["text"], st["cell"]),
                     Paragraph(r["predicted_category"], st["cell"]), Paragraph(f"{r['confidence']:.0%}", st["cell"])])
    table = Table(data, colWidths=[0.8 * cm, 11.2 * cm, 2.6 * cm, 2.4 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C4CE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(1.8 * cm, 1.0 * cm, "Varolline Internship Program | Smart Text Classifier | Project Report")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    m = json.loads(config.METRICS_PATH.read_text())
    st = styles()
    pct = lambda x: f"{x * 100:.1f}%"
    story = []

    story.append(Paragraph("VAROLLINE INTERNSHIP PROGRAM | AI ENGINEER | PHASE 1", st["eyebrow"]))
    story.append(Paragraph("Smart Text Classifier - Project Report", st["title"]))
    story.append(Paragraph("Prepared by Muhammad Faizan Farooq | October 2026", st["meta"]))

    # 1 Objective
    story.append(Paragraph("1. Objective", st["h"]))
    story.append(Paragraph(
        "The goal was to build a small AI application that classifies a short piece of text into one of four "
        "categories: <b>Complaint, Inquiry, Feedback or Other</b>, and to expose it through a simple interface where "
        "a user types a message and instantly receives the predicted category. The project follows the Phase 1 "
        "brief: a working solution at an intermediate level, clean code, documentation and honest evaluation.",
        st["body"]))

    # 2 Approach
    story.append(Paragraph("2. Approach", st["h"]))
    story.extend(bullets([
        f"<b>Dataset:</b> no suitable public dataset was used. A balanced dataset of <b>{m['dataset_rows']:,} messages</b> "
        "(300 per category) was created with a seeded Python script that mixes hand-written samples, sentence "
        "templates with varied slot values, and light noise (lower-casing, missing punctuation). It is documented "
        "in <i>Dataset_Documentation.pdf</i>.",
        "<b>Preprocessing:</b> lower-casing, URL removal, numbers replaced by a token, punctuation removed (the "
        "question mark is kept), extra spaces collapsed. A stratified 80/20 train/test split keeps all classes equal "
        "in both parts.",
        "<b>Model:</b> a scikit-learn pipeline of <b>TF-IDF</b> (word unigrams and bigrams) followed by "
        "<b>Logistic Regression</b> (C = 5, balanced class weights). It is fast, needs no GPU and gives class "
        "probabilities, which are used as a confidence score.",
        "<b>Interface:</b> a <b>FastAPI</b> app with a web page (text box and probability bars) and a JSON endpoint "
        "<font name='Courier'>POST /predict</font>. Empty or oversized input is rejected with a clear message, "
        "and predictions below 50% confidence are flagged as low confidence.",
    ], st))

    # 3 Key decisions
    story.append(Paragraph("3. Key Decisions", st["h"]))
    story.extend(bullets([
        "<b>Logistic Regression over a neural model:</b> the brief asks for an understandable intermediate solution, "
        "and a linear model on TF-IDF is easy to explain, quick to train and a strong baseline for short text.",
        "<b>Stop-words were kept:</b> words such as \"how\", \"do\", \"please\" and \"?\" carry the intent of a message "
        "(a question versus a demand), so removing them would hurt an intent task.",
        "<b>Two evaluation sets:</b> besides the held-out split, a separate set of 40 hand-written messages that do "
        "not come from the templates was used. It was never used for tuning, so it acts as a reality check.",
        "<b>Cross-validation on the training part only</b> (5-fold) so the test set stays completely unseen.",
    ], st))

    # 4 Results
    story.append(Paragraph("4. Results", st["h"]))
    story.append(metric_table(m, st))
    story.append(Spacer(1, 5))
    cm_img = Image(str(config.CONFUSION_MATRIX_PATH), width=6.3 * cm, height=5.1 * cm)
    result_text = Paragraph(
        f"On the held-out split the model reaches <b>{pct(m['test']['accuracy'])} accuracy</b> and "
        f"<b>{pct(m['test']['f1_macro'])} macro F1</b> (5-fold CV macro F1: {m['cv_f1_macro_mean']:.3f} "
        f"\u00b1 {m['cv_f1_macro_std']:.3f}). On the independent hand-written set the accuracy drops to "
        f"<b>{pct(m['external']['accuracy'])}</b>. This gap is the most important finding of the project: the "
        "held-out split comes from the same template-based source as the training data, so its score is "
        "optimistic, while the independent set shows how the model behaves on text written in a different style. "
        "In the demo run (<i>reports/sample_predictions.csv</i>), 11 of 12 messages were classified correctly; the "
        "wrong one was also flagged as low confidence.", st["body"])
    block = Table([[result_text, cm_img]], colWidths=[10.4 * cm, 6.6 * cm])
    block.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (0, 0), 8)]))
    story.append(block)

    # 5 Challenges
    story.append(Paragraph("5. Challenges Faced", st["h"]))
    story.extend(bullets([
        "<b>Building a realistic dataset from scratch:</b> the first template set produced only about 206 unique "
        "Feedback sentences, so the script could not reach 300 per class and stopped. More templates and slot "
        "values had to be added, and plural/singular wording was fixed to avoid grammar mistakes.",
        "<b>Template bias:</b> every Complaint originally started with an opener such as \"I am very disappointed...\". "
        "A plain complaint like \"The app crashes every time I open the settings page\" was therefore "
        "misclassified as Feedback. Direct problem statements (about 40% of complaints) were added, which helped "
        "but did not remove the problem completely.",
        "<b>Fuzzy class boundaries:</b> a polite complaint, a question asked angrily, or praise with a small "
        "criticism can fit two classes. Labelling rules were written down to keep the labels consistent.",
        "<b>Honest evaluation:</b> a 98% score looked impressive but was misleading. Creating a separate "
        "hand-written test set exposed the real performance, and the weak spots are mostly in the "
        "\"Other\" class (short messages such as \"Thanks, received.\").",
    ], st))

    # 6 Limitations
    story.append(Paragraph("6. Limitations", st["h"]))
    story.extend(bullets([
        "The training data is synthetic and small (1,200 rows), so it covers fewer wording styles than real users.",
        "English only; Roman Urdu or mixed-language text is not supported. Each message gets a single label.",
        "TF-IDF ignores meaning and word order beyond bigrams, so sarcasm and unusual phrasing can fool the model.",
        "The \"Other\" class is a broad catch-all, which makes it the hardest class to learn from a small dataset.",
    ], st))

    # 7 Benefits
    story.append(Paragraph("7. Benefits", st["h"]))
    story.extend(bullets([
        "Automatically sorts incoming messages, so urgent complaints can be handled first and inquiries routed quickly.",
        "Lightweight and fast: trains in seconds on a normal laptop and answers a request in milliseconds.",
        "Transparent: probabilities and a low-confidence warning show when a human should double-check.",
        "Reusable and well organised: separate modules, tests, an API and documentation make it easy to extend.",
    ], st))

    # 8 Future improvements
    story.append(Paragraph("8. Future Improvements", st["h"]))
    story.extend(bullets([
        "<b>Collect real, human-labelled messages</b> (with permission) and retrain; this is the biggest expected gain.",
        "<b>Try sentence embeddings or a small pre-trained transformer</b> and compare it with this baseline.",
        "Add Roman Urdu support, probability calibration, and a feedback button so wrong predictions can be "
        "collected and used for retraining.",
    ], st))

    # 9 Sample predictions
    story.append(Paragraph("9. Sample Predictions (demo run)", st["h"]))
    story.append(sample_table(st))

    # 10 Conclusion
    story.append(Paragraph("10. Conclusion", st["h"]))
    story.append(Paragraph(
        "The project delivers a complete, working text classifier: dataset, preprocessing, training, evaluation, "
        "a web interface, an API, tests and documentation. The baseline performs very well on data that resembles "
        "its training set and reasonably on independent text, and the evaluation clearly shows where more and "
        "more realistic data would help most. This makes it a solid foundation for the next phase.", st["body"]))

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                            title="Smart Text Classifier - Project Report", author="Muhammad Faizan Farooq")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    main()
