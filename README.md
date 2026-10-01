# Smart Text Classifier

> Varolline Internship Program | AI Engineer | Phase 1 Assignment
> Author: Muhammad Faizan Farooq

## Description

Smart Text Classifier is a small NLP application that reads a short message and predicts which of four categories it belongs to:

| Category | Meaning | Example |
|---|---|---|
| **Complaint** | The user reports a problem and expects action | "I was charged twice, please refund me." |
| **Inquiry** | The user asks for information | "What are your delivery charges for Lahore?" |
| **Feedback** | The user shares an opinion, praise or a suggestion | "I really like the new design." |
| **Other** | Greetings, announcements, spam, small talk | "Good morning everyone." |

**How it works:** the text is cleaned, converted to numbers with **TF-IDF** (single words and word pairs), and classified by a **Logistic Regression** model. A **FastAPI** web page and JSON API let a user type text and instantly get the predicted category with a confidence score.

**Features**

- Reproducible, balanced dataset (1,200 messages, 300 per category) with a documented creation method
- Text preprocessing, stratified train/test split and 5-fold cross-validation
- Evaluation with accuracy, precision, recall and F1-score, plus confusion matrices
- A second, independent hand-written test set (40 messages) as a reality check
- FastAPI web interface and `POST /predict` API with input validation and a low-confidence warning
- 12 demo predictions, unit tests, and a 2-page project report

## Project Structure

```
smart-text-classifier/
├── data/
│   ├── text_classification_dataset.csv   # 1,200 labelled messages (training data)
│   └── external_test_samples.csv         # 40 independent hand-written test messages
├── docs/
│   ├── Dataset_Documentation.pdf         # How the dataset was created and its statistics
│   └── Project_Report.pdf                # 2-page report (approach, challenges, limitations, benefits)
├── models/
│   └── text_classifier.joblib            # Trained model (created by src/train.py)
├── reports/
│   ├── metrics.json                      # Saved evaluation metrics
│   ├── confusion_matrix.png              # Held-out test set
│   ├── confusion_matrix_external.png     # Hand-written test set
│   └── sample_predictions.csv            # 12 demo predictions
├── src/
│   ├── config.py                         # Paths and settings
│   ├── generate_dataset.py               # Creates the dataset CSV
│   ├── generate_dataset_pdf.py           # Creates Dataset_Documentation.pdf
│   ├── preprocess.py                     # Text cleaning and data loading
│   ├── train.py                          # Training + evaluation + saving the model
│   ├── predict.py                        # Loads the model and predicts new text
│   ├── demo_predictions.py               # Prints/saves the 12 sample predictions
│   ├── app.py                            # FastAPI app (web page + API)
│   └── generate_report_pdf.py            # Creates Project_Report.pdf from saved metrics
├── tests/
│   ├── test_preprocess.py                # Tests for text cleaning and data loading
│   └── test_api.py                       # Tests for the API endpoints
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

**Requirements:** Python 3.9 or newer.

**1. Create a virtual environment and install the libraries**

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

**2. (Optional) Re-create the dataset and its PDF**

The CSV is already included. Run this only if you want to regenerate it (seed 42 gives the same result every time).

```bash
python src/generate_dataset.py
python src/generate_dataset_pdf.py
```

**3. Train and evaluate the model**

```bash
python -m src.train
```

This prints the evaluation results and saves the model to `models/` and the metrics and confusion matrices to `reports/`.

**4. See the 12 sample predictions**

```bash
python -m src.demo_predictions
```

**5. Start the web interface / API**

```bash
uvicorn src.app:app --reload
```

Open **http://127.0.0.1:8000** in your browser, type a message and click **Classify**. Interactive API documentation is available at **http://127.0.0.1:8000/docs**.

**6. Run the tests**

```bash
python -m pytest
```

**7. (Optional) Re-create the project report PDF** after retraining

```bash
python -m src.generate_report_pdf
```

> Always run the commands from the project root folder (the folder that contains `src/`).

## Output

### Training and evaluation (`python -m src.train`)

```
Loaded 1200 rows. Class counts: Complaint 300 | Inquiry 300 | Feedback 300 | Other 300
Train size: 960 | Test size: 240
5-fold CV macro F1 (train only): 0.9938 +/- 0.0039

Held-out test set results
              precision    recall  f1-score   support
   Complaint      0.983     0.983     0.983        60
     Inquiry      0.968     1.000     0.984        60
    Feedback      0.968     1.000     0.984        60
       Other      1.000     0.933     0.966        60

    accuracy                          0.979       240
   macro avg      0.980     0.979     0.979       240
```

| Evaluation set | Rows | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| Held-out test split (20%) | 240 | 97.9% | 98.0% | 97.9% | 97.9% |
| Independent hand-written set | 40 | 77.5% | 78.8% | 77.5% | 76.7% |

**Important:** the held-out score is high because the dataset is synthetic and template-based, so the test split looks very similar to the training data. The independent hand-written set is a more realistic estimate of how the model behaves on new text. See the limitations below and `docs/Project_Report.pdf`.

### Sample predictions (`python -m src.demo_predictions`)

| # | Message | Predicted | Confidence |
|---|---|---|---|
| 1 | My order arrived broken and nobody is answering my emails. | Complaint | 76% |
| 2 | I was charged twice this month, please refund the extra payment. | Complaint | 92% |
| 3 | The app crashes every time I open the settings page. | Feedback (wrong, should be Complaint) | 48% (low confidence) |
| 4 | What are your delivery charges for Karachi? | Inquiry | 82% |
| 5 | How can I change the phone number on my account? | Inquiry | 89% |
| 6 | Do you offer a discount for students? | Inquiry | 93% |
| 7 | I really like the new design, it feels very clean. | Feedback | 82% |
| 8 | It would be great to have an option to schedule deliveries. | Feedback | 86% |
| 9 | The support team was polite and solved my issue quickly. | Feedback | 47% (low confidence) |
| 10 | Good morning, hope you are doing well. | Other | 98% |
| 11 | Congratulations! You won a free phone, click here to claim. | Other | 84% |
| 12 | The meeting is rescheduled to Thursday at 3 pm. | Other | 97% |

11 of 12 are correct. The one error (#3) was flagged as low confidence by the model.

### Web interface

The page at `http://127.0.0.1:8000` shows a text box, a **Classify** button and example buttons. After classification it displays the category, the confidence percentage and a probability bar for every category. If the confidence is below 50%, a warning asks the user to double-check the result.

### API

`POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I reset my password?"}'
```

Response:

```json
{
  "text": "How do I reset my password?",
  "category": "Inquiry",
  "confidence": 0.9017,
  "low_confidence": false,
  "probabilities": {"Complaint": 0.0478, "Feedback": 0.02, "Inquiry": 0.9017, "Other": 0.0305}
}
```

Empty text, whitespace-only text, or text longer than 1,000 characters returns HTTP `422` with an error message. `GET /health` returns `{"status": "ok"}`.

## Dataset

No public dataset was copied. The dataset was **created for this task** with `src/generate_dataset.py`: hand-written samples, sentence templates with varied slot values (products, cities, issues, features), light noise (lower-casing, missing punctuation), duplicate removal and class balancing. Full details are in `docs/Dataset_Documentation.pdf`.

## Limitations

- The dataset is synthetic and small, so it covers fewer wording styles than real user messages. This is why the score on independent hand-written text (77.5%) is much lower than on the held-out split (97.9%).
- Plain complaints without strong complaint words can be misclassified (see sample #3), and the broad "Other" class is the hardest to learn.
- English only; Roman Urdu and mixed-language text are not supported.
- One label per message, although real messages can contain several intents.
- TF-IDF does not understand meaning, sarcasm or context beyond word pairs.

## Future Improvements

1. Collect real, human-labelled messages and retrain the model (largest expected gain).
2. Compare against sentence embeddings or a small pre-trained transformer model.
3. Add Roman Urdu support, probability calibration, and a "wrong prediction" button to collect data for retraining.

## Acknowledgements

Built with [scikit-learn](https://scikit-learn.org), [pandas](https://pandas.pydata.org), [FastAPI](https://fastapi.tiangolo.com), [Uvicorn](https://www.uvicorn.org), [matplotlib](https://matplotlib.org), [seaborn](https://seaborn.pydata.org) and [ReportLab](https://www.reportlab.com). The dataset was created during this internship task.
