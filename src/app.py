"""FastAPI application: web page + JSON API for the Smart Text Classifier.

Run from the project root:
    uvicorn src.app:app --reload
Then open http://127.0.0.1:8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from src.predict import TextClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts (not on every request).
    app.state.classifier = TextClassifier()
    yield


app = FastAPI(
    title="Smart Text Classifier",
    description="Classifies short text into Complaint, Inquiry, Feedback or Other.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Message to classify")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text must not be empty.")
        return value


class PredictResponse(BaseModel):
    text: str
    category: str
    confidence: float
    low_confidence: bool
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = app.state.classifier.predict(request.text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    return {"text": request.text, **result}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Text Classifier</title>
<style>
  :root { --navy:#1f3a5f; --teal:#1b7f79; --bg:#f4f7fa; --card:#ffffff; --text:#1e2a36; --muted:#64748b; --border:#d7e0e8; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; background:var(--bg); color:var(--text); }
  main { max-width: 720px; margin: 0 auto; padding: 32px 18px 48px; }
  .eyebrow { color:var(--teal); font-weight:700; font-size:12px; letter-spacing:.12em; text-transform:uppercase; }
  h1 { color:var(--navy); margin:6px 0 4px; font-size:30px; }
  p.sub { color:var(--muted); margin:0 0 22px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; }
  textarea { width:100%; min-height:110px; padding:12px; font:inherit; border:1px solid var(--border); border-radius:8px; resize:vertical; }
  textarea:focus { outline:2px solid var(--teal); border-color:transparent; }
  .row { display:flex; gap:10px; align-items:center; margin-top:12px; flex-wrap:wrap; }
  button { font:inherit; cursor:pointer; border:0; border-radius:8px; padding:10px 18px; }
  #go { background:var(--navy); color:#fff; font-weight:600; }
  #go:disabled { opacity:.6; cursor:wait; }
  .chip { background:#e8f0f7; color:var(--navy); font-size:13px; padding:6px 10px; }
  .hint { font-size:13px; color:var(--muted); margin:14px 0 0; }
  #result { margin-top:18px; display:none; }
  .badge { display:inline-block; padding:6px 14px; border-radius:999px; color:#fff; font-weight:700; }
  .Complaint{background:#c0392b} .Inquiry{background:#2471a3} .Feedback{background:#1e8449} .Other{background:#6c7a89}
  .bar { display:flex; align-items:center; gap:10px; margin:8px 0; font-size:14px; }
  .bar span.l { width:80px; } .bar span.v { width:48px; text-align:right; color:var(--muted); }
  .track { flex:1; height:10px; background:#e6edf3; border-radius:6px; overflow:hidden; }
  .fill { height:100%; background:var(--teal); width:0; transition:width .4s; }
  .warn { color:#9a6700; background:#fff8e1; border:1px solid #f1d98a; padding:8px 12px; border-radius:8px; margin-top:12px; font-size:14px; display:none; }
  .error { color:#a12622; margin-top:12px; font-size:14px; display:none; }
</style>
</head>
<body>
<main>
  <div class="eyebrow">Varolline Internship Program | AI Engineer Task</div>
  <h1>Smart Text Classifier</h1>
  <p class="sub">Type a short message and the model will classify it as Complaint, Inquiry, Feedback or Other.</p>

  <div class="card">
    <textarea id="text" maxlength="1000" placeholder="Example: I was charged twice for my order, please refund me."></textarea>
    <div class="row">
      <button id="go">Classify</button>
      <button class="chip" data-example="My parcel arrived damaged and nobody replies to my emails.">Complaint example</button>
      <button class="chip" data-example="What are your delivery charges for Lahore?">Inquiry example</button>
      <button class="chip" data-example="I love the new design, great work!">Feedback example</button>
    </div>
    <div class="error" id="error"></div>

    <div id="result">
      <hr style="border:0;border-top:1px solid var(--border);margin:18px 0">
      <div>Predicted category: <span class="badge" id="badge"></span>
        <span id="conf" style="margin-left:8px;color:var(--muted)"></span></div>
      <div class="warn" id="warn">Low confidence: the message is ambiguous, so please double-check this result.</div>
      <div id="bars" style="margin-top:14px"></div>
    </div>
  </div>
  <p class="hint">Model: TF-IDF + Logistic Regression. API docs are available at <a href="/docs">/docs</a>.</p>
</main>

<script>
const textEl = document.getElementById("text");
const goBtn = document.getElementById("go");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");

document.querySelectorAll("[data-example]").forEach(btn =>
  btn.addEventListener("click", () => { textEl.value = btn.dataset.example; textEl.focus(); }));

function showError(message) { errorEl.textContent = message; errorEl.style.display = "block"; resultEl.style.display = "none"; }

async function classify() {
  errorEl.style.display = "none";
  const text = textEl.value.trim();
  if (!text) { showError("Please enter some text first."); return; }
  goBtn.disabled = true; goBtn.textContent = "Classifying...";
  try {
    const response = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    if (response.status === 422) { showError("Please enter between 1 and 1000 characters of text."); return; }
    if (!response.ok) { showError("Something went wrong on the server. Please try again."); return; }
    render(await response.json());
  } catch (err) {
    showError("Could not reach the server. Is it running?");
  } finally {
    goBtn.disabled = false; goBtn.textContent = "Classify";
  }
}

function render(data) {
  const badge = document.getElementById("badge");
  badge.textContent = data.category; badge.className = "badge " + data.category;
  document.getElementById("conf").textContent = (data.confidence * 100).toFixed(1) + "% confidence";
  document.getElementById("warn").style.display = data.low_confidence ? "block" : "none";
  const bars = document.getElementById("bars"); bars.innerHTML = "";
  Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]).forEach(([label, p]) => {
    const row = document.createElement("div"); row.className = "bar";
    const l = document.createElement("span"); l.className = "l"; l.textContent = label;
    const track = document.createElement("div"); track.className = "track";
    const fill = document.createElement("div"); fill.className = "fill"; track.appendChild(fill);
    const v = document.createElement("span"); v.className = "v"; v.textContent = (p * 100).toFixed(1) + "%";
    row.append(l, track, v); bars.appendChild(row);
    requestAnimationFrame(() => { fill.style.width = (p * 100) + "%"; });
  });
  resultEl.style.display = "block";
}

goBtn.addEventListener("click", classify);
textEl.addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) classify(); });
</script>
</body>
</html>
"""
