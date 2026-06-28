[Uploading README.md…]()
# 🏥 MediRisk AI

I built MediRisk AI to explore how machine learning can support clinical decision-making. It predicts the likelihood of a diabetic patient being readmitted to hospital within 30 days of discharge — giving clinicians an early signal to act on before the patient leaves.

🌐 **Try it live:** [subha-dhanusha.github.io/medirisk-ai](https://subha-dhanusha.github.io/medirisk-ai/)  
⚡ **API:** [medirisk-ai.onrender.com/docs](https://medirisk-ai.onrender.com/docs)

---

## What it does

You enter a patient's details — age, diagnoses, medications, visit history — and the model returns:

- A **risk score** from 0–100%
- A **risk level** (LOW, MEDIUM, or HIGH)
- A **clinical recommendation** (e.g. "Schedule follow-up within 7 days")
- The **top factors** driving that patient's risk

---

## How I built it

This was a full end-to-end project, built phase by phase:

1. **Data pipeline** — cleaned and processed the UCI Diabetes 130-US Hospitals dataset (100k+ records)
2. **Feature engineering** — created meaningful clinical features from raw diagnosis codes and medication data
3. **Model training** — trained an XGBoost classifier, used SMOTE to handle class imbalance
4. **API** — wrapped the model in a FastAPI backend with a `/predict` endpoint
5. **Frontend** — built a clean web interface in HTML/CSS/JS
6. **Deployment** — API live on Render, frontend on GitHub Pages

---

## Tech stack

- **ML:** XGBoost, scikit-learn, imbalanced-learn
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Hosting:** Render (API) + GitHub Pages (frontend)

---

## Run it locally

```bash
git clone https://github.com/Subha-Dhanusha/medirisk-ai.git
cd medirisk-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

Then open `http://localhost:8000/docs` to explore the API.

---

## Project structure

```
medirisk-ai/
├── src/
│   ├── data/        # data loading & cleaning
│   ├── features/    # feature engineering
│   ├── models/      # model training
│   └── api/         # FastAPI backend
├── docs/
│   └── index.html   # web frontend
├── models/          # saved XGBoost model
└── app.py           # Streamlit dashboard
```

---

## A note on this project

This was built for learning purposes and is not intended for real clinical use. The model's AUC reflects the complexity of readmission prediction — it's a genuinely hard problem even for clinical researchers. That said, the full pipeline from raw data to a live, usable web app was a great exercise in practical ML engineering.

---

**Built by [Subha Dhanusha](https://github.com/Subha-Dhanusha)**
