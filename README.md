# Karunya Placement Intelligence System v2.0

An AI-powered placement analytics platform built with Python and Streamlit.

## Features
- 🔐 **Secure Login** — only @karunya.edu.in emails accepted
- 🎓 **Full Department List** — all Karunya programmes selectable
- 🤖 **ML Placement Prediction** — Random Forest / Gradient Boosting / Logistic Regression
- 🛠️ **Skill Recommendations** — mapped to your exact programme
- 🏢 **Company Eligibility** — 20 companies across 3 tiers
- 📊 **Analytics Dashboard** — Plotly-powered interactive charts
- 📤 **Dataset Upload** — upload CSV and retrain the model live

## Project Structure
```
karunya_placement/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
└── utils/
    ├── __init__.py
    ├── auth.py             # Email & register number validation
    ├── ml_model.py         # ML training & prediction
    ├── companies.py        # Company eligibility logic
    └── departments.py      # All Karunya departments + skill maps
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

## Login Format
- **Email:** `yourname@karunya.edu.in` (only this domain accepted)
- **Register Number:** `URK25AI1074` → URK (UG), 25 (year), AI (dept code), 1074 (roll)
- **Department:** Select from the full dropdown list
- **Current Year:** Select 1st–4th Year

## Dataset Upload Format (CSV)
| Column | Description |
|--------|-------------|
| CGPA | Float 5.0–10.0 |
| Internships | Integer 0–5 |
| Projects | Integer 0–10 |
| Certifications | Integer 0–10 |
| Communication_Skill | Integer 1–5 |
| Coding_Skill | Integer 1–5 |
| Placed | 0 or 1 |
