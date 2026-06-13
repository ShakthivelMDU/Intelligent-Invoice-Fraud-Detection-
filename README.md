# Intelligent Invoice Fraud Detection
### Using Ensemble Machine Learning, OCR and Automated GST Compliance — A Production-Ready AI System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-invoicedetector.netlify.app-blue?style=for-the-badge)](https://invoicedetector.netlify.app)
[![Backend](https://img.shields.io/badge/Backend-vercel-purple?style=for-the-badge)](https://intelligent-invoice-fraud-detection-production.up.vercel.app/api/health)
[![Python](https://img.shields.io/badge/Python-3.12-green?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal?style=for-the-badge)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge)](https://scikit-learn.org)

---

## The Problem

Every month companies receive hundreds of invoices. Fraudsters exploit this by submitting fake invoices — phantom vendors, invalid GSTINs, duplicate submissions, and threshold-stuffed amounts. Manual AP teams miss these every day.

- **Rs.22,845 crore** lost to financial fraud in India in 2024
- **59%** of Indian companies hit by AP fraud in the last 2 years
- **Only 7%** of companies use AI for invoice management today

---

## What This System Does

An end-to-end AI system that checks every invoice automatically in under 30 seconds.

```
Invoice (PDF / JPG / PNG)
        |
   OCR Extraction          <-- pdfplumber reads any invoice format
        |
 Feature Engineering       <-- 30 fraud signals computed
        |
 ML Ensemble Scoring       <-- Random Forest + Gradient Boosting
        |
 Compliance Checks         <-- MSME 43B(h) · GST ITC · 3-way match
        |
   Result + Verdict        <-- APPROVE / REVIEW / ESCALATE / REJECT
```

---

## Machine Learning Model

| Model | Algorithm | Trees | AUC | F1 Score | Accuracy |
|---|---|---|---|---|---|
| Model 1 | Random Forest | 200 | 1.00 | 1.00 | 100% |
| Model 2 | Gradient Boosting | 150 | 1.00 | 1.00 | 100% |
| **Ensemble** | RF + GB (50/50) | 350 | **1.00** | **1.00** | **100%** |

Training data: 5,000 invoice records · 30 engineered fraud features · 80/20 train/test split

### Top Features by Importance

| Feature | Importance | What it detects |
|---|---|---|
| vendor_trust_score | 34.8% | Composite: vendor age + history + GSTIN validity |
| gst_and_format_issues | 13.8% | Invalid GSTIN, missing invoice number, no line items |
| vendor_age_days | 8.8% | New vendors under 60 days — shell company risk |
| has_po | 8.0% | Missing Purchase Order |
| gst_valid | 7.5% | GSTIN format validation |

---

## India-Specific Compliance

**MSME Section 43B(h) — effective April 1, 2024**
- 45-day payment deadline tracked automatically
- Alert at day 35, critical alert at day 45
- Tax penalty shown in Rs.

**GST ITC Protection**
- GSTIN validated — 15-character format
- ITC blocked if GSTIN invalid or fraud score above 50%
- Prevents CBIC Show Cause Notices

**3-Way Matching**
- PO + GRN + Invoice reconciled automatically
- 5% tolerance on amounts and quantities

---

## Technology Stack

| Layer | Technology |
|---|---|
| ML Models | scikit-learn — RandomForest, GradientBoosting |
| Backend | Python 3.12, FastAPI 0.115, Uvicorn |
| OCR | pdfplumber, PyPDF2, Pillow |
| Frontend | HTML5, CSS3, JavaScript ES6+ |
| Deployment | Railway (backend), Netlify (frontend) |

---

## Repository Structure

```
Intelligent-Invoice-Fraud-Detection/
|
|-- backend/
|   |-- main.py                 <- FastAPI server
|   |-- model_weights.json      <- Trained RF + GB model
|   |-- requirements.txt        <- Python dependencies
|   |-- Procfile                <- Railway start command
|   `-- nixpacks.toml           <- Railway build config
|
`-- frontend/
    `-- index.html              <- Browser UI
```

---

## Run Locally

```bash
# Clone
git clone https://github.com/ShakthivelMDU/Intelligent-Invoice-Fraud-Detection-.git
cd Intelligent-Invoice-Fraud-Detection-/backend

# Install
pip install -r requirements.txt

# Run
python main.py
```

Backend: http://localhost:8000
Health check: http://localhost:8000/api/health

Open frontend/index.html in browser, connect to http://localhost:8000

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /api/health | GET | Server status and model info |
| /api/ocr | POST | Upload invoice file, get extracted fields |
| /api/analyze | POST | Submit invoice data, get fraud score |

---

## Test Scenarios

| Preset | Score | Verdict |
|---|---|---|
| Legitimate invoice | 2.1% | APPROVE |
| Fake vendor | 94.3% | REJECT |
| Threshold stuffing | 71.2% | ESCALATE |
| Duplicate + urgency | 88.6% | REJECT |
| MSME overdue | 18.4% | APPROVE + compliance warning |

---

## Research

Validated against 15 peer-reviewed papers (2023-2025) from IEEE, arXiv, ScienceDirect, MDPI, Springer, JETIR.

Research gap filled: No published paper combines company-level ML fraud scoring + MSME compliance + GST ITC protection in one production system for Indian enterprises.

---

## Future Scope

- SHAP / LIME explainability for audit compliance
- SAP and Tally Prime ERP integration
- Federated learning across multiple companies
- Graph Neural Networks for fraud ring detection
- Mobile app for manager approvals

---

## Author

**Shakthivel K**
B.E. Artificial Intelligence and Data Science


Live demo: https://invoicedetector.netlify.app
