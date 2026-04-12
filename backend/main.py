"""
InvoiceGuard AI — Production Backend
FastAPI server that handles:
  1. POST /api/ocr       — reads invoice file using Claude AI, returns extracted fields
  2. POST /api/analyze   — takes invoice fields, runs ML model, returns fraud score
  3. GET  /api/health    — health check for Railway/Render
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import anthropic
import base64
import json
import math
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="InvoiceGuard AI", version="1.0.0")

# ── CORS — allow your Netlify frontend to call this backend ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # change to your Netlify URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Anthropic client (API key from environment variable) ──
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None

# ══════════════════════════════════════════════════════════
# ML MODEL — embedded weights (same model as frontend JS)
# ══════════════════════════════════════════════════════════

FEATURE_COLS = [
    "vendor_age_days","vendor_invoice_count","vendor_trust_score",
    "log_amount","near_threshold","amount_vs_vendor_avg_ratio",
    "has_po","has_grn","po_amount_match","qty_variance_pct",
    "gst_valid","gst_rate_correct","gst_deviation_pct",
    "payment_urgency","submitted_weekend","submitted_after_hours","days_to_due",
    "duplicate_score","same_vendor_30d_count",
    "has_invoice_number","has_line_items","has_bank_details","address_match_score",
    "is_msme","msme_days_overdue",
    "no_po_no_grn","new_vendor_high_amount","gst_and_format_issues",
    "urgency_no_po","high_duplicate_score",
]

APPROVAL_LIMITS = [50000, 100000, 500000, 1000000]

FEATURE_IMPORTANCE = {
    "vendor_trust_score": 0.348, "gst_and_format_issues": 0.138,
    "vendor_age_days": 0.088, "has_po": 0.080, "gst_valid": 0.075,
    "vendor_invoice_count": 0.042, "po_amount_match": 0.032,
    "has_invoice_number": 0.026, "no_po_no_grn": 0.025,
    "address_match_score": 0.022,
}


def traverse_tree(tree, vec):
    node = 0
    while True:
        f = tree["f"][node]
        if f == -2:
            return tree["v"][node]
        val = vec[f] if f < len(vec) else 0.0
        if val <= tree["th"][node]:
            node = tree["cl"][node]
        else:
            node = tree["cr"][node]


def predict_rf(trees, vec):
    total = sum(traverse_tree(t, vec) for t in trees)
    return total / len(trees)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def predict_gb(trees, vec, lr, init_val):
    score = init_val
    for t in trees:
        score += lr * traverse_tree(t, vec)
    return sigmoid(score)


def validate_gstin(gstin: str) -> bool:
    import re
    if not gstin:
        return False
    pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$'
    return bool(re.match(pattern, gstin.strip().upper()))


def extract_features(inv: dict) -> dict:
    amount = float(inv.get("invoice_amount") or 0)
    gst_amt = float(inv.get("gst_amount") or 0)
    vendor_age = int(inv.get("vendor_age_days") or 365)
    vendor_count = int(inv.get("vendor_invoice_count") or 10)
    vendor_avg = float(inv.get("vendor_avg_amount") or 75000) or 75000

    gst_valid = 1 if validate_gstin(str(inv.get("vendor_gstin", ""))) else 0

    expected_gst = amount * 0.18
    gst_dev = abs(gst_amt - expected_gst) / max(expected_gst, 1) * 100

    has_po = 1 if str(inv.get("po_number", "")).strip() not in ["", "0", "None", "null"] else 0
    has_grn = 1 if str(inv.get("grn_number", "")).strip() not in ["", "0", "None", "null"] else 0
    po_amt = float(inv.get("po_amount") or 0)
    po_match = 1 if po_amt > 0 and abs(po_amt - amount) / max(amount, 1) < 0.05 else 0
    qty_var = float(inv.get("qty_variance_pct") or 0)

    near_threshold = 1 if any(abs(amount - lim) < 2500 for lim in APPROVAL_LIMITS) else 0
    amount_ratio = min(amount / vendor_avg, 50)

    has_inv_no = 1 if str(inv.get("invoice_number", "")).strip() else 0
    has_line_items = 1 if inv.get("has_line_items") else 0
    has_bank_details = 1 if inv.get("has_bank_details") else 0

    addr_score = float(inv.get("address_match_score") or 85)
    addr_match = min(1.0, addr_score / 100.0 if addr_score > 1 else addr_score)

    is_msme = 1 if inv.get("is_msme") else 0
    msme_overdue = int(inv.get("msme_days_overdue") or 0) if is_msme else 0

    pay_urgency = 1 if inv.get("payment_urgency") else 0
    submitted_weekend = 1 if inv.get("submitted_weekend") else 0
    submitted_hour = int(inv.get("submitted_hour") or 10)
    after_hours = 1 if submitted_hour < 7 or submitted_hour > 20 else 0

    from datetime import datetime
    days_to_due = 30
    try:
        due = datetime.strptime(str(inv.get("due_date", "")), "%Y-%m-%d")
        days_to_due = max(0, (due - datetime.now()).days)
    except:
        pass

    dup_score_raw = float(inv.get("duplicate_score") or 0)
    dup_score = min(1.0, dup_score_raw / 100.0 if dup_score_raw > 1 else dup_score_raw)
    same_vendor_30d = int(inv.get("same_vendor_30d_count") or 0)

    trust = min(1.0,
        (min(vendor_age, 1000) / 1000) * 0.4 +
        (min(vendor_count, 50) / 50) * 0.3 +
        gst_valid * 0.3
    )

    no_po_no_grn = 1 if (not has_po and not has_grn) else 0
    new_vendor_high = 1 if (vendor_age < 60 and amount > 100000) else 0
    gst_format_issues = 1 if (not gst_valid or not has_inv_no or not has_line_items) else 0
    urgency_no_po = 1 if (pay_urgency and not has_po) else 0
    high_dup = 1 if dup_score > 0.5 else 0
    gst_rate_ok = 1 if gst_dev < 10 else 0

    return {
        "vendor_age_days": vendor_age,
        "vendor_invoice_count": vendor_count,
        "vendor_trust_score": trust,
        "log_amount": math.log1p(amount),
        "near_threshold": near_threshold,
        "amount_vs_vendor_avg_ratio": amount_ratio,
        "has_po": has_po,
        "has_grn": has_grn,
        "po_amount_match": po_match,
        "qty_variance_pct": qty_var,
        "gst_valid": gst_valid,
        "gst_rate_correct": gst_rate_ok,
        "gst_deviation_pct": gst_dev,
        "payment_urgency": pay_urgency,
        "submitted_weekend": submitted_weekend,
        "submitted_after_hours": after_hours,
        "days_to_due": days_to_due,
        "duplicate_score": dup_score,
        "same_vendor_30d_count": same_vendor_30d,
        "has_invoice_number": has_inv_no,
        "has_line_items": has_line_items,
        "has_bank_details": has_bank_details,
        "address_match_score": addr_match,
        "is_msme": is_msme,
        "msme_days_overdue": msme_overdue,
        "no_po_no_grn": no_po_no_grn,
        "new_vendor_high_amount": new_vendor_high,
        "gst_and_format_issues": gst_format_issues,
        "urgency_no_po": urgency_no_po,
        "high_duplicate_score": high_dup,
        "_amount": amount,
        "_gst_amt": gst_amt,
        "_vendor_age": vendor_age,
        "_gst_dev": gst_dev,
        "_dup_score": dup_score,
        "_same_vendor_30d": same_vendor_30d,
        "_qty_var": qty_var,
        "_days_to_due": days_to_due,
        "_gstin": str(inv.get("vendor_gstin", "")),
    }


def generate_flags(f: dict, prob: float) -> dict:
    flags, positives, compliance = [], [], []

    if not f["has_po"]:
        flags.append({"level": "high", "msg": "No Purchase Order — invoice not pre-authorized"})
    if not f["has_grn"]:
        flags.append({"level": "high", "msg": "No Goods Receipt Note — delivery not confirmed"})
    if not f["gst_valid"]:
        flags.append({"level": "high", "msg": f"GSTIN '{f['_gstin'].upper() or 'missing'}' failed format validation"})
    if f["_vendor_age"] < 60:
        flags.append({"level": "high", "msg": f"Vendor registered only {f['_vendor_age']} days ago"})
    if f["near_threshold"]:
        flags.append({"level": "medium", "msg": f"Amount ₹{f['_amount']:,.0f} is suspiciously close to approval threshold"})
    if f["_dup_score"] > 0.5:
        flags.append({"level": "high", "msg": f"Duplicate similarity {round(f['_dup_score']*100)}% — possible resubmission"})
    if f["urgency_no_po"]:
        flags.append({"level": "high", "msg": "Urgent payment demand with no PO — classic BEC attack pattern"})
    if f["submitted_weekend"]:
        flags.append({"level": "low", "msg": "Invoice submitted on a weekend"})
    if f["submitted_after_hours"]:
        flags.append({"level": "low", "msg": "Submitted outside business hours"})
    if f["_gst_dev"] > 10:
        flags.append({"level": "medium", "msg": f"GST deviates {f['_gst_dev']:.1f}% from expected 18%"})
    if f["_same_vendor_30d"] > 5:
        flags.append({"level": "medium", "msg": f"{f['_same_vendor_30d']} invoices from vendor in last 30 days"})
    if not f["has_line_items"]:
        flags.append({"level": "medium", "msg": "No itemized line items — lump-sum billing"})
    if f["new_vendor_high_amount"]:
        flags.append({"level": "high", "msg": f"New vendor ({f['_vendor_age']} days) with high-value invoice"})

    if f["has_po"] and f["has_grn"] and f["po_amount_match"]:
        positives.append("3-way match passed: PO, GRN, and invoice align")
    if f["gst_valid"]:
        positives.append("GSTIN validated successfully")
    if f["_vendor_age"] > 365:
        positives.append(f"Established vendor ({f['_vendor_age']} days history)")
    if f["vendor_trust_score"] > 0.7:
        positives.append(f"High vendor trust score ({round(f['vendor_trust_score']*100)}%)")
    if f["_dup_score"] < 0.1:
        positives.append("No duplicate invoice detected")

    if f["is_msme"]:
        od = f["msme_days_overdue"]
        if od > 35:
            compliance.append({"type": "danger", "msg": f"MSME 43B(h): {od} days elapsed — OVERDUE, legal risk"})
        elif od > 25:
            compliance.append({"type": "warning", "msg": f"MSME 43B(h): {od} days elapsed — {45-od} days remaining"})
        else:
            compliance.append({"type": "ok", "msg": f"MSME 43B(h): {od} days elapsed — within 45-day limit"})

    itc_blocked = not f["gst_valid"] or prob > 0.5
    if itc_blocked:
        compliance.append({"type": "danger", "msg": f"GST ITC BLOCKED — {'invalid GSTIN' if not f['gst_valid'] else 'high fraud risk'}"})
    else:
        compliance.append({"type": "ok", "msg": f"GST ITC of ₹{f['_gst_amt']:,.0f} is claimable on approval"})

    return {"flags": flags, "positives": positives, "compliance": compliance}


# ══════════════════════════════════════════════════════════
# LOAD MODEL WEIGHTS FROM FILE
# ══════════════════════════════════════════════════════════

MODEL = None

def load_model():
    global MODEL
    model_path = os.path.join(os.path.dirname(__file__), "model_weights.json")
    if os.path.exists(model_path):
        with open(model_path) as f:
            MODEL = json.load(f)
        print(f"Model loaded: {len(MODEL['rf'])} RF trees, {len(MODEL['gb'])} GB trees")
    else:
        print("WARNING: model_weights.json not found. Using rule-based scoring fallback.")


@app.on_event("startup")
async def startup():
    load_model()


# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "ocr_available": client is not None,
    }


@app.post("/api/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    """
    Accepts a PDF or image file.
    Sends it to Claude AI.
    Returns extracted invoice fields as JSON.
    """
    if not client:
        raise HTTPException(
            status_code=503,
            detail="OCR unavailable — ANTHROPIC_API_KEY not set on server. Add it to Railway environment variables."
        )

    allowed = ["application/pdf", "image/jpeg", "image/png", "image/webp", "image/gif"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    content = await file.read()
    b64 = base64.standard_b64encode(content).decode("utf-8")

    is_pdf = file.content_type == "application/pdf"
    content_item = (
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}}
        if is_pdf else
        {"type": "image", "source": {"type": "base64", "media_type": file.content_type, "data": b64}}
    )

    prompt = """You are an expert invoice OCR system for Indian businesses.
Extract ALL available fields from this invoice and return ONLY a valid JSON object.
No markdown, no explanation — just raw JSON.

Return exactly this structure (use null if not found):
{
  "vendor_name": "company name of the seller",
  "vendor_gstin": "15-character GSTIN or null",
  "invoice_number": "invoice ID or number",
  "invoice_date": "YYYY-MM-DD format",
  "due_date": "YYYY-MM-DD format or null",
  "invoice_amount": numeric amount BEFORE GST (number only),
  "gst_amount": numeric GST charged (number only),
  "total_amount": numeric total payable (number only),
  "po_number": "purchase order number or null",
  "category": one of ["IT Services","Cloud Services","Software License","Consulting","Hardware","Maintenance","Training","Other"],
  "has_line_items": true or false,
  "has_bank_details": true or false,
  "is_msme": true if MSME registered else false,
  "payment_terms": "e.g. Net 30 or null",
  "vendor_address": "full address or null",
  "buyer_name": "buying company name or null",
  "line_items_count": number,
  "currency": "INR",
  "confidence": 0-100 integer
}

Return ONLY the JSON."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [content_item, {"type": "text", "text": prompt}]
            }]
        )
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return {"success": True, "data": parsed}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"AI response was not valid JSON: {str(e)}")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


class InvoiceData(BaseModel):
    vendor_name: Optional[str] = ""
    vendor_gstin: Optional[str] = ""
    vendor_age_days: Optional[int] = 365
    vendor_invoice_count: Optional[int] = 10
    invoice_number: Optional[str] = ""
    invoice_date: Optional[str] = ""
    invoice_amount: Optional[float] = 0
    gst_amount: Optional[float] = 0
    due_date: Optional[str] = ""
    category: Optional[str] = "IT Services"
    po_number: Optional[str] = ""
    po_amount: Optional[float] = 0
    grn_number: Optional[str] = ""
    qty_variance_pct: Optional[float] = 0
    is_msme: Optional[bool] = False
    payment_urgency: Optional[bool] = False
    has_line_items: Optional[bool] = True
    has_bank_details: Optional[bool] = True
    submitted_weekend: Optional[bool] = False
    same_vendor_30d_count: Optional[int] = 0
    duplicate_score: Optional[float] = 0
    address_match_score: Optional[float] = 85
    msme_days_overdue: Optional[int] = 0


@app.post("/api/analyze")
def analyze_invoice(data: InvoiceData):
    """
    Accepts invoice fields.
    Runs RF + GB ensemble ML model.
    Returns fraud score, verdict, flags, compliance status.
    """
    inv = data.dict()
    features = extract_features(inv)
    vec = [features.get(k, 0) for k in FEATURE_COLS]

    if MODEL:
        rf_prob = predict_rf(MODEL["rf"], vec)
        gb_prob = predict_gb(MODEL["gb"], vec, MODEL["gb_lr"], MODEL["gb_init"])
    else:
        # Rule-based fallback if model not loaded
        risk_score = 0.0
        if not features["has_po"]: risk_score += 0.25
        if not features["gst_valid"]: risk_score += 0.30
        if features["_vendor_age"] < 30: risk_score += 0.20
        if features["urgency_no_po"]: risk_score += 0.15
        if features["high_duplicate_score"]: risk_score += 0.10
        rf_prob = gb_prob = min(risk_score, 0.99)

    ensemble = (rf_prob + gb_prob) / 2.0

    if ensemble < 0.25:
        verdict = "APPROVE"
    elif ensemble < 0.50:
        verdict = "REVIEW"
    elif ensemble < 0.75:
        verdict = "ESCALATE"
    else:
        verdict = "REJECT"

    explanation = generate_flags(features, ensemble)

    top_features = {
        k: round(float(features.get(k, 0)), 4)
        for k in FEATURE_IMPORTANCE
    }

    return {
        "success": True,
        "fraud_score": round(ensemble * 100, 1),
        "rf_score": round(rf_prob * 100, 1),
        "gb_score": round(gb_prob * 100, 1),
        "verdict": verdict,
        "flags": explanation["flags"],
        "positives": explanation["positives"],
        "compliance": explanation["compliance"],
        "feature_importance": FEATURE_IMPORTANCE,
        "top_features": top_features,
        "is_msme": bool(features["is_msme"]),
        "msme_days_overdue": features["msme_days_overdue"],
        "itc_blocked": not bool(features["gst_valid"]) or ensemble > 0.5,
        "gst_valid": bool(features["gst_valid"]),
    }
    if __name__ == "__main__":
      import uvicorn
      port = int(os.environ.get("PORT", 8000))
      uvicorn.run(app, host="0.0.0.0", port=port)
