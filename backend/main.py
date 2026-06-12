import json, math, os, re, base64, io
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="InvoiceGuard AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://invoicedetector.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {
        "message": "InvoiceGuard AI Backend Running",
        "docs": "/docs",
        "health": "/api/health"
    }
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
    "has_invoice_number": 0.026, "no_po_no_grn": 0.025, "address_match_score": 0.022,
}

MODEL = None

def load_model():
    global MODEL
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights.json")
    if os.path.exists(path):
        with open(path) as f:
            MODEL = json.load(f)
        print(f"Model loaded: {len(MODEL['rf'])} RF trees, {len(MODEL['gb'])} GB trees")
    else:
        print("WARNING: model_weights.json not found. Rule-based fallback active.")

load_model()

def traverse(tree, vec):
    node = 0
    while True:
        f = tree["f"][node]
        if f == -2: return tree["v"][node]
        val = vec[f] if f < len(vec) else 0.0
        node = tree["cl"][node] if val <= tree["th"][node] else tree["cr"][node]

def predict_rf(trees, vec):
    return sum(traverse(t, vec) for t in trees) / len(trees)

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def predict_gb(trees, vec, lr, init_val):
    score = init_val
    for t in trees: score += lr * traverse(t, vec)
    return sigmoid(score)

def validate_gstin(g):
    if not g: return False
    return bool(re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$', str(g).strip().upper()))

# ══════════════════════════════════════════════════════════
# OCR — THREE METHODS, NO EXTERNAL TOOLS NEEDED
# Method 1: pdfplumber  (best — for digital PDFs)
# Method 2: PyPDF2      (fallback for digital PDFs)
# Method 3: pytesseract (for scanned images — optional)
# ══════════════════════════════════════════════════════════

def extract_text_pdf(pdf_bytes):
    """Extract text from digital PDF using pdfplumber — no Tesseract needed."""
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:2]:  # first 2 pages only
                t = page.extract_text()
                if t:
                    text += t + "\n"
        if text.strip():
            return text
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    # Fallback to PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages[:2]:
            text += page.extract_text() or ""
        if text.strip():
            return text
    except Exception as e:
        print(f"PyPDF2 failed: {e}")

    return text

def extract_text_image(image_bytes):
    """Extract text from image — tries pytesseract if installed."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return pytesseract.image_to_string(img, config='--oem 3 --psm 6')
    except ImportError:
        return "IMAGE_OCR_UNAVAILABLE: Install tesseract for image OCR. PDF invoices work without it."
    except Exception as e:
        return f"IMAGE_OCR_ERROR: {str(e)}"

def parse_invoice_text(text):
    """Parse raw text into structured invoice fields using regex."""
    result = {
        "vendor_name": None, "vendor_gstin": None, "invoice_number": None,
        "invoice_date": None, "due_date": None, "invoice_amount": None,
        "gst_amount": None, "total_amount": None, "po_number": None,
        "category": "IT Services", "has_line_items": False, "has_bank_details": False,
        "is_msme": False, "payment_terms": None, "vendor_address": None,
        "buyer_name": None, "confidence": 55,
    }

    if not text or "OCR_UNAVAILABLE" in text or "OCR_ERROR" in text:
        result["confidence"] = 0
        return result

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # GSTIN
    gstin = re.search(r'\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b', text.upper())
    if gstin:
        result["vendor_gstin"] = gstin.group(1)
        result["confidence"] += 15

    # Invoice number
    for pat in [r'invoice\s*(?:no|number|#)[.:\s]*([A-Z0-9\-/]+)',
                r'inv\.?\s*(?:no|#)[.:\s]*([A-Z0-9\-/]+)',
                r'bill\s*(?:no|number)[.:\s]*([A-Z0-9\-/]+)']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            result["invoice_number"] = m.group(1).strip()
            result["confidence"] += 10
            break

    # Dates — DD/MM/YYYY or YYYY-MM-DD
    dates = re.findall(r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b', text)
    if dates:
        result["invoice_date"] = dates[0]
        if len(dates) > 1:
            result["due_date"] = dates[1]

    # Total amount
    for pat in [r'(?:grand\s*total|total\s*amount|amount\s*payable|net\s*payable)[:\s]*(?:rs\.?|inr|₹)?\s*([\d,]+\.?\d*)',
                r'(?:total)[:\s]*(?:rs\.?|₹)\s*([\d,]+\.?\d*)']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                result["total_amount"] = float(m.group(1).replace(',', ''))
                result["confidence"] += 10
                break
            except: pass

    # GST
    gst_total = 0.0
    for m in re.finditer(r'(?:igst|cgst|sgst|gst)[^:]*[:\s]*(?:rs\.?|₹)?\s*([\d,]+\.?\d*)', text, re.IGNORECASE):
        try: gst_total += float(m.group(1).replace(',', ''))
        except: pass
    if gst_total > 0:
        result["gst_amount"] = round(gst_total, 2)
        result["confidence"] += 5

    # Calculate invoice amount
    if result["total_amount"] and result["gst_amount"]:
        result["invoice_amount"] = round(result["total_amount"] - result["gst_amount"], 2)
    elif result["total_amount"]:
        result["invoice_amount"] = round(result["total_amount"] / 1.18, 2)
        result["gst_amount"] = round(result["total_amount"] - result["invoice_amount"], 2)

    # PO number
    po = re.search(r'(?:p\.?o\.?|purchase\s*order)\s*(?:no|number|#)?[.:\s]*([A-Z0-9\-/]+)', text, re.IGNORECASE)
    if po:
        result["po_number"] = po.group(1).strip()

    # Bank details
    if re.search(r'(?:bank|account\s*no|ifsc|neft|rtgs|upi)', text, re.IGNORECASE):
        result["has_bank_details"] = True

    # Line items — multiple price rows
    if len(re.findall(r'(?:rs\.?|₹)\s*[\d,]+', text, re.IGNORECASE)) > 3:
        result["has_line_items"] = True

    # MSME
    if re.search(r'(?:msme|udyam|udyog\s*aadhaar)', text, re.IGNORECASE):
        result["is_msme"] = True

    # Vendor name — first meaningful line
    for line in lines[:6]:
        if len(line) > 4 and not re.match(r'^[\d\s\-/₹]+$', line) and 'invoice' not in line.lower():
            result["vendor_name"] = line[:80]
            break

    # Category
    tl = text.lower()
    if any(w in tl for w in ['software','saas','license','subscription']): result["category"] = "Software License"
    elif any(w in tl for w in ['cloud','aws','azure','hosting']): result["category"] = "Cloud Services"
    elif any(w in tl for w in ['consulting','advisory','consultancy']): result["category"] = "Consulting"
    elif any(w in tl for w in ['hardware','laptop','computer','equipment']): result["category"] = "Hardware"
    elif any(w in tl for w in ['maintenance','support','amc']): result["category"] = "Maintenance"
    elif any(w in tl for w in ['training','workshop','course']): result["category"] = "Training"

    result["confidence"] = min(result["confidence"], 92)
    return result

def extract_features(inv):
    amount = float(inv.get("invoice_amount") or 0)
    gst_amt = float(inv.get("gst_amount") or 0)
    vendor_age = int(inv.get("vendor_age_days") or 365)
    vendor_count = int(inv.get("vendor_invoice_count") or 10)
    vendor_avg = float(inv.get("vendor_avg_amount") or 75000) or 75000
    gst_valid = 1 if validate_gstin(str(inv.get("vendor_gstin", ""))) else 0
    expected_gst = amount * 0.18
    gst_dev = abs(gst_amt - expected_gst) / max(expected_gst, 1) * 100
    po_str = str(inv.get("po_number", "")).strip()
    has_po = 1 if po_str and po_str not in ["0","None","null",""] else 0
    grn_str = str(inv.get("grn_number", "")).strip()
    has_grn = 1 if grn_str and grn_str not in ["0","None","null",""] else 0
    po_amt = float(inv.get("po_amount") or 0)
    po_match = 1 if po_amt > 0 and abs(po_amt - amount) / max(amount, 1) < 0.05 else 0
    qty_var = float(inv.get("qty_variance_pct") or 0)
    near_threshold = 1 if any(abs(amount - lim) < 2500 for lim in APPROVAL_LIMITS) else 0
    amount_ratio = min(amount / vendor_avg, 50)
    inv_no = str(inv.get("invoice_number", "")).strip()
    has_inv_no = 1 if inv_no and inv_no not in ["None","null",""] else 0
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
    days_to_due = 30
    try:
        due = datetime.strptime(str(inv.get("due_date", "")), "%Y-%m-%d")
        days_to_due = max(0, (due - datetime.now()).days)
    except: pass
    dup_raw = float(inv.get("duplicate_score") or 0)
    dup_score = min(1.0, dup_raw / 100.0 if dup_raw > 1 else dup_raw)
    same_30d = int(inv.get("same_vendor_30d_count") or 0)
    trust = min(1.0, (min(vendor_age,1000)/1000)*0.4 + (min(vendor_count,50)/50)*0.3 + gst_valid*0.3)
    return {
        "vendor_age_days": vendor_age, "vendor_invoice_count": vendor_count, "vendor_trust_score": trust,
        "log_amount": math.log1p(amount), "near_threshold": near_threshold, "amount_vs_vendor_avg_ratio": amount_ratio,
        "has_po": has_po, "has_grn": has_grn, "po_amount_match": po_match, "qty_variance_pct": qty_var,
        "gst_valid": gst_valid, "gst_rate_correct": 1 if gst_dev < 10 else 0, "gst_deviation_pct": gst_dev,
        "payment_urgency": pay_urgency, "submitted_weekend": submitted_weekend, "submitted_after_hours": after_hours,
        "days_to_due": days_to_due, "duplicate_score": dup_score, "same_vendor_30d_count": same_30d,
        "has_invoice_number": has_inv_no, "has_line_items": has_line_items, "has_bank_details": has_bank_details,
        "address_match_score": addr_match, "is_msme": is_msme, "msme_days_overdue": msme_overdue,
        "no_po_no_grn": 1 if (not has_po and not has_grn) else 0,
        "new_vendor_high_amount": 1 if (vendor_age < 60 and amount > 100000) else 0,
        "gst_and_format_issues": 1 if (not gst_valid or not has_inv_no or not has_line_items) else 0,
        "urgency_no_po": 1 if (pay_urgency and not has_po) else 0,
        "high_duplicate_score": 1 if dup_score > 0.5 else 0,
        "_amount": amount, "_gst_amt": gst_amt, "_vendor_age": vendor_age,
        "_gst_dev": gst_dev, "_dup_score": dup_score, "_same_vendor_30d": same_30d,
        "_gstin": str(inv.get("vendor_gstin", "")),
    }

def generate_flags(f, prob):
    flags, positives, compliance = [], [], []
    if not f["has_po"]: flags.append({"level":"high","msg":"No Purchase Order — invoice not pre-authorized"})
    if not f["has_grn"]: flags.append({"level":"high","msg":"No Goods Receipt Note — delivery not confirmed"})
    if not f["gst_valid"]: flags.append({"level":"high","msg":f"GSTIN '{f['_gstin'].upper() or 'missing'}' failed validation"})
    if f["_vendor_age"] < 60: flags.append({"level":"high","msg":f"Vendor registered only {f['_vendor_age']} days ago"})
    if f["near_threshold"]: flags.append({"level":"medium","msg":"Amount near approval threshold"})
    if f["_dup_score"] > 0.5: flags.append({"level":"high","msg":f"Duplicate similarity {round(f['_dup_score']*100)}%"})
    if f["urgency_no_po"]: flags.append({"level":"high","msg":"Urgent payment demand with no PO — BEC attack pattern"})
    if f["submitted_weekend"]: flags.append({"level":"low","msg":"Invoice submitted on a weekend"})
    if f["_gst_dev"] > 10: flags.append({"level":"medium","msg":f"GST deviates {f['_gst_dev']:.1f}% from expected 18%"})
    if not f["has_line_items"]: flags.append({"level":"medium","msg":"No itemized line items — lump-sum billing"})
    if f["new_vendor_high_amount"]: flags.append({"level":"high","msg":f"New vendor ({f['_vendor_age']} days) with high-value invoice"})
    if f["has_po"] and f["has_grn"] and f["po_amount_match"]: positives.append("3-way match passed: PO, GRN, and invoice align")
    if f["gst_valid"]: positives.append("GSTIN validated successfully")
    if f["_vendor_age"] > 365: positives.append(f"Established vendor ({f['_vendor_age']} days history)")
    if f["vendor_trust_score"] > 0.7: positives.append(f"High vendor trust score ({round(f['vendor_trust_score']*100)}%)")
    if f["_dup_score"] < 0.1: positives.append("No duplicate invoice detected")
    if f["is_msme"]:
        od = f["msme_days_overdue"]
        if od > 35: compliance.append({"type":"danger","msg":f"MSME 43B(h): {od} days — OVERDUE, legal risk"})
        elif od > 25: compliance.append({"type":"warning","msg":f"MSME 43B(h): {od} days — {45-od} days remaining"})
        else: compliance.append({"type":"ok","msg":f"MSME 43B(h): {od} days — within 45-day limit"})
    itc_blocked = not f["gst_valid"] or prob > 0.5
    if itc_blocked: compliance.append({"type":"danger","msg":f"GST ITC BLOCKED — {'invalid GSTIN' if not f['gst_valid'] else 'high fraud risk'}"})
    else: compliance.append({"type":"ok","msg":f"GST ITC of Rs.{f['_gst_amt']:,.0f} is claimable on approval"})
    return {"flags": flags, "positives": positives, "compliance": compliance}

@app.get("/api/health")
def health():
    return {"status":"ok","model_loaded":MODEL is not None,"ocr_available":True,"ocr_engine":"pdfplumber+regex"}

@app.post("/api/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    allowed = ["application/pdf","image/jpeg","image/png","image/webp","image/jpg"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    content = await file.read()
    try:
        if file.content_type == "application/pdf":
            raw_text = extract_text_pdf(content)
        else:
            raw_text = extract_text_image(content)
        parsed = parse_invoice_text(raw_text)
        return {"success": True, "data": parsed, "engine": "pdfplumber+regex"}
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
    inv = data.dict()
    features = extract_features(inv)
    vec = [features.get(k, 0) for k in FEATURE_COLS]
    if MODEL:
        rf_prob = predict_rf(MODEL["rf"], vec)
        gb_prob = predict_gb(MODEL["gb"], vec, MODEL["gb_lr"], MODEL["gb_init"])
    else:
        risk = 0.0
        if not features["has_po"]: risk += 0.25
        if not features["gst_valid"]: risk += 0.30
        if features["_vendor_age"] < 30: risk += 0.20
        if features["urgency_no_po"]: risk += 0.15
        if features["high_duplicate_score"]: risk += 0.10
        rf_prob = gb_prob = min(risk, 0.99)
    ensemble = (rf_prob + gb_prob) / 2.0
    verdict = "APPROVE" if ensemble < 0.25 else "REVIEW" if ensemble < 0.50 else "ESCALATE" if ensemble < 0.75 else "REJECT"
    explanation = generate_flags(features, ensemble)
    return {
        "success": True, "fraud_score": round(ensemble * 100, 1),
        "rf_score": round(rf_prob * 100, 1), "gb_score": round(gb_prob * 100, 1),
        "verdict": verdict, "flags": explanation["flags"],
        "positives": explanation["positives"], "compliance": explanation["compliance"],
        "feature_importance": FEATURE_IMPORTANCE,
        "top_features": {k: round(float(features.get(k, 0)), 4) for k in FEATURE_IMPORTANCE},
        "is_msme": bool(features["is_msme"]), "msme_days_overdue": features["msme_days_overdue"],
        "itc_blocked": not bool(features["gst_valid"]) or ensemble > 0.5,
        "gst_valid": bool(features["gst_valid"]),
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
