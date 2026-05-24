"""
create_sample_data.py

Generates three synthetic mortgage PDFs for one loan and saves them to
the loan's input folder under loans/<loan_id>/input/. Two ready-made
profiles are bundled:

  loan_001  borrower with a $9,200 vs $7,400 income mismatch, 47% DTI,
            90.5% LTV, 718 credit score — designed to trip several rules
            and produce STATUS: HOLD.
  loan_002  clean borrower (no income variance, 32% DTI, 75% LTV, 765
            score) — designed to produce STATUS: CLEAR.

Any other loan_id falls back to the loan_002 profile with the borrower
name replaced so you can spin up extra test loans without writing data.

Run:
    python create_sample_data.py --loan_id loan_001
    python create_sample_data.py --loan_id loan_002
"""

import argparse
import os
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import ensure_loan_dirs, loan_input_dir


# ---------- loan profiles ----------

LOAN_PROFILES = {
    "loan_001": {
        # Designed to trigger RULE-001 (HIGH variance), RULE-004 (DTI 43-50),
        # RULE-005 (LTV>90), RULE-007 (credit < 720) -> overall HOLD.
        "borrower": "John Martinez",
        "ssn_last4": "4821",
        "dob": "07/14/1986",
        "marital": "Married",
        "monthly_income_app": 9200,
        "monthly_income_paystub": 7400,
        "loan_amount": 380000,
        "purchase_price": 420000,
        "ltv": 90.5,
        "dti": 47,
        "credit_score": 718,
        "employer": "Meridian Logistics Inc",
        "employer_address": "880 Industrial Pkwy, Columbus OH 43219",
        "title": "Senior Operations Manager",
        "years_on_job": 6,
        "address_line1": "142 Birchwood Ave",
        "address_city_zip": "Columbus, OH 43215",
        "interest_rate": 6.875,
        "monthly_payment": 2496,
        "closing_costs": 11240,
        "loan_costs": 6180,
        "other_costs": 5060,
        "cash_to_close": 51240,
        "escrow_estimate": 612,
        "lender": "Axia Home Lending, NA",
        "pay_period": "April 1 - April 30, 2026",
        "pay_date": "May 1, 2026",
        "issue_date": "May 5, 2026",
        "closing_date": "May 15, 2026",
        "regular_pay": 6830.77,
        "regular_hours": 160,
        "regular_rate": 42.6923,
        "overtime_pay": 512.31,
        "overtime_hours": 8,
        "overtime_rate": 64.0385,
        "bonus_pay": 56.92,
        "ytd_gross": 29600,
        "file_id": "URLA-1003-2026-04-22",
        "paystub_id": "PAYSTUB-MLI-2026-04",
        "cd_id": "CD-2026-04821",
    },
    "loan_002": {
        # Clean loan: no variance, sub-43 DTI, sub-90 LTV, 765 score.
        # Should trigger no rules -> overall CLEAR.
        "borrower": "Sarah Patel",
        "ssn_last4": "9203",
        "dob": "03/22/1991",
        "marital": "Single",
        "monthly_income_app": 11500,
        "monthly_income_paystub": 11500,
        "loan_amount": 450000,
        "purchase_price": 600000,
        "ltv": 75.0,
        "dti": 32,
        "credit_score": 765,
        "employer": "Northgate Financial Services",
        "employer_address": "210 Pearl Street, Boulder CO 80302",
        "title": "Director, Risk Analytics",
        "years_on_job": 9,
        "address_line1": "88 Maplewood Drive",
        "address_city_zip": "Boulder, CO 80302",
        "interest_rate": 6.500,
        "monthly_payment": 2845,
        "closing_costs": 9800,
        "loan_costs": 5450,
        "other_costs": 4350,
        "cash_to_close": 159800,
        "escrow_estimate": 540,
        "lender": "Axia Home Lending, NA",
        "pay_period": "April 1 - April 30, 2026",
        "pay_date": "May 1, 2026",
        "issue_date": "May 5, 2026",
        "closing_date": "May 18, 2026",
        "regular_pay": 10615.38,
        "regular_hours": 160,
        "regular_rate": 66.3461,
        "overtime_pay": 0.0,
        "overtime_hours": 0,
        "overtime_rate": 0.0,
        "bonus_pay": 884.62,
        "ytd_gross": 46000,
        "file_id": "URLA-1003-2026-04-25",
        "paystub_id": "PAYSTUB-NGS-2026-04",
        "cd_id": "CD-2026-09203",
    },
}


def get_profile(loan_id: str) -> dict:
    """Return the data profile for a loan_id.

    Known ids (loan_001, loan_002) get bundled profiles. Anything else
    falls back to a copy of loan_002's profile with the borrower name
    swapped, so demos can spin up arbitrary loan_NNN ids without code edits.
    """
    if loan_id in LOAN_PROFILES:
        return dict(LOAN_PROFILES[loan_id])

    fallback = dict(LOAN_PROFILES["loan_002"])
    fallback["borrower"] = f"Test Borrower {loan_id}"
    fallback["file_id"] = f"URLA-1003-{loan_id}"
    fallback["paystub_id"] = f"PAYSTUB-{loan_id}"
    fallback["cd_id"] = f"CD-{loan_id}"
    return fallback


# ---------- low-level drawing helpers ----------

def _draw_header(c, title, subtitle=None):
    """Top banner with title and optional subtitle."""
    width, height = LETTER
    c.setFillColor(colors.HexColor("#0b3d91"))
    c.rect(0, height - 0.9 * inch, width, 0.9 * inch, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.6 * inch, height - 0.55 * inch, title)

    if subtitle:
        c.setFont("Helvetica", 10)
        c.drawString(0.6 * inch, height - 0.78 * inch, subtitle)

    c.setFillColor(colors.black)


def _draw_footer(c, doc_id):
    """Small footer with document id and a 'sample / not for use' notice."""
    width, _ = LETTER
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.grey)
    c.drawString(0.6 * inch, 0.5 * inch, f"Document ID: {doc_id}")
    c.drawRightString(
        width - 0.6 * inch,
        0.5 * inch,
        "SAMPLE — synthetic data for testing only",
    )
    c.setFillColor(colors.black)


def _section_title(c, x, y, text):
    """Bold blue section title with an underline."""
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.HexColor("#0b3d91"))
    c.drawString(x, y, text)
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.HexColor("#0b3d91"))
    c.setLineWidth(0.5)
    c.line(x, y - 2, x + 6.8 * inch, y - 2)


def _kv_row(c, x, y, label, value, label_w=2.2 * inch):
    """One label + value row with the label bold and value plain."""
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y, label)
    c.setFont("Helvetica", 10)
    c.drawString(x + label_w, y, str(value))


def _money(n):
    """Format a number as $X,XXX.XX for display."""
    return f"${n:,.2f}"


# ---------- document 1: 1003 ----------

def make_1003(path: str, p: dict) -> None:
    """Render a Fannie Mae 1003 / URLA PDF using the values in profile p."""
    c = canvas.Canvas(path, pagesize=LETTER)
    _draw_header(
        c,
        "Uniform Residential Loan Application",
        "Fannie Mae Form 1003 / Freddie Mac Form 65",
    )

    width, height = LETTER
    y = height - 1.3 * inch
    left = 0.6 * inch

    _section_title(c, left, y, "Section 1 — Borrower Information")
    y -= 0.28 * inch
    _kv_row(c, left, y, "Borrower Full Name:", p["borrower"]); y -= 0.22 * inch
    _kv_row(c, left, y, "SSN (last 4):", f"XXX-XX-{p['ssn_last4']}"); y -= 0.22 * inch
    _kv_row(c, left, y, "Date of Birth:", p["dob"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Marital Status:", p["marital"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Citizenship:", "U.S. Citizen")

    y -= 0.4 * inch
    _section_title(c, left, y, "Section 2 — Employment & Income")
    y -= 0.28 * inch
    _kv_row(c, left, y, "Current Employer:", p["employer"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Position / Title:", p["title"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Years on Job:", p["years_on_job"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Monthly Gross Income (claimed):", _money(p["monthly_income_app"])); y -= 0.22 * inch
    _kv_row(c, left, y, "Other Income:", "$0.00")

    y -= 0.4 * inch
    _section_title(c, left, y, "Section 3 — Property & Loan Information")
    y -= 0.28 * inch
    _kv_row(c, left, y, "Property Address:", p["address_line1"]); y -= 0.22 * inch
    _kv_row(c, left, y, "City / State / ZIP:", p["address_city_zip"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Property Type:", "Single Family Residence"); y -= 0.22 * inch
    _kv_row(c, left, y, "Occupancy:", "Primary Residence"); y -= 0.22 * inch
    _kv_row(c, left, y, "Loan Purpose:", "Purchase"); y -= 0.22 * inch
    _kv_row(c, left, y, "Purchase Price:", _money(p["purchase_price"])); y -= 0.22 * inch
    _kv_row(c, left, y, "Loan Amount Requested:", _money(p["loan_amount"])); y -= 0.22 * inch
    _kv_row(c, left, y, "Loan Term:", "30 years (Fixed)")

    y -= 0.4 * inch
    _section_title(c, left, y, "Section 4 — Qualifying Ratios")
    y -= 0.28 * inch
    _kv_row(c, left, y, "Loan-to-Value (LTV):", f"{p['ltv']}%"); y -= 0.22 * inch
    _kv_row(c, left, y, "Debt-to-Income (DTI):", f"{p['dti']}%"); y -= 0.22 * inch
    _kv_row(c, left, y, "Credit Score (mid):", p["credit_score"])

    y -= 0.5 * inch
    _section_title(c, left, y, "Borrower Acknowledgement")
    y -= 0.3 * inch
    c.setFont("Helvetica", 9)
    c.drawString(
        left,
        y,
        "I certify that the information provided in this application is true and complete to the best of my knowledge.",
    )
    y -= 0.6 * inch
    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Borrower Signature: ______/s/ {p['borrower']}______")
    c.drawString(left + 4.5 * inch, y, "Date: 04/22/2026")

    _draw_footer(c, p["file_id"])
    c.showPage()
    c.save()


# ---------- document 2: pay stub ----------

def make_paystub(path: str, p: dict) -> None:
    """Render an earnings-statement / pay stub PDF using the values in profile p."""
    c = canvas.Canvas(path, pagesize=LETTER)
    _draw_header(c, p["employer"], "Earnings Statement / Pay Stub")

    width, height = LETTER
    left = 0.6 * inch
    y = height - 1.3 * inch

    _section_title(c, left, y, "Employer & Employee")
    y -= 0.28 * inch
    _kv_row(c, left, y, "Employer:", p["employer"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Employer Address:", p["employer_address"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Employee:", p["borrower"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Employee ID:", "EMP-04417"); y -= 0.22 * inch
    _kv_row(c, left, y, "Pay Period:", p["pay_period"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Pay Date:", p["pay_date"])

    y -= 0.4 * inch
    _section_title(c, left, y, "Earnings")
    y -= 0.3 * inch

    col_x = [left, left + 2.6 * inch, left + 4.0 * inch, left + 5.4 * inch]
    c.setFont("Helvetica-Bold", 9)
    c.drawString(col_x[0], y, "Description")
    c.drawString(col_x[1], y, "Hours")
    c.drawString(col_x[2], y, "Rate")
    c.drawString(col_x[3], y, "Current")
    c.setStrokeColor(colors.black)
    c.line(left, y - 3, left + 6.8 * inch, y - 3)
    y -= 0.22 * inch

    c.setFont("Helvetica", 9)
    rows = [
        ("Regular Pay",       f"{p['regular_hours']:.2f}", f"${p['regular_rate']:.4f}/hr", _money(p["regular_pay"])),
        ("Overtime",          f"{p['overtime_hours']:.2f}", f"${p['overtime_rate']:.4f}/hr", _money(p["overtime_pay"])),
        ("Performance Bonus", "—",                          "—",                              _money(p["bonus_pay"])),
    ]
    for desc, hrs, rate, amt in rows:
        c.drawString(col_x[0], y, desc)
        c.drawString(col_x[1], y, hrs)
        c.drawString(col_x[2], y, rate)
        c.drawString(col_x[3], y, amt)
        y -= 0.2 * inch

    y -= 0.05 * inch
    c.line(left, y, left + 6.8 * inch, y)
    y -= 0.22 * inch
    c.setFont("Helvetica-Bold", 9)
    c.drawString(col_x[0], y, "Gross Earnings (this period)")
    c.drawString(col_x[3], y, _money(p["monthly_income_paystub"]))

    # Deductions block — values are computed proportionally from gross
    # so the totals look reasonable across different income levels.
    gross = float(p["monthly_income_paystub"])
    fed_tax = round(gross * 0.14, 2)
    state_tax = round(gross * 0.035, 2)
    ss = round(gross * 0.062, 2)
    medicare = round(gross * 0.0145, 2)
    retirement = round(gross * 0.05, 2)
    health = 185.00
    total_ded = round(fed_tax + state_tax + ss + medicare + retirement + health, 2)
    net = round(gross - total_ded, 2)

    y -= 0.4 * inch
    _section_title(c, left, y, "Deductions")
    y -= 0.28 * inch
    deductions = [
        ("Federal Income Tax",  fed_tax),
        ("State Income Tax",    state_tax),
        ("Social Security",     ss),
        ("Medicare",            medicare),
        ("401(k) Contribution", retirement),
        ("Health Insurance",    health),
    ]
    for label, amt in deductions:
        _kv_row(c, left, y, label + ":", _money(amt))
        y -= 0.2 * inch

    y -= 0.25 * inch
    _section_title(c, left, y, "Summary")
    y -= 0.28 * inch
    _kv_row(c, left, y, "Monthly Gross Income:", _money(gross)); y -= 0.22 * inch
    _kv_row(c, left, y, "Total Deductions:", _money(total_ded)); y -= 0.22 * inch
    _kv_row(c, left, y, "Net Pay (this period):", _money(net)); y -= 0.22 * inch
    _kv_row(c, left, y, "YTD Gross (Jan-Apr 2026):", _money(p["ytd_gross"]))

    _draw_footer(c, p["paystub_id"])
    c.showPage()
    c.save()


# ---------- document 3: closing disclosure ----------

def make_closing_disclosure(path: str, p: dict) -> None:
    """Render a TRID Closing Disclosure PDF using the values in profile p."""
    c = canvas.Canvas(path, pagesize=LETTER)
    _draw_header(
        c,
        "Closing Disclosure",
        "This form is a statement of final loan terms and closing costs.",
    )

    width, height = LETTER
    left = 0.6 * inch
    y = height - 1.3 * inch

    _section_title(c, left, y, "Transaction Information")
    y -= 0.28 * inch
    _kv_row(c, left, y, "Date Issued:", p["issue_date"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Closing Date:", p["closing_date"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Disbursement Date:", p["closing_date"]); y -= 0.22 * inch
    _kv_row(c, left, y, "File #:", p["cd_id"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Borrower:", p["borrower"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Lender:", p["lender"]); y -= 0.22 * inch
    _kv_row(c, left, y, "Property:", f"{p['address_line1']}, {p['address_city_zip']}"); y -= 0.22 * inch
    _kv_row(c, left, y, "Sale Price:", _money(p["purchase_price"]))

    y -= 0.4 * inch
    _section_title(c, left, y, "Loan Terms")
    y -= 0.3 * inch

    col_x = [left, left + 3.4 * inch, left + 5.4 * inch]
    c.setFont("Helvetica-Bold", 9)
    c.drawString(col_x[0], y, "Term")
    c.drawString(col_x[1], y, "Value")
    c.drawString(col_x[2], y, "Can this amount increase?")
    c.line(left, y - 3, left + 6.8 * inch, y - 3)
    y -= 0.22 * inch

    c.setFont("Helvetica", 9)
    terms = [
        ("Loan Amount",                  _money(p["loan_amount"]),              "NO"),
        ("Interest Rate",                f"{p['interest_rate']}%",              "NO"),
        ("Monthly Principal & Interest", _money(p["monthly_payment"]),          "NO"),
        ("Loan Term",                    "30 years",                            "—"),
        ("Loan Type",                    "Conventional Fixed",                  "—"),
        ("Prepayment Penalty",           "None",                                "—"),
        ("Balloon Payment",              "None",                                "—"),
    ]
    for label, val, change in terms:
        c.drawString(col_x[0], y, label)
        c.drawString(col_x[1], y, val)
        c.drawString(col_x[2], y, change)
        y -= 0.22 * inch

    total_pmt = p["monthly_payment"] + p["escrow_estimate"]
    y -= 0.2 * inch
    _section_title(c, left, y, "Projected Payments")
    y -= 0.28 * inch
    _kv_row(c, left, y, "Principal & Interest:", _money(p["monthly_payment"])); y -= 0.22 * inch
    _kv_row(c, left, y, "Estimated Escrow (Tax + Ins):", _money(p["escrow_estimate"])); y -= 0.22 * inch
    _kv_row(c, left, y, "Estimated Total Monthly Payment:", _money(total_pmt))

    y -= 0.4 * inch
    _section_title(c, left, y, "Costs at Closing")
    y -= 0.28 * inch
    _kv_row(c, left, y, "Closing Costs:", _money(p["closing_costs"])); y -= 0.22 * inch
    _kv_row(c, left, y, "  Loan Costs:", _money(p["loan_costs"])); y -= 0.22 * inch
    _kv_row(c, left, y, "  Other Costs:", _money(p["other_costs"])); y -= 0.22 * inch
    _kv_row(c, left, y, "Cash to Close:", _money(p["cash_to_close"]))

    y -= 0.45 * inch
    _section_title(c, left, y, "Confirm Receipt")
    y -= 0.45 * inch
    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Borrower: ______/s/ {p['borrower']}______")
    c.drawString(left + 4.5 * inch, y, f"Date: {p['closing_date']}")

    _draw_footer(c, p["cd_id"])
    c.showPage()
    c.save()


# ---------- entry point ----------

def main(loan_id: str = "loan_001") -> None:
    """Generate all three PDFs for one loan into loans/<loan_id>/input/."""
    profile = get_profile(loan_id)
    ensure_loan_dirs(loan_id)
    out_dir = loan_input_dir(loan_id)

    targets = [
        (f"{loan_id}_1003.pdf",                make_1003),
        (f"{loan_id}_paystub.pdf",             make_paystub),
        (f"{loan_id}_closing_disclosure.pdf",  make_closing_disclosure),
    ]

    created = []
    for filename, builder in targets:
        path = out_dir / filename
        builder(str(path), profile)
        created.append((filename, path, os.path.getsize(path) / 1024))

    print(f"Generated synthetic mortgage documents for {loan_id}:")
    print("-" * 60)
    for name, _, size_kb in created:
        print(f"  {name:<40}  {size_kb:6.1f} KB")
    print("-" * 60)
    print(f"Output folder: {out_dir}")
    print()
    print(f"  Borrower:                    {profile['borrower']}")
    print(f"  Monthly income (1003):       ${profile['monthly_income_app']:,}")
    print(f"  Monthly income (pay stub):   ${profile['monthly_income_paystub']:,}")
    print(f"  Loan amount:                 ${profile['loan_amount']:,} @ {profile['interest_rate']}%")
    print(f"  LTV / DTI / Credit:          {profile['ltv']}% / {profile['dti']}% / {profile['credit_score']}")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic mortgage PDFs for one loan.")
    p.add_argument("--loan_id", default="loan_001")
    args = p.parse_args()
    main(loan_id=args.loan_id)


if __name__ == "__main__":
    _cli()
