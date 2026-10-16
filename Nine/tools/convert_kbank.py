#!/usr/bin/env python3
"""
Convert K PLUS (main, X6042) + MAKE (X1472) KBank statement PDFs into SafeBelt input files,
merged into ONE combined main account.

Usage:
  python3 convert_kbank.py --main <main_pdf>... --make <make_pdf>... --out data_real/

Outputs (data_real/):
  transactions.csv   23 columns in DATA_SPEC order, one account A_MAIN
  accounts.csv       one row
  customers.csv      one row
  conversion_report.txt
"""
import argparse, re, os, sys
from datetime import datetime, timedelta
import pdfplumber
import pandas as pd

OWN_ACCTS = {"main": "X6042", "make": "X1472"}
OTHER_OWN = {"X3663"}            # third own account (not in data) -> kept as own transfers
CUSTOMER_ID = "C00001"
ACCOUNT_ID = "A_MAIN"
OWN_MASKED = "<OWNER NAME AS MASKED IN THE STATEMENT>"  # same masking the statements use -> own-name match works
PAYROLL_CP = "KTB X3059"         # regular monthly support transfer treated as payday
PAYROLL_MIN = 10000.0
SRC_LABEL = {"main": "KPLUS", "make": "MAKE", "adj": "MAKE_GAP_ADJ"}
BANK_CODES = {"KBANK": "004", "BBL": "002", "KTB": "006", "TTB": "011", "SCB": "014", "BAY": "025"}

TXN_RE = re.compile(
    r"^(\d{2}-\d{2}-\d{2}) (\d{2}:\d{2}) (\S+(?: QR)?) ([\d,]+\.\d{2}) ([\d,]+\.\d{2}) "
    r"(K PLUS|MAKE by KBank|EDC/K SHOP/MYQR|ATM RCM|K BIZ|Internet/Mobile \w+|โอนเข้า/หักบัญชีอัตโนมัติ|ต่างธนาคาร|[^ ]+)\s*(.*)$")
CARRY_RE = re.compile(r"^(\d{2}-\d{2}-\d{2}) ยอดยกมา ([\d,]+\.\d{2})")
JUNK_RE = re.compile(r"^(\)|FDPBK|ออกโดย|สอบถามข้อมูล|หน้าที่ \(PAGE|ที่ DD\.|ชื่อบัญชี|\d{3} ถ\.|296 |รอบระหว่าง|"
                     r"สาขาเจ้าของ|ยอดยกไป|\d+/\d+\(|รวมถอน|รวมฝาก|เวลา/|วันที่ รายการ|วันที่มีผล)")

IN_TYPES = {"รับโอนเงิน", "รับโอนเงินผ่าน QR", "รับเงินคืน", "รับดอกเบี้ยเงินฝาก"}
OUT_TYPES = {"ชำระเงิน", "โอนเงิน", "ถอนเงินสด"}


def num(s):
    return float(s.replace(",", ""))


def parse_pdf(path, src):
    rows, carries, period = [], [], None
    with pdfplumber.open(path) as pdf:
        lines = []
        for pg in pdf.pages:
            lines += (pg.extract_text() or "").split("\n")
    in_body, cur = False, None
    for ln in lines:
        ln = ln.strip()
        m = re.search(r"รอบระหว่างวันที่ (\S+) - (\S+)", ln)
        if m and period is None:
            period = m.groups()
        if ln.startswith("วันที่มีผล"):
            in_body = True; continue
        if ln.startswith("หน้าที่ (PAGE") or ln.startswith(")"):
            in_body = False; cur = None; continue
        if not in_body:
            continue
        c = CARRY_RE.match(ln)
        if c:
            carries.append((c.group(1), num(c.group(2)))); cur = None; continue
        t = TXN_RE.match(ln)
        if t:
            d, tm, typ, amt, bal, ch, det = t.groups()
            cur = dict(src=src, date=d, time=tm, type=typ, amount=num(amt), balance=num(bal),
                       channel_raw=ch, details=det.strip())
            rows.append(cur); continue
        if re.match(r"^\d{2}-\d{2}-\d{2} ", ln):
            raise ValueError(f"unparsed txn line in {path}: {ln}")
        if cur is not None and not JUNK_RE.match(ln):
            cur["details"] = (cur["details"] + " " + ln).strip()
    return rows, carries, period


def check_chain(rows, opening, label, report):
    bal, bad = opening, 0
    for r in rows:
        if r["type"] in IN_TYPES:
            r["direction"] = "CR"; exp = round(bal + r["amount"], 2)
        elif r["type"] in OUT_TYPES:
            r["direction"] = "DR"; exp = round(bal - r["amount"], 2)
        else:
            raise ValueError(f"unknown type {r['type']}")
        if abs(exp - r["balance"]) > 0.005:
            bad += 1
            report.append(f"  [{label}] chain break at {r['date']} {r['time']} {r['type']} {r['amount']}: expected {exp} got {r['balance']}")
        bal = r["balance"]
    report.append(f"{label}: {len(rows)} rows, opening {opening:,.2f}, closing {bal:,.2f}, chain breaks {bad}")
    return bal


def parse_counterparty(det):
    """'โอนไป [พร้อมเพย์ ][BANK ]X1234 NAME' or 'จาก ...' -> (acct_id, bank_code, name, proxy)"""
    m = re.match(r"^(?:โอนไป|จาก) (พร้อมเพย์ )?(?:([A-Z]{2,5}) )?(X[0-9A-Za-z]{4}) (.*)$", det)
    if not m:
        return "", "", "", ""
    pp, bank, acc, name = m.groups()
    bank = bank or ("" if pp else "KBANK")
    acct = (f"{bank} {acc}" if bank and bank != "KBANK" else acc)
    if pp:
        acct = f"PP {acc}"
    return acct, BANK_CODES.get(bank, ""), name.strip(), ("mobile" if pp else "")


MCC_RULES = [  # (regex on merchant, mcc) - best-effort, used only by the questions step
    (r"TrueMoney|LINE Pay|LinePay|ShopeePay|Rabbit", "6540"),
    (r"Shopee|Lazada|TikTok", "5399"),
    (r"7-?ELEVEN|เซเว่น|ซีพี ออลล์|Lawson|ลอว์สัน|แฟมิลี่มาร์ท|FamilyMart", "5499"),
    (r"TOPS|Lotus|โลตัส|บิ๊กซี|Big C|Gourmet|MaxValu|แม็กซ์แวลู", "5411"),
    (r"คาเฟ่|Cafe|Café|Coffee|กาแฟ|Starbucks|อเมซอน|ชาตรามือ|ชานม|Tea|เครื่องดื่ม|Bakery|เบเกอรี่", "5814"),
    (r"ไลน์แมน|LINE MAN|Grab|Food|ก๋วยเตี๋ยว|ข้าว|ราเมง|ร้าน|McDonald|แมคโดนัลด์|KFC|ซูชิ|หมูกระทะ|ลูกชิ้น|อาหาร|คัตซึ|Restaurant", "5812"),
    (r"บางจาก|ปตท|PTT|Shell|เชลล์|Caltex", "5541"),
    (r"เมเจอร์|Major|SF Cinema|เอสเอฟ", "7832"),
    (r"BEAUTRIUM|Watsons|วัตสัน|Boots|EVEANDBOY", "5977"),
    (r"ร้านยา|Pharmacy|ฟาร์มา", "5912"),
    (r"โรงพยาบาล|คลินิก|Hospital|Clinic", "8062"),
    (r"BTS|MRT|รถไฟฟ้า|Bolt", "4111"),
]


def clean_merchant(det):
    m = re.match(r"^เพื่อชำระ Ref \S+ (.*)$", det)
    if not m:
        return ""
    s = m.group(1)
    s = re.sub(r"\s*\(\s*ชื่อ\s*บัญชี.*$", "", s)  # drop the receiver account name (often truncated)
    s = re.sub(r"^([^\d\s-]{3,}?)\s*-\s*\S.*$", r"\1", s)  # 'ชาตรามือ-สีลม...' -> 'ชาตรามือ' (branches -> one payee)
    s = re.sub(r"\s+สาขา.*$", "", s)                 # drop branch so branches share one payee key
    s = re.sub(r"\s+", " ", s).strip(" (")
    return s


def mcc_for(merchant):
    for rx, code in MCC_RULES:
        if re.search(rx, merchant, re.I):
            return code
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", nargs="+", required=True)
    ap.add_argument("--make", nargs="+", required=True)
    ap.add_argument("--out", default="data_real")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = []

    acc_rows = {}
    for key, files in (("main", a.main), ("make", a.make)):
        parsed = []
        for f in files:
            rows, carries, period = parse_pdf(f, key)
            parsed.append((datetime.strptime(period[0], "%d/%m/%Y"), period, rows, carries, f))
        parsed.sort(key=lambda x: x[0])
        all_rows, segments = [], []
        for start, period, rows, carries, f in parsed:
            opening = carries[0][1]
            report.append(f"{key} statement {os.path.basename(f)} period {period[0]}-{period[1]}")
            closing = check_chain(rows, opening, key, report)
            segments.append((start, datetime.strptime(period[1], "%d/%m/%Y"), opening, closing))
            all_rows += rows
        acc_rows[key] = (all_rows, segments)

    # ---- timestamps
    for key in acc_rows:
        for i, r in enumerate(acc_rows[key][0]):
            r["dt"] = datetime.strptime(f"{r['date']} {r['time']}", "%d-%m-%y %H:%M")
            r["seq"] = i

    main_rows, main_segs = acc_rows["main"]
    make_rows, make_segs = acc_rows["make"]

    # ---- internal transfers between X6042 and X1472: match pairs and drop both legs
    def is_internal(r, other):
        return re.match(rf"^(?:โอนไป|จาก) {other} ", r["details"]) is not None

    m_int = [r for r in main_rows if is_internal(r, OWN_ACCTS["make"])]
    k_int = [r for r in make_rows if is_internal(r, OWN_ACCTS["main"])]
    used, pairs = set(), 0
    for r in m_int:
        best = None
        for k in k_int:
            if id(k) in used or k["direction"] == r["direction"] or abs(k["amount"] - r["amount"]) > 0.005:
                continue
            gap = abs((k["dt"] - r["dt"]).total_seconds())
            if gap <= 30 * 60 and (best is None or gap < best[0]):
                best = (gap, k)
        if best:
            used.add(id(best[1])); r["drop"] = best[1]["drop"] = "pair"; pairs += 1
    orphans = [r for r in m_int + k_int if "drop" not in r]
    for r in orphans:
        r["drop"] = "orphan"
    report.append(f"internal pairs removed: {pairs} (= {2*pairs} rows); one-sided internal legs removed: {len(orphans)}")
    for r in orphans:
        report.append(f"  orphan [{r['src']}] {r['dt']:%Y-%m-%d %H:%M} {r['direction']} {r['amount']:,.2f} {r['details'][:40]}")

    # ---- combined opening balance: main opening + MAKE opening back-dated to main start
    #      (MAKE's own external activity before its first statement is unknown -> assumed none)
    main_start = main_segs[0][0]
    make_first_start, make_first_open = make_segs[0][0], make_segs[0][2]
    pre_make_internal = sum((r["amount"] if r["direction"] == "DR" else -r["amount"])
                            for r in main_rows if r.get("drop") == "orphan" and r["dt"] < make_first_start)
    make_open_est = round(make_first_open - pre_make_internal, 2)
    opening_combined = round(main_segs[0][2] + make_open_est, 2)
    report.append(f"combined opening {main_start:%Y-%m-%d}: main {main_segs[0][2]:,.2f} + MAKE est {make_open_est:,.2f} "
                  f"(MAKE {make_first_open:,.2f} on {make_first_start:%Y-%m-%d} minus {pre_make_internal:,.2f} moved in from main before it)")

    # ---- MAKE statement gaps -> one balancing row per gap
    adjustments = []
    for (s1, e1, o1, c1), (s2, e2, o2, c2) in zip(make_segs, make_segs[1:]):
        if (s2 - e1).days > 1:
            gap_internal = sum((r["amount"] if r["direction"] == "DR" else -r["amount"])
                               for r in main_rows if r.get("drop") == "orphan" and e1 < r["dt"] < s2)
            # MAKE moved c1 -> o2 during the gap; part of it is internal money already removed from main
            ext = round(o2 - c1 - gap_internal, 2)
            report.append(f"MAKE gap {e1:%Y-%m-%d}..{s2:%Y-%m-%d}: MAKE {c1:,.2f} -> {o2:,.2f}; internal from main {gap_internal:,.2f}; "
                          f"unexplained external net {ext:+,.2f} -> balancing row on {s2 - timedelta(minutes=1):%Y-%m-%d %H:%M}")
            if abs(ext) >= 0.01:
                adjustments.append(dict(src="adj", dt=s2 - timedelta(minutes=1), seq=0, type="ADJ",
                                        amount=abs(ext), direction="CR" if ext > 0 else "DR",
                                        channel_raw="ADJ", details=f"BALANCE ADJUSTMENT missing MAKE statement {e1:%d-%m-%y}..{s2 - timedelta(days=1):%d-%m-%y}"))

    keep = [r for r in main_rows + make_rows if "drop" not in r] + adjustments
    order = {"main": 0, "make": 1, "adj": 2}
    keep.sort(key=lambda r: (r["dt"], order[r["src"]], r["seq"]))

    # ---- payroll rule: monthly large credit from PAYROLL_CP, one per month (closest to day 27)
    cands = [r for r in keep if r["direction"] == "CR" and r["details"].startswith(f"จาก {PAYROLL_CP} ")
             and r["amount"] >= PAYROLL_MIN and r["dt"].day >= 24]
    bymonth = {}
    for r in cands:
        k = (r["dt"].year, r["dt"].month)
        if k not in bymonth or abs(r["dt"].day - 27) < abs(bymonth[k]["dt"].day - 27):
            bymonth[k] = r
    for r in bymonth.values():
        r["payroll"] = True
    report.append(f"payroll_in rows (from {PAYROLL_CP}, >= {PAYROLL_MIN:,.0f}, day >= 24, one per month): {len(bymonth)}")
    for r in sorted(bymonth.values(), key=lambda r: r["dt"]):
        report.append(f"  {r['dt']:%Y-%m-%d %H:%M} {r['amount']:,.2f}")

    # ---- build output rows
    out, bal = [], opening_combined
    for i, r in enumerate(keep, 1):
        det, typ, ch = r["details"], r["type"], r["channel_raw"]
        cp_acct = cp_bank = cp_name = proxy = merchant = mcc = ""
        if typ == "ADJ":
            txn_type, channel = "transfer_in" if r["direction"] == "CR" else "transfer_out", "AUTO"
            cp_name = OWN_MASKED; cp_acct = "X1472"; cp_bank = "004"
        elif typ == "ชำระเงิน":
            txn_type, channel = "qr_payment", "KPLUS"
            merchant = clean_merchant(det); mcc = mcc_for(merchant)
            if merchant.replace(" ", "") == "<OWNER FULL NAME WITHOUT SPACES>":   # QR paid to own name -> own transfer
                cp_name, mcc = OWN_MASKED, ""
        elif typ == "โอนเงิน":
            cp_acct, cp_bank, cp_name, proxy = parse_counterparty(det)
            txn_type, channel = ("promptpay_out" if proxy else "transfer_out"), "KPLUS"
        elif typ == "ถอนเงินสด":
            txn_type, channel = "atm_withdrawal", "ATM"
        elif typ == "รับโอนเงิน":
            cp_acct, cp_bank, cp_name, proxy = parse_counterparty(det)
            if r.get("payroll"):
                txn_type, channel = "payroll_in", "AUTO"
            else:
                txn_type, channel = ("promptpay_in" if proxy else "transfer_in"), "KPLUS"
        elif typ == "รับโอนเงินผ่าน QR":
            txn_type, channel = "promptpay_in", "KPLUS"
        elif typ == "รับเงินคืน":
            txn_type, channel = "refund_in", "KPLUS"
        elif typ == "รับดอกเบี้ยเงินฝาก":
            txn_type, channel = "transfer_in", "AUTO"
            det = "INTEREST " + det
        else:
            raise ValueError(typ)
        if r["src"] != "adj" and ch == "ATM RCM":
            channel = "ATM"
        bal = round(bal + (r["amount"] if r["direction"] == "CR" else -r["amount"]), 2)
        out.append({
            "txn_id": f"T{i:09d}", "customer_id": CUSTOMER_ID, "account_id": ACCOUNT_ID,
            "posted_datetime": r["dt"].strftime("%Y-%m-%d %H:%M:00"),
            "amount": f"{r['amount']:.2f}", "direction": r["direction"], "txn_type": txn_type,
            "channel": channel, "balance_after": f"{bal:.2f}",
            "description": f"{typ if typ != 'ADJ' else ''} {det}".strip(),
            "counterparty_account_id": cp_acct, "counterparty_bank_code": cp_bank,
            "counterparty_name": cp_name, "promptpay_proxy_type": proxy,
            "biller_id": "", "ref1": "", "ref2": "", "mandate_id": "", "mcc": mcc,
            "merchant_id": "", "merchant_name": merchant, "original_currency": "THB" if merchant else "",
            "original_amount": f"{r['amount']:.2f}" if merchant else "",
            "source": SRC_LABEL[r["src"]],
        })
    df = pd.DataFrame(out)

    # ---- true combined balance check (main + MAKE known balances) at statement ends
    final_true = main_segs[-1][3] + make_segs[-1][3]
    report.append(f"final combined balance: computed {bal:,.2f} vs main {main_segs[-1][3]:,.2f} + MAKE {make_segs[-1][3]:,.2f} = {final_true:,.2f}"
                  f" -> diff {bal - final_true:+.2f}")
    neg = df[df["balance_after"].astype(float) < 0]
    report.append(f"negative balance rows: {len(neg)}")
    report.append(f"rows out: {len(df)}; by txn_type:\n" + df["txn_type"].value_counts().to_string())
    report.append(f"date range: {df['posted_datetime'].min()} .. {df['posted_datetime'].max()}")

    report.append("rows by source:\n" + df["source"].value_counts().to_string())
    df.to_csv(os.path.join(a.out, "transactions.csv"), index=False, encoding="utf-8")
    # KPLUS<->MAKE moves: not counted (same account), kept here for reference
    internal = sorted([r for r in main_rows + make_rows if "drop" in r], key=lambda r: (r["dt"], order[r["src"]]))
    pd.DataFrame([{"posted_datetime": r["dt"].strftime("%Y-%m-%d %H:%M:00"), "source": SRC_LABEL[r["src"]],
                   "direction": r["direction"], "amount": f"{r['amount']:.2f}",
                   "match": "paired" if r["drop"] == "pair" else "other_leg_missing",
                   "description": f"{r['type']} {r['details']}"} for r in internal]
                 ).to_csv(os.path.join(a.out, "internal_transfers_excluded.csv"), index=False, encoding="utf-8")
    first = df["posted_datetime"].min()[:10]
    pd.DataFrame([{"customer_id": CUSTOMER_ID, "account_id": ACCOUNT_ID, "account_type": "savings_combined",
                   "open_date": first}]).to_csv(os.path.join(a.out, "accounts.csv"), index=False)
    pd.DataFrame([{"customer_id": CUSTOMER_ID, "customer_name": OWN_MASKED,
                   "first_account_open_date": first}]).to_csv(os.path.join(a.out, "customers.csv"), index=False)
    with open(os.path.join(a.out, "conversion_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
