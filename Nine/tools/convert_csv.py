#!/usr/bin/env python3
"""
Convert an already-extracted KBank statement CSV
(columns: วันที่,เวลา,รายการ,ถอนเงิน,ฝากเงิน,ยอดคงเหลือ,ช่องทาง,รายละเอียด,ไฟล์ต้นฉบับ,หน้า)
into the SafeBelt format: 23 spec columns + `source`, plus accounts.csv / customers.csv.

Usage:
  python3 convert_csv.py --csv transactions_combined.csv --out data_real_c2 \
      --customer-id C00002 [--owner-name "น.ส. xxx yyy++"] [--payroll-cp "X8978"]
"""
import argparse, os, re, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_kbank import parse_counterparty, clean_merchant, mcc_for  # same rules as the PDF converter

COLS = ["txn_id", "customer_id", "account_id", "posted_datetime", "amount", "direction", "txn_type", "channel",
        "balance_after", "description", "counterparty_account_id", "counterparty_bank_code", "counterparty_name",
        "promptpay_proxy_type", "biller_id", "ref1", "ref2", "mandate_id", "mcc", "merchant_id", "merchant_name",
        "original_currency", "original_amount", "source"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data_real_c2")
    ap.add_argument("--customer-id", default="C00002")
    ap.add_argument("--owner-name", default="", help="masked owner name as statements show it, for own-transfer detection")
    ap.add_argument("--payroll-cp", default="", help="counterparty account (e.g. X8978) whose monthly credits are the payday")
    ap.add_argument("--payroll-min", type=float, default=3000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    acct = f"A_{a.customer_id}"
    rep = []

    d = pd.read_csv(a.csv, dtype=str, encoding="utf-8-sig").fillna("")
    d["_seq"] = range(len(d))
    d["_dt"] = pd.to_datetime(d["วันที่"] + " " + d["เวลา"], format="%Y-%m-%d %H:%M")
    d["_out"] = pd.to_numeric(d["ถอนเงิน"].str.replace(",", ""), errors="coerce").fillna(0)
    d["_in"] = pd.to_numeric(d["ฝากเงิน"].str.replace(",", ""), errors="coerce").fillna(0)
    d["_bal"] = pd.to_numeric(d["ยอดคงเหลือ"].str.replace(",", ""))
    assert ((d["_out"] > 0) ^ (d["_in"] > 0)).all(), "each row needs exactly one of withdraw / deposit"

    # balance chain check in original statement order
    opening = round(d["_bal"].iloc[0] + d["_out"].iloc[0] - d["_in"].iloc[0], 2)
    bal, breaks = opening, 0
    for _, r in d.iterrows():
        bal = round(bal - r["_out"] + r["_in"], 2)
        if abs(bal - r["_bal"]) > 0.005:
            breaks += 1; rep.append(f"  chain break {r['_dt']} expected {bal} got {r['_bal']}"); bal = r["_bal"]
    rep.append(f"input rows {len(d)}, files {d['ไฟล์ต้นฉบับ'].unique().tolist()}, opening {opening:,.2f}, "
               f"closing {d['_bal'].iloc[-1]:,.2f}, chain breaks {breaks}")
    d = d.sort_values(["_dt", "_seq"], kind="stable")

    # optional payday rule: one credit per month from the given counterparty
    payroll_idx = set()
    if a.payroll_cp:
        c = d[(d["_in"] >= a.payroll_min) & d["รายละเอียด"].str.contains(rf"จาก (?:[A-Z]+ )?{a.payroll_cp} ")]
        for _, g in c.groupby(c["_dt"].dt.to_period("M")):
            payroll_idx.add(g["_in"].idxmax())
    rep.append(f"payroll_in rows: {len(payroll_idx)}" + ("" if a.payroll_cp else " (no --payroll-cp given -> freelancer fallback)"))

    out, bal = [], opening
    for i, (idx, r) in enumerate(d.iterrows(), 1):
        typ, ch, det = r["รายการ"], r["ช่องทาง"], r["รายละเอียด"]
        cr = r["_in"] > 0
        amt = r["_in"] if cr else r["_out"]
        cp_acct = cp_bank = cp_name = proxy = merchant = merchant_id = mcc = ""
        channel = "KPLUS"
        if typ == "ชำระเงิน":
            txn_type = "qr_payment"; merchant = clean_merchant(det); mcc = mcc_for(merchant)
        elif typ == "ชำระด้วยบัตรเดบิต":
            txn_type, channel = "card_purchase", "CARD"
            merchant_id = re.sub(r"^รหัสอ้างอิง\s*", "", det)   # EDC ref; not a stable merchant key
        elif typ == "โอนเงิน":
            cp_acct, cp_bank, cp_name, proxy = parse_counterparty(det)
            txn_type = "promptpay_out" if proxy else "transfer_out"
        elif typ == "รับโอนเงิน":
            cp_acct, cp_bank, cp_name, proxy = parse_counterparty(det)
            txn_type = "promptpay_in" if proxy else "transfer_in"
            if idx in payroll_idx:
                txn_type, channel = "payroll_in", "AUTO"
        elif typ == "รับโอนเงินผ่าน QR":
            txn_type = "promptpay_in"
        elif typ == "รับโอนเงินอัตโนมัติ":          # debit-card refund (Direct Credit)
            txn_type, channel = "refund_in", "AUTO"
        elif typ == "รายการแก้ไข":                  # card reversal
            txn_type, channel = "refund_in", "CARD"
            merchant_id = re.sub(r"^รหัสอ้างอิง\s*", "", det)
        elif typ == "ฝากเงินสด":
            txn_type, channel = "transfer_in", "BRANCH"
        elif typ == "รับดอกเบี้ยเงินฝาก":
            txn_type, channel = "transfer_in", "AUTO"
        elif typ == "ถอนเงินสด":
            txn_type, channel = "atm_withdrawal", "ATM"
        else:
            raise ValueError(f"unknown รายการ: {typ}")
        bal = round(bal + (amt if cr else -amt), 2)
        out.append({
            "txn_id": f"T{i:09d}", "customer_id": a.customer_id, "account_id": acct,
            "posted_datetime": r["_dt"].strftime("%Y-%m-%d %H:%M:00"), "amount": f"{amt:.2f}",
            "direction": "CR" if cr else "DR", "txn_type": txn_type, "channel": channel,
            "balance_after": f"{bal:.2f}", "description": f"{typ} {det}".strip(),
            "counterparty_account_id": cp_acct, "counterparty_bank_code": cp_bank, "counterparty_name": cp_name,
            "promptpay_proxy_type": proxy, "biller_id": "", "ref1": "", "ref2": "", "mandate_id": "", "mcc": mcc,
            "merchant_id": merchant_id, "merchant_name": merchant,
            "original_currency": "THB" if (merchant or merchant_id) and not cr else "",
            "original_amount": f"{amt:.2f}" if (merchant or merchant_id) and not cr else "",
            "source": "KPLUS",
        })
    df = pd.DataFrame(out, columns=COLS)

    rep.append(f"final balance computed {bal:,.2f} vs statement {d.sort_values('_seq')['_bal'].iloc[-1]:,.2f}")
    rep.append(f"negative balance rows: {(df['balance_after'].astype(float) < 0).sum()}")
    rep.append("rows by txn_type:\n" + df["txn_type"].value_counts().to_string())
    rep.append(f"date range: {df['posted_datetime'].min()} .. {df['posted_datetime'].max()}")
    if not a.owner_name:
        rep.append("WARNING: no --owner-name -> customer_name empty, own-account transfers cannot be detected")

    df.to_csv(os.path.join(a.out, "transactions.csv"), index=False, encoding="utf-8")
    first = df["posted_datetime"].min()[:10]
    pd.DataFrame([{"customer_id": a.customer_id, "account_id": acct, "account_type": "savings", "open_date": first}]
                 ).to_csv(os.path.join(a.out, "accounts.csv"), index=False)
    pd.DataFrame([{"customer_id": a.customer_id, "customer_name": a.owner_name, "first_account_open_date": first}]
                 ).to_csv(os.path.join(a.out, "customers.csv"), index=False)
    with open(os.path.join(a.out, "conversion_report.txt"), "w") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
