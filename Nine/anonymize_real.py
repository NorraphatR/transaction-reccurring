#!/usr/bin/env python3
"""
Anonymise a converted real statement folder for sharing: every person / shop name and account number becomes a code.

  counterparty_name        -> OWNER (the account owner) or PAYEE_001 ...   (same name -> same code)
  counterparty_account_id  -> ACC_001 ...                                  (same account -> same code)
  merchant_name / _id      -> MERCHANT_001 ...                             (mcc is kept)
  description              -> "<txn_type> <payee code>"                    (free text removed)
  customers.customer_name  -> OWNER
  truth_by_user.csv        -> payee keys rewritten with the same codes; names made generic
Amounts, dates, balances, txn types, channels and mcc are kept, so the pipeline gives the same groups and results.

Usage: python anonymize_real.py --src real_data --dst <out>/real_data --owner "<owner name as masked in the statement>"
"""
import argparse
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--truth-names", default="{}", help='JSON {series_id: generic name}')
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    t = pd.read_csv(f"{a.src}/transactions.csv", dtype=str, keep_default_na=False)
    codes = {}

    def code(kind, v):
        if v == "":
            return ""
        d = codes.setdefault(kind, {})
        if v not in d:
            d[v] = f"{kind}_{len(d) + 1:03d}"
        return d[v]
    t["counterparty_name"] = [("OWNER" if v == a.owner else code("PAYEE", v)) for v in t.counterparty_name]
    t["counterparty_account_id"] = [code("ACC", v) for v in t.counterparty_account_id]
    t["merchant_name"] = [code("MERCHANT", v) for v in t.merchant_name]
    t["merchant_id"] = [codes.get("MERCHANT", {}).get(v, "") if v else "" for v in t.merchant_name] if "merchant_id" in t else ""
    t["merchant_id"] = ""
    t["description"] = [f"{ty} {(m or n or ac)}".strip() for ty, m, n, ac in zip(t.txn_type, t.merchant_name, t.counterparty_name, t.counterparty_account_id)]
    for c in ("ref1", "ref2", "biller_id", "mandate_id"):
        if c in t:
            t[c] = [code(c.upper(), v) for v in t[c]]
    t.to_csv(f"{a.dst}/transactions.csv", index=False, encoding="utf-8")
    cu = pd.read_csv(f"{a.src}/customers.csv", dtype=str, keep_default_na=False)
    cu["customer_name"] = "OWNER"
    cu.to_csv(f"{a.dst}/customers.csv", index=False, encoding="utf-8")
    pd.read_csv(f"{a.src}/accounts.csv", dtype=str).to_csv(f"{a.dst}/accounts.csv", index=False, encoding="utf-8")
    tr = pd.read_csv(f"{a.src}/truth_by_user.csv", dtype=str, keep_default_na=False)
    mp = {"P": codes.get("ACC", {}), "M": codes.get("MERCHANT", {})}
    tr["payee_keys"] = [";".join(f"{k[0]}|{mp[k[0]][k[2:]]}" for k in keys.split(";")) for keys in tr.payee_keys]
    names = json.loads(a.truth_names)
    tr["name"] = [names.get(s, n) for s, n in zip(tr.series_id, tr.name)]
    tr["note"] = "labelled by the account owner"
    tr.to_csv(f"{a.dst}/truth_by_user.csv", index=False, encoding="utf-8")
    print(f"{a.dst}: {len(t)} rows | {len(codes.get('PAYEE', {}))} payees, {len(codes.get('ACC', {}))} accounts, {len(codes.get('MERCHANT', {}))} merchants coded")
    print(tr[["series_id", "name", "payee_keys"]].to_string(index=False))


if __name__ == "__main__":
    main()
