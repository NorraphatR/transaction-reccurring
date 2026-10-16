#!/usr/bin/env python3
"""Validate a v3 dataset: schema, leakage, balances, labels, split/censoring, cycle-type mix.

Usage: python check_v3.py data_v3
"""
import os
import sys

import numpy as np
import pandas as pd

from generate_v2 import TX_COLS, SPLIT_TRAIN_END, SPLIT_CALIB_END

CODES = {"counterparty_bank_code": str, "ref1": str, "ref2": str, "mcc": str}


def load(d, name, **kw):
    f = f"{d}/{name}.csv.gz" if os.path.exists(f"{d}/{name}.csv.gz") else f"{d}/{name}.csv"
    return pd.read_csv(f, encoding="utf-8-sig", **kw)


d = sys.argv[1] if len(sys.argv) > 1 else "data_v3"
tx = load(d, "transactions", dtype=CODES)
lb = load(d, "txn_labels")
st = load(d, "series_truth")
cu, ac, ut = load(d, "customers"), load(d, "accounts"), load(d, "users_truth")
res = []


def check(name, ok, info=""):
    res.append((name, "PASS" if ok else "FAIL", info))


# ---- schema & leakage
check("transactions: exactly the 23 standard fields", list(tx.columns) == TX_COLS, f"{len(tx.columns)} cols")
leak = [c for c in tx.columns if c.startswith(("gt_", "truth_")) or c in ("memo", "is_own_account", "is_payroll")]
check("transactions: no truth / removed fields", not leak, ",".join(leak))
check("customers / accounts: no truth fields", not any(c.startswith(("gt_", "truth_")) for c in list(cu.columns) + list(ac.columns)),
      f"customers={list(cu.columns)} accounts={list(ac.columns)}")
check("txn_id unique", tx.txn_id.is_unique, f"{len(tx):,} rows")
check("labels 1:1 with transactions", len(lb) == len(tx) and set(lb.txn_id) == set(tx.txn_id))
check("codes read as text (mcc keeps leading zeros possible)", tx.mcc.dropna().map(type).eq(str).all())

# ---- balances
tx["signed"] = np.where(tx.direction == "CR", tx.amount, -tx.amount)
tx = tx.sort_values(["account_id", "posted_datetime", "txn_id"])
check("balance_after never negative", (tx.balance_after >= -0.005).all(), f"min={tx.balance_after.min():,.2f}")
prev = tx.groupby("account_id").balance_after.shift()
diff = (prev + tx.signed - tx.balance_after).abs()
check("balance_after = previous + signed amount", (diff.dropna() < 0.011).all(), f"max diff={diff.max():.4f}")
check("amount > 0", (tx.amount > 0).all())
check("txn account belongs to customer", tx.merge(ac, on=["customer_id", "account_id"]).shape[0] == len(tx))

# ---- series truth
check("every series has cycle_type / true_period_days", st.cycle_type.notna().all() and st.true_period_days.notna().all())
labelled = set(lb.gt_series_id.dropna())
check("label series ids exist in series_truth", labelled <= set(st.series_id))

# ---- split & censoring
held = set(ut.customer_id[ut.user_split == "heldout"])
check("users_truth.user_split is train / heldout", ut.user_split.isin(["train", "heldout"]).all(), f"{len(held)} held-out ({len(held) / len(ut):.0%})")
has_sn = all(os.path.exists(f"{d}/snapshots_{p}.csv.gz") or os.path.exists(f"{d}/snapshots_{p}.csv") for p in ("train", "calib", "test"))
sn = {p: load(d, f"snapshots_{p}", parse_dates=["snapshot_date", "next_date"]) for p in ("train", "calib", "test")} if has_sn else {}
if has_sn:
    check("held-out customers absent from train/calib", not (set(sn["train"].customer_id) | set(sn["calib"].customer_id)) & held)
# payday validation needs: a June 2026 payroll credit, >= 2 credits before it, next credit 10-40 days later
pr = tx[(tx.direction == "CR") & (tx.txn_type == "payroll_in")].copy()
pr["d"] = pd.to_datetime(pr.posted_datetime).dt.normalize()
ok = []
for cid, g in pr.groupby("customer_id"):
    d_ = sorted(g.d.unique())
    june = [x for x in d_ if pd.Timestamp("2026-06-01") <= x < pd.Timestamp("2026-07-01")]
    if not june:
        ok.append((cid, False, False, False))
        continue
    j = june[0]
    before, after = [x for x in d_ if x < j], [x for x in d_ if x > j]
    ok.append((cid, True, len(before) >= 2, bool(after) and 10 <= (after[0] - j).days <= 40))
ok = pd.DataFrame(ok, columns=["customer_id", "june", "two_before", "next_10_40"])
check("payday: customers with payroll have a June 2026 salary credit", ok.june.mean() >= 0.95, f"{ok.june.mean():.1%} of {len(ok):,}")
check("payday: >= 2 salary credits before the June payday", ok[ok.june].two_before.mean() >= 0.95, f"{ok[ok.june].two_before.mean():.1%}")
check("payday: next salary credit 10-40 days after June payday", ok[ok.june].next_10_40.mean() >= 0.95, f"{ok[ok.june].next_10_40.mean():.1%}")
check("data runs to 2026-09-30", str(tx.posted_datetime.max()) >= "2026-09-30", str(tx.posted_datetime.max()))
for p, cut in ((("train", SPLIT_TRAIN_END), ("calib", SPLIT_CALIB_END)) if has_sn else ()):
    s, c = sn[p], pd.Timestamp(cut)
    check(f"{p}: snapshots before cutoff", (s.snapshot_date < c).all())
    check(f"{p}: no outcome after cutoff (censored)", ((s.next_date < c) | s.next_date.isna()).all() and
          (s.snapshot_date + pd.to_timedelta(s.duration_days, unit="D") <= c).all())
    check(f"{p}: active_label empty when unknown", s.loc[s.event_observed == 0, "active_label"].isin([0]).sum()
          + s.loc[s.event_observed == 0, "active_label"].isna().sum() == (s.event_observed == 0).sum())
if has_sn:
    check("test snapshots 2026-02..2026-06", sn["test"].snapshot_date.min() >= pd.Timestamp("2026-02-01")
          and sn["test"].snapshot_date.max() < pd.Timestamp("2026-07-01"))
    check("snapshots carry truth_cycle_type", all(s.truth_cycle_type.notna().all() for s in sn.values()))
else:
    print("(no snapshot files: snapshot checks skipped; the PACM pipeline does not need them)")

pd.set_option("display.width", 200)
print(pd.DataFrame(res, columns=["check", "result", "info"]).to_string(index=False))

# ---- summary tables
print(f"\ncustomers={len(cu):,} (held-out {len(held)}) accounts={len(ac):,} transactions={len(tx):,} series={len(st):,}")
obl = st[st.kind.isin(["obligation", "recurring_transfer"])]
print("\nseries by cycle_type x status (obligations + recurring transfers):")
print(obl.groupby(["cycle_type", "status"]).size().unstack(fill_value=0).to_string())
iv = st[st.cycle_type == "interval"]
print("\ninterval series per customer:", st.groupby("customer_id").cycle_type.apply(lambda x: (x == "interval").sum()).value_counts().sort_index().to_dict())
print("\ninterval series by name: n, period range, sigma range, share with skipped cycles / late start / cancelled")
print(iv.groupby("name").agg(n=("series_id", "size"), period_min=("true_period_days", "min"), period_max=("true_period_days", "max"),
                             skipped=("quirks", lambda q: q.fillna("").str.contains("skipped").mean().round(2)),
                             late=("quirks", lambda q: q.fillna("").str.contains("late_start").mean().round(2)),
                             cancelled=("status", lambda s: (s == "cancelled").mean().round(2))).to_string())
if has_sn:
    print("\nsnapshots (expense kinds) by split x cycle_type:")
    rows = {p: s_[s_.truth_kind.isin(["obligation", "recurring_transfer"])].truth_cycle_type.value_counts() for p, s_ in sn.items()}
    print(pd.DataFrame(rows).fillna(0).astype(int).to_string())
sys.exit(0 if all(r[1] == "PASS" for r in res) else 1)
