#!/usr/bin/env python3
"""
SafeBelt baseline evaluation on the official v2 split
=====================================================

Split (made by generate_v2.py, leak-free):
  snapshots_train.csv.gz  train customers, snapshots < 2025-10-01, labels censored at 2025-10-01
  snapshots_calib.csv.gz  train customers, 2025-10-01..2026-01-31, labels censored at 2026-02-01
  snapshots_test.csv.gz   ALL customers,   2026-02-01..2026-06-30, outcomes observed to 2026-09-30
                          user_split = "train" (seen customers, future months) or "heldout" (never seen)

Baselines (no training needed; ML models train on train, calibrate on calib, report on test):
  B0 naive     next = last + median gap (1 occurrence -> 30 days); amount = last; active if overdue < 1.5x gap (or 45 d)
  B1 calendar  gap snapped to canonical period (within 15%), monthly-like keeps day of month; amount = median of last 3

Usage: python backtest_v2.py --data data_v2 --out results_v2
"""
import argparse
import os

import numpy as np
import pandas as pd

CANON = np.array([7, 14, 15.2, 30.4, 91.3, 182.6, 365.25])
MONTHLY_LIKE = {30.4: 1, 91.3: 3, 182.6: 6, 365.25: 12}
EXPENSE_KINDS = ["obligation", "recurring_transfer"]


def snap_period(med):
    if pd.isna(med):
        return np.nan
    j = np.argmin(np.abs(CANON - med) / CANON)
    return CANON[j] if abs(CANON[j] - med) / CANON[j] <= 0.15 else np.nan


def predict(df):
    df = df.copy()
    last = pd.to_datetime(df.last_date)
    gap0 = df.median_gap_prior.fillna(30.0)
    df["b0_days"] = ((last + pd.to_timedelta(gap0, unit="D")) - df.snapshot_date).dt.days.clip(lower=0)
    df["b0_amount"] = df.last_amount
    df["b0_active"] = (df.days_since_last <= np.where(df.median_gap_prior.notna(), 1.5 * df.median_gap_prior, 45)).astype(int)
    per = df.median_gap_prior.apply(snap_period)
    nxt = last + pd.to_timedelta(per.fillna(df.median_gap_prior).fillna(30.4), unit="D")
    k = per.map(MONTHLY_LIKE)
    m = k.notna()
    nxt.loc[m] = [l + pd.DateOffset(months=int(x)) for l, x in zip(last[m], k[m])]
    df["b1_days"] = (nxt - df.snapshot_date).dt.days.clip(lower=0)
    df["b1_amount"] = df.amount_med3
    lim = np.where(per.notna(), 1.5 * per, np.where(df.median_gap_prior.notna(), 1.5 * df.median_gap_prior, 45))
    df["b1_active"] = (df.days_since_last <= lim).astype(int)
    return df


def metrics(g, m):
    ev = g[g.event_observed == 1]
    err = (ev[f"{m}_days"] - ev.duration_days).abs()
    amt = ev[ev.next_amount.notna()]
    ape = (amt[f"{m}_amount"] - amt.next_amount).abs() / amt.next_amount.clip(lower=1)
    t, p = g.active_label == 0, g[f"{m}_active"] == 0
    tp = int((t & p).sum())
    return pd.Series({"n": len(g), "timing_MAE_d": err.mean(), "within_3d": (err <= 3).mean(), "within_7d": (err <= 7).mean(),
                      "amount_within_5pct": (ape <= 0.05).mean(), "cancel_precision": tp / max(int(p.sum()), 1),
                      "cancel_recall": tp / max(int(t.sum()), 1), "n_cancel": int(t.sum())})


def fmt(t):
    t = t.copy()
    for c, f in {"timing_MAE_d": "{:.1f}", "within_3d": "{:.0%}", "within_7d": "{:.0%}", "amount_within_5pct": "{:.0%}",
                 "cancel_precision": "{:.0%}", "cancel_recall": "{:.0%}"}.items():
        t[c] = t[c].map(lambda v: f.format(v) if pd.notna(v) else "-")
    for c in ("n", "n_cancel"):
        t[c] = t[c].astype(int)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v2")
    ap.add_argument("--out", default="results_v2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    te = pd.read_csv(f"{a.data}/snapshots_test.csv.gz", parse_dates=["snapshot_date", "last_date", "next_date"])

    # trap: own-account transfers look recurring; if not excluded they become fake obligations
    own = te[te.truth_kind == "own_transfer"]
    trap = own.groupby(["customer_id", "snapshot_date"]).last_amount.sum()
    exp = te[te.truth_kind.isin(EXPENSE_KINDS)]
    p = predict(exp)
    p["hist"] = pd.cut(p.n_prior, [0, 1, 3, 6, 1000], labels=["1", "2-3", "4-6", "7+"])

    tables = {}
    for m in ("b0", "b1"):
        tables[f"{m} overall"] = metrics(p, m).to_frame(m).T
        tables[f"{m} by customer split"] = p.groupby("user_split").apply(lambda g: metrics(g, m), include_groups=False)
        tables[f"{m} by period"] = p.groupby("truth_period_type").apply(lambda g: metrics(g, m), include_groups=False)
        tables[f"{m} by history"] = p.groupby("hist", observed=True).apply(lambda g: metrics(g, m), include_groups=False)
    with open(f"{a.out}/baseline_report.md", "w") as fh:
        fh.write(f"# Baseline report (test: {te.snapshot_date.min():%Y-%m} to {te.snapshot_date.max():%Y-%m})\n\n")
        fh.write(f"Own-account trap: {own.series_id.nunique()} own-transfer series in test; if treated as obligations they would "
                 f"wrongly reserve a median of {trap.median():,.0f} THB per affected customer-month "
                 f"({trap.index.get_level_values(0).nunique()} customers affected).\n\n")
        for k, t in tables.items():
            fh.write(f"## {k}\n\n{fmt(t).to_markdown()}\n\n")
            print(f"\n== {k}\n{fmt(t).to_string()}")
    print(f"\nown-account trap: {own.series_id.nunique()} series, median wrongly reserved {trap.median():,.0f} THB per customer-month")
    p.to_csv(f"{a.out}/baseline_predictions_test.csv.gz", index=False)


if __name__ == "__main__":
    main()
