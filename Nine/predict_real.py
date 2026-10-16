#!/usr/bin/env python3
"""
Run the trained SafeBelt model on real bank data (real_data/: one customer, converted KBank statements).

No model change: the models are fitted exactly as in pacm_v3.py (same code, same seed) on the v7b synthetic training
candidates (results_v7b/pacm_cands.csv.gz), with the v7b bill split prior and timing model, then applied to the real data.
Validation days = the day after every salary credit (payroll_in) with 2 earlier ones; the last one is today.
Truth = the account owner's own answers (real_data/truth_by_user.csv), used only to label and score, never as features.

Usage: python predict_real.py [--real real_data] [--res results_v7b] [--out results_real]
"""
import argparse
import json
import os
import pickle
from datetime import date

import numpy as np
import pandas as pd

import pacm_v3 as pm
import timing_v3 as tm
from backtest_v2 import predict as b1_predict


def fit_models(res, data):
    """the pacm_v3 model section, unchanged, on the synthetic candidates"""
    df = pd.read_csv(f"{res}/pacm_cands.csv.gz")
    st = tm.load(data, "series_truth")
    df["y_bill"] = (df.exp_share >= 0.5) & ~df.dom_sid.map(st.set_index("series_id").category).isin(pm.NOT_BILL_CATEGORIES)
    tr, ca = df.split == "train", df.split == "calib"
    det = pm.lgb_fit(pm.frame(df, pm.D_FEATURES)[tr], df.y_bill[tr])
    df["det_score"] = det.predict(pm.frame(df, pm.D_FEATURES))
    df["detected"] = df.det_score >= 0.5
    df["logs"] = df.log_amounts.astype(str).str.split().map(lambda v: np.array(v, float))
    trs = df[tr & df.y_bill].drop_duplicates(["gkey", "cand"], keep="last")
    am = pm.AmountModel().fit(list(trs.logs), list(trs.txn_type))
    Xp = pm.frame(df, pm.P_FEATURES)
    ptr = tr & df.active_label.notna() & df.detected
    pmod = pm.lgb_fit(Xp[ptr], df.active_label[ptr].astype(int), 400)
    df["p_raw"] = pmod.predict(Xp)
    pca = ca & df.active_label.notna() & df.detected
    iso_p = pm.Isotonic().fit(df.p_raw[pca], df.active_label[pca])
    df["P"] = iso_p.predict(df.p_raw)
    pr = [am.predict(x, c, q=0.9) for x, c in zip(df.logs, df.txn_type)]
    df["p_amt"] = [pm.AmountModel.cdf_t(np.log(p[1]), *p[2]) for p in pr]
    df["in_window"] = df.next_days.notna() & (df.next_days >= df.t_p10) & (df.next_days <= df.t_p90)
    df["amt_ok"] = df.next_amount.notna() & (df.next_amount <= np.array([p[1] for p in pr]) * 1.0001)
    df["c_target"] = (df.in_window & df.amt_ok).astype(int)
    df["c_observable"] = df.next_days.notna() | (df.active_label == 0) | ((df.snapshot.map(lambda s: date.fromisoformat(s).toordinal()) + df.t_p90)
                                                                        < df.split.map(pm.CENSOR))
    df["c_raw"] = df.P * 0.8 * df.p_amt
    cca = ca & df.detected & df.c_observable
    iso_c = pm.Isotonic().fit(df.c_raw[cca], df.c_target[cca])
    return det, am, pmod, iso_p, iso_c


def real_groups(real):
    """payee groups (same as the model) + the user's truth table: a payment belongs to a bill series when its payee key is
    listed in truth_by_user.csv (labels for validation only, never features)"""
    tx = pm.build_groups(real)
    truth = tm.load(real, "truth_by_user")
    key2sid = {k: r.series_id for r in truth.itertuples() for k in r.payee_keys.split(";")}
    tx["sid"] = tx.gkey.str.split("#").str[1].map(key2sid)
    tx["is_exp"] = tx.sid.notna()
    day = tx.sort_values("posted_datetime").groupby(["gkey", "o"]).agg(
        amount=("amount", "sum"), txn_type=("txn_type", "first"), auto=("channel", lambda v: (v == "AUTO").any()),
        mand=("mandate_id", lambda v: v.notna().any()), sid=("sid", "first"), is_exp=("is_exp", "max")).reset_index()
    return tx, day, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="real_data")
    ap.add_argument("--res", default="results_v7b")
    ap.add_argument("--data", default="data_v7", help="synthetic data the models were trained on (for the bill labels)")
    ap.add_argument("--out", default="results_real")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    det, am, pmod, iso_p, iso_c = fit_models(a.res, a.data)
    tmodel = pickle.load(open(f"{a.res}/timing_model.pkl", "rb"))
    split = json.load(open(f"{a.res}/amount_split_prior.json"))
    pm.init(tmodel, split)

    tx, day, truth = real_groups(a.real)
    pay = tm.load(a.real, "transactions", usecols=["posted_datetime", "direction", "txn_type"])
    pay = pd.to_datetime(pay[(pay.direction == "CR") & (pay.txn_type == "payroll_in")].posted_datetime).dt.date.sort_values()
    end = pd.to_datetime(tx.posted_datetime).max().date()
    # validation days: the day after every payday with at least 2 earlier paydays (the last one = "today")
    todays = [date.fromordinal(d.toordinal() + 1) for d in pay.iloc[2:]]
    pm.CENSOR["test"] = end.toordinal() + 1                   # outcomes are read to the end of the real data
    rows = []
    for k, g in day.groupby("gkey"):
        if len(g) < 3:
            continue
        cid = k.split("#")[0]
        for today in todays:
            task = {"gkey": k, "customer_id": cid, "user_split": "real", "key_type": k.split("#")[1][0], "cat": g.txn_type.mode().iloc[0],
                    "o": g.o.to_numpy(), "amount": g.amount.to_numpy(float), "is_exp": g.is_exp.to_numpy(bool), "sid": g.sid.to_numpy(object),
                    "auto": g.auto.to_numpy(float), "mandate": g.mand.to_numpy(float), "splits": ("test",), "today": today}
            rows += [r for r in pm.candidates(task) if r["is_today"]]
    today, last_salary = todays[-1], pay.iloc[-1]
    df = pd.DataFrame(rows)
    df["det_score"] = det.predict(pm.frame(df, pm.D_FEATURES))
    df["detected"] = df.det_score >= 0.5
    df["logs"] = df.log_amounts.astype(str).str.split().map(lambda v: np.array(v, float))
    pr = [am.predict(x, c, q=0.9) for x, c in zip(df.logs, df.txn_type)]
    df["a_med"], df["a_p90"] = [p[0] for p in pr], [p[1] for p in pr]
    df["p_amt"] = [pm.AmountModel.cdf_t(np.log(p[1]), *p[2]) for p in pr]
    df["P"] = iso_p.predict(pmod.predict(pm.frame(df, pm.P_FEATURES)))
    df["C"] = iso_c.predict(df.P * 0.8 * df.p_amt)
    names = pm.payee_names(tx)
    df["payee"] = df.gkey.map(names)
    df["y_bill"] = df.exp_share >= 0.5
    df["today"], df["last_salary"], df["data_end"] = df.snapshot, last_salary.isoformat(), end.isoformat()
    df["is_last"] = df.snapshot == today.isoformat()
    df.drop(columns=["logs"]).to_csv(f"{a.out}/predictions_real.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(f"{len(todays)} validation days ({todays[0]} .. {today}), data to {end}; {len(df)} candidate rows, {int(df.detected.sum())} detected")
    x = df[df.detected | df.y_bill]
    cols = ["snapshot", "payee", "y_bill", "detected", "det_score", "P", "t_p10", "t_p50", "t_p90", "a_med", "a_p90", "C", "next_days", "next_amount"]
    print(x.sort_values(["snapshot", "payee"])[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
