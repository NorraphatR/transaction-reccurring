#!/usr/bin/env python3
"""
Live prediction for the real statement: today = the day after the last transaction in the data (no outcome to check yet).
Model unchanged (fitted as in pacm_v3 on --res / --data); the owner's standing answers are applied (moment 5b: small very
regular payment confirmed as a bill). Reserve and safe to spend: the safe_v5 rules.
Usage: python predict_now.py [--res results_v7b] [--data data_v7]  -> results_real/now_<date>.json
"""
import argparse
import json
import pickle
from datetime import date

import numpy as np
import pandas as pd

import pacm_v3 as pm
import safe_v5 as sv
import timing_v3 as tm
from predict_real import fit_models, real_groups

fmt = lambda o: date.fromordinal(int(o)).strftime("%d %b %Y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_v7b")
    ap.add_argument("--data", default="data_v7")
    a = ap.parse_args()
    det, am, pmod, iso_p, iso_c = fit_models(a.res, a.data)
    pm.init(pickle.load(open(f"{a.res}/timing_model.pkl", "rb")), json.load(open(f"{a.res}/amount_split_prior.json")))
    tx, day, truth = real_groups("real_data")
    full = tm.load("real_data", "transactions")
    full["o"] = (pd.to_datetime(full.posted_datetime).dt.normalize() - pd.Timestamp("1970-01-01")).dt.days + tm.EPOCH
    full = full.sort_values("posted_datetime")
    S = int(full.o.max()) + 1
    today = date.fromordinal(S)
    pm.CENSOR["test"] = S
    rows = []
    for k, g in day.groupby("gkey"):
        if len(g) < 3:
            continue
        task = {"gkey": k, "customer_id": k.split("#")[0], "user_split": "real", "key_type": k.split("#")[1][0], "cat": g.txn_type.mode().iloc[0],
                "o": g.o.to_numpy(), "amount": g.amount.to_numpy(float), "is_exp": g.is_exp.to_numpy(bool), "sid": g.sid.to_numpy(object),
                "auto": g.auto.to_numpy(float), "mandate": g.mand.to_numpy(float), "splits": ("test",), "today": today}
        rows += [r for r in pm.candidates(task) if r["is_today"]]
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
    ans = json.load(open("results_real/real_answers.json", encoding="utf-8"))
    yes5 = {q["gkey"] for q in ans["app_questions"] if q["moment"] == 5 and q["answer"].startswith("Yes")}
    df["source"] = np.where(df.detected, "model", np.where(df.gkey.isin(yes5), "you (moment 5b)", ""))
    pay = full[(full.direction == "CR") & (full.txn_type == "payroll_in")]
    nxt, L = sv.next_payday(np.sort(pay.o.unique()), S)
    left = nxt - S
    bal = float(full.balance_after.iloc[-1])
    out = {}
    for tag, sel in [("model", df.detected), ("questions", df.detected | df.gkey.isin(yes5))]:
        lines = []
        for r in df[sel & (df.P >= 0.5)].itertuples():
            per = max(1.0, float(np.floor(r.median_gap)))
            amt = r.a_p90 / per * (left + 1)
            how = f"฿{r.a_p90:,.0f} every {per:.0f} days = ฿{r.a_p90 / per:,.1f}/day × ({left} + 1) days"
            if r.t_p10 < left and amt < r.a_p90:
                amt, how = r.a_p90, "due before payday: one full payment"
            lines.append({"payee": r.payee, "source": r.source, "P": round(r.P, 2), "T": fmt(S + r.t_p50), "window": f"{fmt(S + r.t_p10)} – {fmt(S + min(r.t_p90, 400))}",
                          "A": round(r.a_med, 2), "A_P90": round(r.a_p90, 2), "C": round(r.C, 2), "reserve": round(amt, 2), "how": how})
        up = sum(l["reserve"] for l in lines)
        spend = bal - up
        buf = 0.2 * max(spend, 0)
        out[tag] = {"lines": lines, "upcoming": up, "spendable": spend, "buffer": buf, "safe": spend - buf, "per_day": (spend - buf) / max(1, left)}
    res = {"today": fmt(S), "next_payday": fmt(nxt), "days_to_payday": left, "balance": bal, "model_from": a.res, **out}
    json.dump(res, open(f"results_real/now_{today.isoformat()}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
