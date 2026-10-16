#!/usr/bin/env python3
"""SafeBelt v5 overview page (results_v5/safebelt_v5.html), flow:
 1 correctness, model only vs model + questions   2 P, A, T, C validation
 3 one customer (model only / model + questions): transactions with recurring highlighted, amount peaks per payee,
   safe to spend with every step, safety buffer, shortfall, model predictions
Usage: python viz_v5.py [--today 2026-06-01] [--customers 240]
"""
import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

import amount_split as asp
import pacm_v3 as pm


def levels(amounts, near=np.log(1.2), apart=np.log(1.45), min_share=0.1):
    """the same amount-peak rule the series builder uses, returning the peaks (amount, payments within +-20%)"""
    la = np.log(np.maximum(np.asarray(amounts, float), 1))
    if len(la) < 4:
        return []
    support = (np.abs(la[:, None] - la[None, :]) < near).sum(1)
    out = []
    for j in np.argsort(-support, kind="stable"):
        if support[j] < max(2, min_share * len(la)):
            break
        if all(abs(la[j] - np.log(l)) >= apart for l, _ in out):
            out.append((float(np.exp(la[j])), int(support[j])))
    return sorted(out)


def bill_split(o, amounts, prior, model, cat):
    """the series builder's split (amount_split.py): bill levels (amount, payments) and the cut points between them"""
    if len(amounts) < 2:
        return [], []
    segs = asp.split(o, amounts, prior, model, cat)
    lv = [(float(np.exp(m)), int(n)) for n, m, _ in segs]
    if len(segs) < 2:
        return lv, []
    grid = np.exp(np.linspace(np.log(max(min(amounts), 1)) - 1, np.log(max(amounts)) + 1, 2000))
    lab = asp.assign(grid, segs, prior)
    return lv, [round(float(np.sqrt(grid[i] * grid[i + 1])), 2) for i in np.where(np.diff(lab) != 0)[0]]


def reliability(p, y, bins=(0, .2, .4, .6, .7, .8, .9, 1.0001)):
    b = pd.cut(p, bins, right=False)
    t = pd.DataFrame({"p": p, "y": y}).groupby(b, observed=True).agg(n=("y", "size"), says=("p", "mean"), happened=("y", "mean"))
    return [{"n": int(r.n), "says": float(r.says), "happened": float(r.happened)} for _, r in t.iterrows()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v6")
    ap.add_argument("--res", default="results_v6")
    ap.add_argument("--today", default="payday", help="label of the per-customer 'today' outputs")
    ap.add_argument("--customers", type=int, default=240)
    ap.add_argument("--name", default=None, help="page name, e.g. v7 (default: from --res)")
    a = ap.parse_args()
    import os, pickle
    bayes = os.path.exists(f"{a.res}/amount_split_prior.json")
    if bayes:
        prior = json.load(open(f"{a.res}/amount_split_prior.json"))
        tmodel = pickle.load(open(f"{a.res}/timing_model.pkl", "rb"))
    name = a.name or a.res.replace("results_", "")
    st, ut, tx, day = pm.load_data(a.data)
    names = pm.payee_names(tx)
    split = dict(zip(ut.customer_id, ut.user_split))
    real = st.set_index("series_id")

    pred = pd.read_csv(f"{a.res}/pacm_predictions.csv.gz", low_memory=False)
    te = pred[pred.split == "test"].copy()
    hb = pd.read_csv(f"{a.res}/hitl_bills_{a.today}.csv", encoding="utf-8-sig", low_memory=False)
    mo = pd.read_csv(f"{a.res}/pacm_output_{a.today}.csv", encoding="utf-8-sig", low_memory=False)
    sf = pd.read_csv(f"{a.res}/safe_to_spend_{a.today}.csv", encoding="utf-8-sig")
    sl = pd.read_csv(f"{a.res}/safe_lines_{a.today}.csv", encoding="utf-8-sig")
    ss = pd.read_csv(f"{a.res}/safe_short_{a.today}.csv", encoding="utf-8-sig")
    qs = pd.read_csv(f"{a.res}/hitl_questions_{a.today}.csv", encoding="utf-8-sig", low_memory=False)
    out = {"today": "each customer's payday (+1 day), June 2026"}
    SM = {c: date.fromisoformat(d).toordinal() for c, d in zip(sf.customer_id, sf.today)}

    # ---- 1 correctness
    def money_ok(v, act):
        amt = pd.to_numeric(act.str.extract(r"(\d[\d,]*)$")[0].str.replace(",", ""), errors="coerce")
        m = hb.truth.loc[act.index].str.contains(r"\[bill", na=False) & amt.notna()
        return float(amt[m & v.str.startswith("correct")].sum() / amt[m].sum())
    vb, va = hb.verdict_before.value_counts(), hb.verdict_after.value_counts()
    t = pd.DataFrame({"before": vb, "after": va}).fillna(0).astype(int).sort_values("before", ascending=False)
    out["verdicts"] = [{"verdict": k, "before": int(r.before), "after": int(r.after)} for k, r in t.iterrows() if k != "correct: not a bill"]
    keep = hb[hb.verdict_before != "correct: not a bill"]
    out["correct"] = {"model": float(keep.verdict_before.str.startswith("correct").mean()), "questions": float(keep.verdict_after.str.startswith("correct").mean()),
                      "model_money": money_ok(hb.verdict_before, hb.actual), "questions_money": money_ok(hb.verdict_after, hb.actual), "rows": int(len(keep))}
    # correctness by customer history length (months from first account opening to today), cumulative buckets
    cu = pm.tm.load(a.data, "customers")
    opened = dict(zip(cu.customer_id, pd.to_datetime(cu.first_account_open_date).map(lambda t: t.toordinal())))
    age = hb.customer_id.map(lambda c: (SM[c] - opened[c]) / 30.4 if c in SM else np.nan)
    sage = sf.customer_id.map(lambda c: (SM[c] - opened[c]) / 30.4)
    out["by_age"] = []
    for lim in (3, 6, 12, None):
        m = (age <= lim) if lim else age.notna()
        k = m & (hb.verdict_before != "correct: not a bill")
        out["by_age"].append({"label": f"0 – {lim} months" if lim else "all customers", "customers": int((sage <= lim).sum() if lim else len(sage)),
                              "bills": int(k.sum()), "model": float(hb.verdict_before[k].str.startswith("correct").mean()) if k.any() else None,
                              "questions": float(hb.verdict_after[k].str.startswith("correct").mean()) if k.any() else None,
                              "model_money": money_ok(hb.verdict_before[m], hb.actual[m]) if k.any() else None,
                              "questions_money": money_ok(hb.verdict_after[m], hb.actual[m]) if k.any() else None})
        print(out["by_age"][-1])
    out["money"] = {tag: {"upcoming": float(sf[f"upcoming_{tag}"].mean()), "real": float(sf.real_bills.mean()), "short": float(sf[f"bill_short_{tag}"].mean()),
                          "short500": float((sf[f"bill_short_{tag}"] > 500).mean()), "short2000": float((sf[f"bill_short_{tag}"] > 2000).mean()),
                          "extra": float(sf[f"reserved_extra_{tag}"].mean()),
                          "overall_short": float((-sf[f"overall_{tag}"]).clip(lower=0).mean()), "overall_neg": float((sf[f"overall_{tag}"] < 0).mean()),
                          "overall_2000": float((sf[f"overall_{tag}"] < -2000).mean())} for tag in ("model", "questions")}
    qq = qs[~qs.get("skipped", pd.Series(False, index=qs.index)).fillna(False).astype(bool)]
    out["nonrec_month"] = {"median": float(sf.nonrec_month_avg.median()), "mean": float(sf.nonrec_month_avg.mean()),
                           "p10": float(sf.nonrec_month_avg.quantile(.1)), "p90": float(sf.nonrec_month_avg.quantile(.9))}
    out["name"] = name
    out["questions_per_customer"] = float(len(qq[(qq.moment > 1) | qq.this_month.astype(bool)]) / max(1, sf.shape[0]))

    # ---- 2 P A T C (model, all test snapshots)
    det = {}
    for us, g in [("all", te)] + list(te.groupby("user_split")):
        tp, fp, fn = int((g.detected & g.y_bill).sum()), int((g.detected & ~g.y_bill).sum()), int((~g.detected & g.y_bill).sum())
        det[us] = {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}
    ev = te[te.detected & te.y_bill & te.next_days.notna()]
    e, e1 = (ev.t_p50 - ev.next_days).abs(), (ev.b1_days - ev.next_days).abs()
    inwin = (ev.next_days >= ev.t_p10) & (ev.next_days <= ev.t_p90)
    out["pacm"] = {"det": det, "t_mae": float(e.mean()), "b1_mae": float(e1.mean()), "t_3d": float((e <= 3).mean()), "b1_3d": float((e1 <= 3).mean()),
                   "in_win": float(inwin.mean()), "amt_ok": float((ev.next_amount <= ev.a_p90 * 1.0001).mean()),
                   "ape": float(((ev.a_med - ev.next_amount).abs() / ev.next_amount).median()),
                   "p_rel": reliability(te[te.detected & te.active_label.notna()].P, te[te.detected & te.active_label.notna()].active_label),
                   "c_rel": reliability(te[te.detected & te.c_observable.astype(bool)].C, te[te.detected & te.c_observable.astype(bool)].c_target)}

    # ---- 3 customers
    sf["diff"] = sf.bill_short_model - sf.bill_short_questions
    rng = np.random.default_rng(1)
    chosen = list(sf.sort_values("diff", ascending=False).customer_id[:60]) + list(sf.sort_values("overall_model").customer_id[:40])
    rest = [c for c in sf.customer_id if c not in chosen]
    chosen += list(rng.choice(rest, size=min(len(rest), a.customers - len(chosen)), replace=False))
    chosen = list(dict.fromkeys(chosen))
    td = te[te.is_today.astype(bool)]
    det_model = set(td[td.detected.astype(bool)].gkey)
    hq = hb.copy()
    det_q = set(hq[hq.detected.astype(str) == "True"].gkey.dropna())
    tx_c = tx[tx.customer_id.isin(chosen)]
    sfi = sf.set_index("customer_id")
    custs = []
    for cid in chosen:
        g = tx_c[tx_c.customer_id == cid]
        S = SM[cid]
        nxt_pay = date.fromisoformat(sfi.loc[cid, "next_payday"]).toordinal()
        g = g[(g.o >= S - 730) & (g.o < nxt_pay)]
        gk = list(dict.fromkeys(g.gkey))
        gi = {k: i for i, k in enumerate(gk)}
        groups = []
        for k in gk:
            h = g[g.gkey == k]
            hist = h[h.o < S].groupby("o").amount.sum()
            truth_bill = bool(h.is_exp.mean() >= 0.5)
            isp = k.split("#")[1].startswith("P")
            if bayes and isp:
                lv, cuts = bill_split(hist.index.to_numpy(), hist.values, prior, tmodel, h.txn_type.mode().iloc[0])
            else:
                lv = levels(hist.values) if isp or len(hist) >= 4 else []
                cuts = [round(float(np.sqrt(lv[i][0] * lv[i + 1][0])), 2) for i in range(len(lv) - 1)] if isp else []
            groups.append({"payee": names.get(k, "?"), "type": k.split("#")[1][0], "model": k in det_model, "questions": k in det_q, "truth": truth_bill,
                           "n": int(len(hist)), "peaks": [{"amount": round(x, 2), "n": n} for x, n in lv], "cuts": cuts,
                           "amounts": [round(float(v), 2) for v in hist.values[-120:]]})
        txs = [[int(o - S), round(float(am), 2), gi[k]] for o, am, k in zip(g.o, g.amount, g.gkey)]
        r = sfi.loc[cid]
        money = {k: (None if pd.isna(r[k]) else (bool(r[k]) if isinstance(r[k], (bool, np.bool_)) else (float(r[k]) if not isinstance(r[k], str) else r[k])))
                 for k in ("today", "balance", "salary_received", "salary_date", "next_payday", "real_next_payday", "days_to_payday", "regular_salary", "real_bills", "real_nonrecurring", "nonrec_month_avg", "nonrec_days",
                           "upcoming_model", "spendable_model", "buffer_model", "safe_model", "bill_short_model", "reserved_extra_model", "overall_model", "overall_questions",
                           "upcoming_questions", "spendable_questions", "buffer_questions", "safe_questions", "bill_short_questions", "reserved_extra_questions")}
        lines = {tag: sl[(sl.customer_id == cid) & (sl.version == tag)][["payee", "kind", "reserve", "how"]].round(2).to_dict(orient="records") for tag in ("model", "questions")}
        short = {tag: ss[(ss.customer_id == cid) & (ss.version == tag)][["date", "payee", "bill", "amount", "covered", "short", "reserved_for_it"]]
                 .replace({np.nan: None}).round(2).to_dict(orient="records") for tag in ("model", "questions")}
        bm = mo[(mo.customer_id == cid) & (mo.verdict != "correct: not a bill")][["payee", "truth", "detected", "P", "T", "T_window", "a_p90", "C", "actual", "verdict"]]
        bq = hq[(hq.customer_id == cid) & (hq.verdict_after != "correct: not a bill")][["payee", "truth", "source", "detected", "P", "T", "T_window", "a_p90", "C", "actual", "asked", "verdict_after"]] \
            .rename(columns={"verdict_after": "verdict"})
        custs.append({"id": cid, "split": split[cid], "hist": round((S - opened[cid]) / 30.4, 1), "groups": groups, "txns": txs, "money": money, "lines": lines, "short": short,
                      "bills": {"model": bm.replace({np.nan: None}).to_dict(orient="records"), "questions": bq.replace({np.nan: None}).to_dict(orient="records")}})
    out["customers"] = custs
    data = json.dumps(out, ensure_ascii=False, default=float)
    data = data.replace("</", "<" + "\\" + "/")
    html = open("viz_v5_template.html", encoding="utf-8").read().replace("/*DATA*/null", data).replace("SafeBelt v6", f"SafeBelt {name}")
    open(f"{a.res}/safebelt_{name}.html", "w", encoding="utf-8").write(html)
    print(f"wrote {a.res}/safebelt_{name}.html ({len(html) / 1e6:.1f} MB, {len(custs)} customers)")


if __name__ == "__main__":
    main()
