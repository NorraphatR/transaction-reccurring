#!/usr/bin/env python3
"""
SafeBelt human in the loop v5 - a separate step after the model (pacm_v3.py). It never changes the model.

Reads the model's predictions for `today`, decides which questions the app would ask, answers them from the truth
table (simulated user: always answers, always correctly), applies the answers to the bill outputs and labels every
bill with the question asked and its verdict before / after.

Moments
  1  new payment >= 1,000 THB to a payee never paid before (also in the account's first 30 days; only if the model has no prediction for that bill today; not ATM, not one-off
     merchant categories): "When will you pay this again?" -> declared bill (P 1, T from the answer, A = amount)
  3  variable bill (amount spread > 15%, >= 1,000 THB) due within 30 days: "About how much this month?" -> A = answer
  4  overdue bill (overdue ratio > 1.15, 0.2 <= P < 0.95): "Is it still active?" -> P = 1 / 0
  5  unsure detection (score 0.3-0.7, >= 1,000 THB): "Is this a bill?" -> detected = answer
  Moments 3-5: at most 3 questions per customer per month, highest money at stake first.

Usage: python hitl_v5.py [--today 2026-06-01]   -> results_v5/hitl_bills_<today>.csv, hitl_questions_<today>.csv, hitl_app.html
"""
import argparse
import json
import pickle
from datetime import date

import numpy as np
import pandas as pd

import amount_split as asp
import pacm_v3 as pm
import timing_v3 as tm

NO_ASK_MCC = {"4511", "4722", "8011", "5812", "5814", "5311", "5411", "5651", "5712", "5732", "5200", "5942", "5331", "7832"}
OPTIONS = [("About a month", 30.4), ("In 3 months", 91.3), ("In 6 months", 182.6), ("Next year", 365.25)]
MONTHS = {"About a month": 1, "In 3 months": 3, "In 6 months": 6, "Next year": 12}
MAX_Q = 3


def fmt(o):
    return date.fromordinal(int(o)).strftime("%d %b %Y")


def verdict(x):
    active, paid = x.P >= 0.5, x.next_days.notna()
    date_ok = paid & (x.next_days >= x.t_p10) & (x.next_days <= x.t_p90)
    amt_ok = x.next_amount.notna() & (x.next_amount <= x.a_p90 * 1.0001)
    return np.select(
        [x.no_pred, x.detected & ~x.y_bill, ~x.detected & x.y_bill, ~x.detected & ~x.y_bill,
         active & ~paid, ~active & paid, ~active & ~paid, date_ok & amt_ok, ~date_ok & amt_ok, date_ok & ~amt_ok],
        ["WRONG: real bill has no prediction", "WRONG: not a bill in truth", "WRONG: real bill not detected", "correct: not a bill",
         "WRONG: said active, no payment", "WRONG: said inactive, but paid", "correct: inactive, no payment",
         "correct", "WRONG: date outside window", "WRONG: amount above P90"], "WRONG: date and amount")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v6")
    ap.add_argument("--res", default="results_v6")
    ap.add_argument("--today", default="payday", help="label of the model's per-customer 'today' (pacm_v3 --today)")
    a = ap.parse_args()
    st, ut, tx, day = pm.load_data(a.data)
    real = st.set_index("series_id")
    names = pm.payee_names(tx)
    cu = pm.tm.load(a.data, "customers")
    opened = dict(zip(cu.customer_id, pd.to_datetime(cu.first_account_open_date).map(lambda t: t.toordinal())))

    # ---------------- the model's output for today (unchanged)
    p = pd.read_csv(f"{a.res}/pacm_predictions.csv.gz", low_memory=False)
    x = p[(p.split == "test") & p.is_today.astype(bool)].copy()
    x["sid"], x["no_pred"], x["source"] = x.dom_sid, False, "model"
    SM = x.groupby("customer_id").snapshot.first().map(lambda v: date.fromisoformat(v).toordinal())    # each customer's "today"
    custs = set(x.customer_id)
    lab = tx[tx.is_exp & ~tx.gt_flags.fillna("").str.contains("duplicate_charge|split_part2")].sort_values("o")
    have = set(x.dom_sid.dropna())
    labc = lab[lab.customer_id.isin(custs)].copy()
    labc["cso"] = labc.customer_id.map(SM)
    nxt = labc[(labc.o >= labc.cso) & (labc.o < labc.cso + 60)].groupby("gt_series_id").first()
    n_before = labc[labc.o < labc.cso].groupby("gt_series_id").size()
    miss = []
    for sid, r in nxt.iterrows():
        if sid in have:
            continue
        miss.append({"customer_id": r.customer_id, "gkey": r.gkey, "sid": sid, "y_bill": True, "detected": False, "no_pred": True, "source": "none",
                     "next_days": r.o - r.cso, "next_amount": r.amount, "n_before": int(n_before.get(sid, 0)), "user_split": None})
    m = pd.DataFrame(miss)
    b = pd.concat([x, m], ignore_index=True)
    b["user_split"] = b.customer_id.map(dict(zip(ut.customer_id, ut.user_split)))
    b["so"] = b.customer_id.map(SM)
    b["today"] = b.so.map(lambda o: date.fromordinal(int(o)).isoformat())
    for c in ("P", "t_p10", "t_p50", "t_p90", "a_med", "a_p90", "C", "det_score"):
        b[c] = b[c].astype(float)
    b["detected"] = b.detected.astype(bool)
    b["y_bill"] = b.y_bill.astype(bool)
    b["no_pred"] = b.no_pred.astype(bool)
    b["verdict_before"] = verdict(b)
    b["payee"] = b.gkey.map(names)
    b["truth"] = np.where(b.y_bill, b.sid.map(real.name).fillna("?") + " [bill]", "not a bill")
    before = b.copy()
    Q = []
    prior = json.load(open(f"{a.res}/amount_split_prior.json"))
    tmodel = pickle.load(open(f"{a.res}/timing_model.pkl", "rb"))
    dg = {k: (g.o.to_numpy(), g.amount.to_numpy(float), g.txn_type.mode().iloc[0]) for k, g in day[day.gkey.str.split("#").str[0].isin(custs)].groupby("gkey")}

    def bill_of(k, o, amt):
        """the bill a payment to payee k belongs to, on the history up to the customer's today -> (segment = the model's
        cand, date of that bill's last payment before today); the split is amount_split, the same one the model used"""
        S = SM[k.split("#")[0]]
        if k not in dg:
            return 0, o
        do, da, cat = dg[k]
        h = (do >= S - asp.HIST) & (do < S)
        if k.split("#")[1][0] != "P" or h.sum() < 2:
            return 0, int(do[do < S].max()) if (do < S).any() else o
        seg = asp.labels(do[h], da[h], np.r_[da[h], amt], prior, tmodel, cat)
        mine = do[h][seg[:-1] == seg[-1]]
        return int(seg[-1]), int(mine.max()) if len(mine) else o

    # ---------------- moment 1: first payment to a new payee (whole history; the answer is remembered)
    firsts = tx.sort_values("o").groupby("gkey").head(1)
    firsts = firsts[(firsts.amount >= 1000) & ~firsts.gkey.str.contains(r"#T\|atm") & ~firsts.mcc.isin(NO_ASK_MCC)
                    & firsts.customer_id.isin(custs)]
    firsts = firsts[firsts.o < firsts.customer_id.map(SM)]
    # ask only when the model has no prediction for that bill today: the payment's bill = its segment of the payee's
    # split on today's history (amount_split, the same split the model used); predicted = that (payee, segment) called a bill
    predicted = set(zip(b[b.detected & ~b.no_pred].gkey, b[b.detected & ~b.no_pred].cand.astype(int)))
    bills = [bill_of(k, o, amt) for k, o, amt in zip(firsts.gkey, firsts.o, firsts.amount)]
    firsts = firsts.assign(seg=[x[0] for x in bills], last_o=[x[1] for x in bills])
    firsts = firsts[[(k, sg) not in predicted for k, sg in zip(firsts.gkey, firsts.seg)]]
    sid_o = {k: np.sort(g.to_numpy()) for k, g in lab.groupby("gt_series_id").o}
    for r in firsts.itertuples():
        sid = r.gt_series_id if isinstance(r.gt_series_id, str) and r.is_exp else None
        q = {"customer_id": r.customer_id, "moment": 1, "asked_on": fmt(r.o + 1), "payee": names.get(r.gkey, "?"), "amount": float(r.amount),
             "text": f"You paid ฿{r.amount:,.0f} to {names.get(r.gkey, '?')} on {fmt(r.o)}. When will you pay this again?",
             "options": [o for o, _ in OPTIONS] + ["Pick a date", "Not again"], "sid": sid, "this_month": bool(r.o >= SM[r.customer_id] - 30)}
        if sid is None:
            q.update(answer="Not again", effect="nothing reserved")
        else:
            per = real.true_period_days[sid]
            os_ = sid_o.get(sid, np.array([]))
            # the answer is kept; the next due date counts from this bill's last payment before today (not the first one)
            last = max(int(r.o), int(r.last_o))
            later = os_[os_ > last]
            if real.cycle_type[sid] == "calendar":
                opt, days = min(OPTIONS, key=lambda o: abs(np.log(o[1] / per)))
                # "about a month / 3 / 6 months / next year" = the same day of the month, 1 / 3 / 6 / 12 months later
                due, half = int(last + tm.month_gap(last, MONTHS[opt])), max(3, round(0.05 * days))
                period_days = days
                q.update(answer=opt)
            else:
                due = int(later[0]) if len(later) else int(last + round(per))
                half = 2
                period_days = float(per)
                q.update(answer=f"Pick a date: {fmt(due)}")
            q.update(effect=f"bill added: next {fmt(due)}, about ฿{r.amount:,.0f}", due=due, half=half)
            hit = (b.sid == sid) & b.no_pred
            if hit.any():
                i = b.index[hit][0]
                b.loc[i, ["detected", "no_pred", "P", "C", "source"]] = [True, False, 1.0, 0.9, "user (moment 1)"]
                Sc = SM[r.customer_id]
                b.loc[i, ["t_p50", "t_p10", "t_p90"]] = [due - Sc, due - half - Sc, due + half - Sc]
                b.loc[i, ["a_med", "a_p90", "median_gap"]] = [r.amount, r.amount * 1.1, period_days]
                q["bill_row"] = int(i)
        Q.append(q)

    # ---------------- moments 3-5 for today (model rows only), capped per customer by money at stake
    mrows = b[b.source == "model"]
    cand = []
    for i, r in mrows.iterrows():
        big = np.exp(r.log_amount) if r.log_amount == r.log_amount else 0
        if r.detected and r.P >= 0.5 and r.t_p50 <= 30 and r.amount_cv > 0.15 and big >= 1000:
            cand.append((r.customer_id, big, 3, i))
        if r.detected and r.overdue_ratio > 1.15 and 0.2 <= r.P < 0.95:
            cand.append((r.customer_id, big * (1 - abs(2 * r.P - 1)) + 1, 4, i))
        if 0.3 <= r.det_score < 0.7 and big >= 1000:
            cand.append((r.customer_id, big * (1 - abs(2 * r.det_score - 1)) + 1, 5, i))
    cq = pd.DataFrame(cand, columns=["customer_id", "stake", "moment", "row"]).sort_values("stake", ascending=False)
    cq["rank"] = cq.groupby("customer_id").cumcount()
    for r in cq.itertuples():
        row = b.loc[r.row]
        Sc = SM[r.customer_id]
        due = fmt(Sc + row.t_p50) if row.t_p50 == row.t_p50 else "?"
        base = {"customer_id": r.customer_id, "moment": int(r.moment), "asked_on": fmt(Sc), "payee": row.payee, "amount": float(np.exp(row.log_amount)),
                "bill_row": int(r.row), "this_month": True}
        if r.rank >= MAX_Q:
            Q.append({**base, "text": "(not asked: weekly question limit)", "answer": "", "effect": "", "skipped": True})
            continue
        paid = row.next_days == row.next_days and row.next_days <= 30
        if r.moment == 3:
            q = {**base, "text": f"Your {row.payee} bill is due around {due}. About how much this month?",
                 "options": ["Enter amount", "No bill this month", "This isn't a bill"]}
            if not row.y_bill:
                q.update(answer="This isn't a bill", effect="removed from bills")
                b.loc[r.row, "detected"] = False
            elif paid:
                q.update(answer=f"฿{row.next_amount:,.0f}", effect=f"amount set to ฿{row.next_amount:,.0f}")
                b.loc[r.row, ["a_med", "a_p90"]] = [row.next_amount, row.next_amount]
            else:
                q.update(answer="No bill this month", effect="no change (only tells it is not due this month)")
        elif r.moment == 4:
            q = {**base, "text": f"{row.payee} usually comes by now but hasn't yet. Is it still active?", "options": ["Still active", "Cancelled"]}
            if row.next_days == row.next_days:
                q.update(answer="Still active", effect="kept as active (P = 1)")
                b.loc[r.row, "P"] = 1.0
            else:
                q.update(answer="Cancelled", effect="marked inactive (P = 0)")
                b.loc[r.row, "P"] = 0.0
        else:
            q = {**base, "text": f"You pay about ฿{np.exp(row.log_amount):,.0f} to {row.payee} regularly. Is this a bill?", "options": ["Yes, a bill", "No"]}
            q.update(answer="Yes, a bill" if row.y_bill else "No", effect="treated as a bill" if row.y_bill else "not treated as a bill")
            b.loc[r.row, "detected"] = bool(row.y_bill)
        Q.append(q)

    # ---------------- labels: question + verdict before / after
    b["verdict_after"] = verdict(b)
    b["verdict_before"] = before.verdict_before
    for col, v in (("verdict_before", before), ("verdict_after", b)):
        nb = b.n_before.fillna(-1)
        lbl = np.where(nb == 0, " (first payment ever)", np.where(nb == 1, " (seen once)", " (hidden inside another payment group)"))
        b[col] = np.where(b[col] == "WRONG: real bill has no prediction", b[col] + lbl, b[col])
    qd = pd.DataFrame(Q)
    asked = qd[~qd.get("skipped", pd.Series(False, index=qd.index)).fillna(False).astype(bool) & qd.bill_row.notna()]
    b["asked"] = ""
    for r in asked.itertuples():
        b.loc[int(r.bill_row), "asked"] = f"Q{r.moment}: {r.text} -> {r.answer}"
    b["changed"] = b.verdict_before != b.verdict_after
    keep = b[(b.verdict_before != "correct: not a bill") | (b.verdict_after != "correct: not a bill")].copy()
    tf = lambda so, v: fmt(so + v) if v == v else ""
    keep["T"] = [tf(so, v) for so, v in zip(keep.so, keep.t_p50)]
    keep["T_window"] = [f"{tf(so, lo)} - {tf(so, hi)}" if lo == lo else "" for so, lo, hi in zip(keep.so, keep.t_p10, keep.t_p90)]
    keep["actual"] = [f"{fmt(so + d)} {am:,.0f}" if d == d else "no payment by 30 Sep 2026" for so, d, am in zip(keep.so, keep.next_days, keep.next_amount)]
    cols = ["customer_id", "user_split", "today", "payee", "truth", "source", "detected", "P", "T", "T_window", "a_p90", "C", "actual", "asked", "verdict_before", "verdict_after",
            "gkey", "median_gap", "t_p10", "t_p50", "next_days", "next_amount", "y_bill", "sid"]
    keep[cols].to_csv(f"{a.res}/hitl_bills_{a.today}.csv", index=False, encoding="utf-8-sig")
    qd.drop(columns=[c for c in ("due", "half") if c in qd]).to_csv(f"{a.res}/hitl_questions_{a.today}.csv", index=False, encoding="utf-8-sig")

    # ---------------- summary
    vb, va = keep.verdict_before.value_counts(), keep.verdict_after.value_counts()
    t = pd.DataFrame({"before": vb, "after": va}).fillna(0).astype(int)
    t["before %"] = (t.before / t.before.sum() * 100).round(1)
    t["after %"] = (t.after / t.after.sum() * 100).round(1)
    t = t.sort_values("before", ascending=False)
    pd.set_option("display.width", 220)
    print(f"== today = each customer's payday (+1 day): {len(custs):,} customers, {len(keep):,} bill rows\n" + t.to_string())
    real_q = qd[~qd.get("skipped", pd.Series(False, index=qd.index)).fillna(False).astype(bool)]
    months = (np.array([SM[c] for c in custs]) - np.array([opened[c] for c in custs])).clip(30) / 30.4
    print("\n== questions")
    for mo, g in real_q.groupby("moment"):
        useful = g.bill_row.notna() & g.bill_row.map(lambda i: bool(b.loc[int(i), "changed"]) if i == i else False)
        print(f"  moment {mo}: {len(g):,} asked | changed a bill's result: {int(useful.sum()):,} ({useful.mean():.0%})"
              + (f" | answered 'Not again' {int((g.answer == 'Not again').sum()):,}" if mo == 1 else ""))
    q1m = real_q[(real_q.moment == 1) & real_q.this_month]
    per_cust = (q1m.groupby("customer_id").size().reindex(list(custs), fill_value=0)
                + real_q[real_q.moment > 1].groupby("customer_id").size().reindex(list(custs), fill_value=0))
    print(f"  questions per customer this month: mean {per_cust.mean():.2f}, 90th pct {per_cust.quantile(.9):.0f}, max {per_cust.max()}; "
          f"customers asked anything: {(per_cust > 0).mean():.0%}; moment-1 rate over history {len(real_q[real_q.moment == 1]) / months.sum():.2f} per customer-month")
    ok_b = keep.verdict_before.str.startswith("correct").mean()
    ok_a = keep.verdict_after.str.startswith("correct").mean()
    print(f"\n  fully correct: {ok_b:.1%} -> {ok_a:.1%}")

    # ---------------- app page data
    data = {"today": a.today, "summary": t.reset_index().rename(columns={"index": "verdict"}).to_dict(orient="records"), "ok_before": ok_b, "ok_after": ok_a,
            "q_stats": {int(mo): {"asked": int(len(g)), "changed": int((g.bill_row.notna() & g.bill_row.map(lambda i: bool(b.loc[int(i), 'changed']) if i == i else False)).sum())}
                        for mo, g in real_q.groupby("moment")},
            "per_customer_mean": float(per_cust.mean())}
    show_c = sorted(set(real_q.customer_id), key=lambda c: -int((keep.customer_id == c).sum()))[:400]
    qq = qd[qd.customer_id.isin(show_c)].copy()
    qq = qq[(qq.moment > 1) | qq.this_month | qq.bill_row.notna()]
    data["questions"] = qq[["customer_id", "moment", "asked_on", "payee", "amount", "text", "options", "answer", "effect"]].replace({np.nan: None}).to_dict(orient="records")
    kb = keep[keep.customer_id.isin(show_c)][cols].replace({np.nan: None})
    data["bills"] = kb.to_dict(orient="records")
    data["customers"] = [{"id": c, "split": usp, "q": int((qq.customer_id == c).sum()), "changed": int(((keep.customer_id == c) & keep.changed).sum())}
                         for c, usp in zip(show_c, [dict(zip(ut.customer_id, ut.user_split))[c] for c in show_c])]
    data["customers"].sort(key=lambda d: -d["changed"])
    html = open("hitl_app_template.html", encoding="utf-8").read().replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=float).replace("</", "<\\/"))
    open(f"{a.res}/hitl_app.html", "w", encoding="utf-8").write(html)
    print(f"\nwrote {a.res}/hitl_app.html ({len(html) / 1e3:.0f} kB, {len(show_c)} customers)")


if __name__ == "__main__":
    main()
