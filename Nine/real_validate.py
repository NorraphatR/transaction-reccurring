#!/usr/bin/env python3
"""
Validate the model on the real statement at every payday, model only and with the owner's answers (app question rules),
against the account owner's truth table.

For each validation day (the day after a salary credit) the window runs to the next salary credit (or the end of the data):
  detected bills      called a bill by the model: right (a real bill) or wrong (not a bill, per the owner)
  missed bills        real bills paid in the window that the model did not call a bill
  T / A               for detected real bills paid in the window: date inside P10-P90, amount <= A_P90
  reserve / shortfall the safe_v5 reserve rule (P >= 0.5): max(A_P90 / floor(cycle) x (days to payday + 1), A_P90 if due);
                      real bill payments in the window are matched to the reserve of the SAME bill (truth series), so rent
                      paid to the new landlord is covered by the reserve made for the old one
Usage: python real_validate.py  -> results_real/real_validation.json
"""
import json
from datetime import date

import numpy as np
import pandas as pd

import pacm_v3 as pm
import timing_v3 as tm
from predict_real import real_groups

OUT = "results_real"
MONTHS = {"About a month": 1, "In 3 months": 3, "In 6 months": 6, "Next year": 12}


def with_answers(g, S, tx, ans):
    """the model's rows for one payday after the owner's answers (same rules as hitl_v5): a moment-1 answer adds the bill when
    the model has no row called a bill for that payee, next date = same day of month after that bill's last payment;
    a moment-4 / moment-3 amount answer sets A"""
    det = g[g.detected].copy()
    det["source"] = "model"
    for q in ans["app_questions"]:
        asked = date.fromisoformat(q["date"]).toordinal()
        if asked >= S or q["answer"] not in MONTHS:
            continue
        if q["moment"] == 1 and q["gkey"] not in set(det.gkey):
            h = tx[(tx.gkey == q["gkey"]) & (tx.o < S)]
            last = int(h.o.max())
            due = last
            while due < S - 3:
                due += tm.month_gap(due, MONTHS[q["answer"]])
            nxt = tx[(tx.gkey == q["gkey"]) & (tx.o >= S)].sort_values("o")
            row = {"gkey": q["gkey"], "dom_sid": h.sid.dropna().iloc[0] if h.sid.notna().any() else None, "y_bill": bool(h.sid.notna().any()),
                   "detected": True, "P": 1.0, "C": 0.9, "median_gap": 30.4 * MONTHS[q["answer"]], "t_p10": due - 3 - S, "t_p50": due - S, "t_p90": due + 3 - S,
                   "a_med": q["amount"], "a_p90": q["amount"] * 1.1, "next_days": (int(nxt.o.iloc[0]) - S) if len(nxt) else np.nan,
                   "next_amount": float(nxt.amount.iloc[0]) if len(nxt) else np.nan, "source": "you (moment 1)"}
            det = pd.concat([det, pd.DataFrame([row])], ignore_index=True)
    # moment 5b (small, very regular payment the model did not call a bill): "Yes" keeps it as a bill from that payday on,
    # with the model's own T and A
    for q in ans["app_questions"]:
        if q["moment"] == 5 and q["answer"].startswith("Yes") and date.fromisoformat(q["date"]).toordinal() <= S and q["gkey"] not in set(det.gkey):
            r = g[g.gkey == q["gkey"]]
            if len(r):
                det = pd.concat([det, r.assign(detected=True, source="you (moment 5b)")], ignore_index=True)
    for q in ans["app_questions"]:
        if q["moment"] in (3, 4) and q["date"] == date.fromordinal(S).isoformat() and "amount" in q:
            m = det.gkey == q["gkey"]
            det.loc[m, ["a_med", "a_p90", "source"]] = [q["amount"], q["amount"] * 1.1, f"you (moment {q['moment']})"]
    return det


def main():
    p = pd.read_csv(f"{OUT}/predictions_real.csv", encoding="utf-8-sig")
    tx, day, truth = real_groups("real_data")
    names = pm.payee_names(tx)
    full = tm.load("real_data", "transactions")
    full["o"] = (pd.to_datetime(full.posted_datetime).dt.normalize() - pd.Timestamp("1970-01-01")).dt.days + tm.EPOCH
    pay = np.sort(full[(full.direction == "CR") & (full.txn_type == "payroll_in")].o.unique())
    end = int(full.o.max())
    sname = dict(zip(truth.series_id, truth.name))
    ans = json.load(open(f"{OUT}/real_answers.json", encoding="utf-8"))
    result = {}
    for version in ("model", "questions"):
        result[version] = score(p, tx, names, pay, end, sname, ans, version)
    json.dump(result, open(f"{OUT}/real_validation.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    pd.set_option("display.width", 250)
    for version, res in result.items():
        out, tot = res["paydays"], res["total"]
        t = pd.DataFrame([{k: o[k] for k in ("label", "window_end", "detected", "right", "wrong", "n_missed", "t_ok", "a_ok", "n_eval",
                                             "reserved", "bills_paid", "shortfall", "reserved_not_bill")} for o in out])
        print(f"== {version}")
        print(t.round(0).to_string(index=False))
        print("total", {k: round(v) for k, v in tot.items()})


def score(p, tx, names, pay, end, sname, ans, version):
    out = []
    for snap, g in p.groupby("snapshot"):
        S = date.fromisoformat(snap).toordinal()
        later = pay[pay >= S]
        E = int(later[0]) if len(later) else min(end + 1, S + 30)          # next real payday, else the end of the data
        cut = E < end + 1 and len(later) > 0
        left = E - S
        bills = tx[(tx.o >= S) & (tx.o < E) & tx.sid.notna()]
        det = g[g.detected] if version == "model" else with_answers(g, S, tx, ans)
        rows = []
        reserve = {}
        for r in det.itertuples():
            per = max(1.0, float(np.floor(r.median_gap)))
            amt = r.a_p90 / per * (left + 1)
            if r.t_p10 < left and amt < r.a_p90:
                amt = r.a_p90
            amt = amt if r.P >= 0.5 else 0.0
            sid = r.dom_sid if isinstance(r.dom_sid, str) else None
            if sid:
                reserve[sid] = reserve.get(sid, 0.0) + amt
            paid_in = r.next_days == r.next_days and S + r.next_days < E
            rows.append({"payee": names.get(r.gkey, "?"), "source": getattr(r, "source", "model"), "bill": sname.get(sid, "not a bill"), "is_bill": bool(r.y_bill), "P": round(r.P, 2),
                         "t_p10": int(r.t_p10), "t_p50": int(r.t_p50), "t_p90": int(r.t_p90), "a_p90": round(r.a_p90), "C": round(r.C, 2), "reserve": round(amt),
                         "next": None if r.next_days != r.next_days else int(r.next_days), "next_amount": None if r.next_amount != r.next_amount else float(r.next_amount),
                         "t_ok": bool(paid_in and r.t_p10 <= r.next_days <= r.t_p90), "a_ok": bool(paid_in and r.next_amount <= r.a_p90 * 1.0001), "paid_in": bool(paid_in)})
        found = set(det.dom_sid.dropna())
        missed = []
        short = 0.0
        for sid, b in bills.groupby("sid"):
            need = float(b.amount.sum())
            have = reserve.get(sid, 0.0)
            short += max(0.0, need - have)
            if sid not in found:
                missed.append({"bill": sname[sid], "paid": need, "dates": " ".join(date.fromordinal(int(o)).strftime("%d %b") for o in b.o)})
        R = pd.DataFrame(rows)
        rb = R[R.is_bill] if len(R) else R
        ev = rb[rb.paid_in] if len(rb) else rb
        out.append({"snapshot": snap, "label": date.fromordinal(S - 1).strftime("%d %b"), "window_end": date.fromordinal(E).strftime("%d %b %Y"),
                    "window_complete": bool(cut), "days": left,
                    "detected": int(len(R)), "right": int(R.is_bill.sum()) if len(R) else 0, "wrong": int((~R.is_bill).sum()) if len(R) else 0,
                    "missed": missed, "n_missed": len(missed),
                    "t_ok": int(ev.t_ok.sum()) if len(ev) else 0, "a_ok": int(ev.a_ok.sum()) if len(ev) else 0, "n_eval": int(len(ev)),
                    "reserved": float(R.reserve.sum()) if len(R) else 0.0, "bills_paid": float(bills.amount.sum()), "shortfall": short,
                    "reserved_not_bill": float(R[~R.is_bill].reserve.sum()) if len(R) else 0.0, "rows": rows})
    tot = {k: sum(o[k] for o in out) for k in ("detected", "right", "wrong", "n_missed", "t_ok", "a_ok", "n_eval", "reserved", "bills_paid", "shortfall", "reserved_not_bill")}
    return {"paydays": out, "total": tot}


if __name__ == "__main__":
    main()
