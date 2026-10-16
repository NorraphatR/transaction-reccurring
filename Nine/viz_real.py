#!/usr/bin/env python3
"""
The v7b page (viz_v5_template.html) for the real statement: one person, so every payday is one entry in the picker.

Per payday (the day after a salary credit, history before it only):
  model only        the trained model, unchanged (predict_real.py)
  with answers      the questions the app asked by then, answered by the account owner (real_answers.json, real_validate.py rules)
  reserve           safe_v5 rule: max(A_P90 / floor(cycle) x (days to the estimated next payday + 1), A_P90 if due), P >= 0.5
  check             real bill payments from today to the REAL next payday (or the end of the statement), matched to the
                    reserve of the same bill (truth series), as in safe_v5; truth = real_data/truth_by_user.csv
Usage: python viz_real.py  -> results_real/safebelt_real.html
"""
import json
import os
import pickle
from datetime import date

import numpy as np
import pandas as pd

import pacm_v3 as pm
import safe_v5 as sv
import timing_v3 as tm
import viz_v5 as vz
from backtest_v2 import predict as b1_predict
from predict_real import real_groups
from real_validate import with_answers

OUT, RES = "results_real", os.environ.get("SAFEBELT_RES", "results_v7b")
fmt = lambda o: date.fromordinal(int(o)).strftime("%d %b %Y")


def verdict(r, end_o, S):
    paid = r["next_days"] == r["next_days"] and S + r["next_days"] <= end_o
    if r["detected"] and not r["y_bill"]:
        return "WRONG: not a bill in truth"
    if not r["detected"]:
        return "WRONG: real bill not detected" if r["y_bill"] else "correct: not a bill"
    active = r["P"] >= 0.5
    if active and not paid:
        return "WRONG: said active, no payment"
    if not active:
        return "WRONG: said inactive, but paid" if paid else "correct: inactive, no payment"
    t_ok = r["t_p10"] <= r["next_days"] <= r["t_p90"]
    a_ok = r["next_amount"] <= r["a_p90"] * 1.0001
    return "correct" if t_ok and a_ok else "WRONG: date outside window" if a_ok else "WRONG: amount above P90" if t_ok else "WRONG: date and amount"


def build(REAL, ROUT, WHO):
    p = pd.read_csv(f"{ROUT}/predictions_real.csv", encoding="utf-8-sig")
    ans = json.load(open(f"{ROUT}/real_answers.json", encoding="utf-8"))
    tx, day, truth = real_groups(REAL)
    names = pm.payee_names(tx)
    sname = dict(zip(truth.series_id, truth.name))
    full = tm.load(REAL, "transactions")
    full["o"] = (pd.to_datetime(full.posted_datetime).dt.normalize() - pd.Timestamp("1970-01-01")).dt.days + tm.EPOCH
    full = full.sort_values("posted_datetime")
    pay = full[(full.direction == "CR") & (full.txn_type == "payroll_in")]
    pay_o = np.sort(pay.o.unique())
    end = int(full.o.max())
    opened = int(full.o.min())
    prior = json.load(open(f"{RES}/amount_split_prior.json"))
    tmodel = pickle.load(open(f"{RES}/timing_model.pkl", "rb"))
    qs_by_date = {}
    for q in ans["app_questions"]:
        qs_by_date.setdefault(q["date"], []).append(q)

    OPTS = {1: ["About a month", "In 3 months", "In 6 months", "Next year", "Pick a date", "Not again"],
            3: ["About ฿1,000", "About ฿3,000", "About ฿8,500", "About ฿25,500", "About ฿6,400", "Nothing this month"], 4: ["Yes, about ฿1,000", "Yes, a bigger amount", "No, not this month", "Not a bill"],
            5: ["Yes, monthly bill", "No, not a bill"]}
    from datetime import datetime
    asked = [dict(q, o=datetime.strptime(q["asked"], "%d %b %Y").toordinal()) for q in ans["questions"]]
    custs, allrows = [], []
    prev_S = 0
    for snap, g in p.groupby("snapshot"):
        S = date.fromisoformat(snap).toordinal()
        # questions shown with this payday: asked since the previous payday (a first payment is asked the day it happens)
        q_here = [{"moment": q["moment"], "asked": q["asked"], "text": q["text"], "answer": q["answer"], "effect": q["effect"],
                   "options": OPTS[q["moment"]], "changed": q["effect"] != "nothing reserved"} for q in asked if prev_S < q["o"] <= S]
        prev_S = S
        est = sv.next_payday(pay_o, S)
        nxt, L = est if est else (S + 30, 30.4)
        left = nxt - S
        later = pay_o[pay_o >= S]
        real_end = int(later[0]) if len(later) else end + 1                 # check window: to the real next payday or the end of the data
        bal = float(full[full.o < S].balance_after.iloc[-1])
        sal = pay[(pay.o < S)].iloc[-1]
        versions = {"model": g[g.detected].assign(source="model"), "questions": with_answers(g, S, tx, ans)}
        money = {"today": date.fromordinal(S).isoformat(), "balance": bal, "salary_received": float(sal.amount), "salary_date": date.fromordinal(int(sal.o)).isoformat(),
                 "next_payday": date.fromordinal(nxt).isoformat(), "real_next_payday": date.fromordinal(real_end).isoformat() if len(later) else f"after {fmt(end)} (statement ends)",
                 "days_to_payday": left, "regular_salary": True}
        win = tx[(tx.o >= S) & (tx.o < real_end)]
        realb = win[win.sid.notna()]
        money["real_bills"] = float(realb.amount.sum())
        money["real_nonrecurring"] = float(win[win.sid.isna()].amount.sum())
        hist = tx[(tx.o >= S - 90) & (tx.o < S)]
        lines, shorts, billrows = {}, {}, {}
        for tag, det in versions.items():
            det = det[det.P >= 0.5]
            L_rows, reserve = [], {}
            for r in det.itertuples():
                per = max(1.0, float(np.floor(r.median_gap)))
                daily = r.a_p90 / per
                amt = daily * (left + 1)
                how = f"฿{r.a_p90:,.0f} every {per:.0f} days = ฿{daily:,.1f}/day × ({left} + 1) days = ฿{amt:,.0f}"
                if r.t_p10 < left and amt < r.a_p90:
                    amt, how = r.a_p90, f"due before payday: one full payment ฿{r.a_p90:,.0f}"
                L_rows.append({"payee": names.get(r.gkey, "?"), "kind": "repeat", "reserve": round(float(amt), 2), "how": how})
                sid = r.dom_sid if isinstance(r.dom_sid, str) else None
                if sid:
                    reserve[sid] = reserve.get(sid, 0.0) + float(amt)
            lines[tag] = L_rows
            up = sum(l["reserve"] for l in L_rows)
            spend = bal - up
            buf = 0.2 * max(spend, 0)
            S_rows, short = [], 0.0
            left_money = dict(reserve)
            for r in realb.sort_values("o").itertuples():
                cov = min(r.amount, left_money.get(r.sid, 0.0))
                left_money[r.sid] = left_money.get(r.sid, 0.0) - cov
                S_rows.append({"date": date.fromordinal(int(r.o)).isoformat(), "payee": names.get(r.gkey, "?"), "bill": sname[r.sid], "amount": float(r.amount),
                               "covered": float(cov), "short": float(r.amount - cov), "reserved_for_it": r.sid in reserve})
                short += r.amount - cov
            shorts[tag] = S_rows
            used = sum(x["covered"] for x in S_rows)
            money.update({f"upcoming_{tag}": up, f"spendable_{tag}": spend, f"buffer_{tag}": buf, f"safe_{tag}": spend - buf, f"bill_short_{tag}": short,
                          f"reserved_extra_{tag}": max(0.0, up - used), f"overall_{tag}": buf - short})
            # bill cards: every candidate that is detected or a real bill, plus real bills paid with no candidate
            rows = []
            cand = pd.concat([g[~g.gkey.isin(det.gkey)], det], ignore_index=True)
            for r in cand.to_dict(orient="records"):
                if not (r.get("detected") or r.get("y_bill")):
                    continue
                v = verdict(r, real_end - 1, S)
                nd = r["next_days"]
                rows.append({"payee": names.get(r["gkey"], "?"), "truth": f"{sname[r['dom_sid']]} [bill]" if isinstance(r.get("dom_sid"), str) else "not a bill",
                             "detected": bool(r.get("detected")), "P": r.get("P"), "T": fmt(S + r["t_p50"]), "T_window": f"{fmt(S + r['t_p10'])} - {fmt(S + min(r['t_p90'], 400))}",
                             "a_p90": r.get("a_p90"), "C": r.get("C"), "source": r.get("source", "model"),
                             "actual": f"{fmt(S + nd)} {r['next_amount']:,.0f}" if nd == nd and S + nd <= end else f"no payment by {fmt(end)}", "verdict": v,
                             "asked": "; ".join(f"Q{q['moment']} ({q['date']}): {q['answer']}" for q in ans["app_questions"]
                                               if q["gkey"] == r["gkey"] and date.fromisoformat(q["date"]).toordinal() < S + 1) or None})
            seen = set(cand[cand.detected.astype(bool) | cand.y_bill.astype(bool)].dom_sid.dropna())
            for sid, b in realb.groupby("sid"):
                if sid not in seen:
                    first = b.sort_values("o").iloc[0]
                    n_before = int((tx[(tx.sid == sid) & (tx.o < S)].gkey == first.gkey).sum())
                    rows.append({"payee": names.get(first.gkey, "?"), "truth": f"{sname[sid]} [bill]", "detected": False, "P": None,
                                 "actual": f"{fmt(first.o)} {first.amount:,.0f}", "verdict": "WRONG: real bill has no prediction (" + ("first payment ever" if n_before == 0 else f"seen {n_before}x in its group") + ")"})
            billrows[tag] = rows
            for x in rows:
                allrows.append({"snapshot": f"{WHO} · payday {fmt(S - 1)}", "version": tag, **{k: x.get(k) for k in ("verdict", "truth", "actual")}})
        # transactions (2 years back is the whole statement) and groups
        g_tx = tx[tx.o < real_end]
        gk = list(dict.fromkeys(g_tx.gkey))
        gi = {k: i for i, k in enumerate(gk)}
        det_m, det_q = set(versions["model"][versions["model"].P >= 0.5].gkey), set(versions["questions"][versions["questions"].P >= 0.5].gkey)
        groups = []
        for k in gk:
            hh = day[(day.gkey == k) & (day.o < S)]
            isp = k.split("#")[1].startswith("P")
            if isp and len(hh) >= 2:
                lv, cuts = vz.bill_split(hh.o.to_numpy(), hh.amount.to_numpy(float), prior, tmodel, hh.txn_type.mode().iloc[0])
            else:
                lv = vz.levels(hh.amount.values) if len(hh) >= 4 else []
                cuts = []
            groups.append({"payee": names.get(k, "?"), "type": k.split("#")[1][0], "model": k in det_m, "questions": k in det_q,
                           "truth": bool(tx[tx.gkey == k].sid.notna().any()), "n": int(len(hh)), "peaks": [{"amount": round(a, 2), "n": n} for a, n in lv],
                           "cuts": cuts, "amounts": [round(float(v), 2) for v in hh.amount.values[-120:]]})
        money["nonrec_month_avg"] = float(hist[~hist.gkey.isin(det_m)].amount.sum()) / 90 * 30.4
        money["nonrec_days"] = 90
        custs.append({"id": f"{WHO} · payday {fmt(S - 1)}", "split": "real", "hist": round((S - opened) / 30.4, 1), "groups": groups,
                      "txns": [[int(o - S), round(float(a), 2), gi[k]] for o, a, k in zip(g_tx.o, g_tx.amount, g_tx.gkey)],
                      "money": money, "lines": lines, "short": shorts, "bills": billrows, "questions": q_here})

    return custs, allrows, p.assign(who=WHO), len(ans["app_questions"])


SUF = os.environ.get("SAFEBELT_SUFFIX", "")          # "_v8": the v8 model's results for both people
PEOPLE = [("real_data", f"results_real{SUF}", "C00001 (me)"), ("real_data_2", f"results_real_2{SUF}", "C00002 (friend)")]


def main():
    custs, allrows, ps, nq = [], [], [], 0
    for real, rout, who in PEOPLE:
        c, a, p1, q = build(real, rout, who)
        custs += c; allrows += a; ps.append(p1); nq += q
    p = pd.concat(ps, ignore_index=True)
    # ---- section 1: correctness over all paydays
    A = pd.DataFrame(allrows)
    keep = A[A.verdict != "correct: not a bill"]
    vb = keep[keep.version == "model"].verdict.value_counts()
    va = keep[keep.version == "questions"].verdict.value_counts()
    vt = pd.DataFrame({"before": vb, "after": va}).fillna(0).astype(int).sort_values("before", ascending=False)

    def money_ok(df):
        amt = pd.to_numeric(df.actual.str.extract(r"(\d[\d,]*)$")[0].str.replace(",", ""), errors="coerce")
        m = df.truth.str.contains(r"\[bill", na=False) & amt.notna()
        return float(amt[m & df.verdict.str.startswith("correct")].sum() / max(1, amt[m].sum()))
    km, kq = keep[keep.version == "model"], keep[keep.version == "questions"]
    M = {}
    for tag in ("model", "questions"):
        ms = pd.DataFrame([c["money"] for c in custs])
        M[tag] = {"upcoming": ms[f"upcoming_{tag}"].mean(), "real": ms.real_bills.mean(), "short": ms[f"bill_short_{tag}"].mean(),
                  "short500": (ms[f"bill_short_{tag}"] > 500).mean(), "short2000": (ms[f"bill_short_{tag}"] > 2000).mean(), "extra": ms[f"reserved_extra_{tag}"].mean(),
                  "overall_short": (-ms[f"overall_{tag}"]).clip(lower=0).mean(), "overall_neg": (ms[f"overall_{tag}"] < 0).mean(), "overall_2000": (ms[f"overall_{tag}"] < -2000).mean()}
    nm = pd.Series([c["money"]["nonrec_month_avg"] for c in custs])

    # ---- section 2: P A T C on all paydays (model only)
    tp, fp, fn = int((p.detected & p.y_bill).sum()), int((p.detected & ~p.y_bill).sum()), int((~p.detected & p.y_bill).sum())
    ev = p[p.detected & p.y_bill & p.next_days.notna()].copy()
    b = ev[["snapshot", "last_date", "median_gap", "since_days"]].rename(columns={"median_gap": "median_gap_prior", "since_days": "days_since_last"}).copy()
    b["snapshot_date"], b["last_date"] = pd.to_datetime(b.snapshot), pd.to_datetime(b.last_date)
    b["amount_med3"] = b["last_amount"] = 0.0
    ev["b1_days"] = b1_predict(b).b1_days.values
    e, e1 = (ev.t_p50 - ev.next_days).abs(), (ev.b1_days - ev.next_days).abs()
    pe = p[p.detected & p.active_label.notna()]
    pacm = {"det": {"all": {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}}, "t_mae": float(e.mean()), "b1_mae": float(e1.mean()),
            "t_3d": float((e <= 3).mean()), "b1_3d": float((e1 <= 3).mean()), "in_win": float(((ev.next_days >= ev.t_p10) & (ev.next_days <= ev.t_p90)).mean()),
            "amt_ok": float((ev.next_amount <= ev.a_p90 * 1.0001).mean()), "ape": float(((ev.a_med - ev.next_amount).abs() / ev.next_amount).median()),
            "p_rel": vz.reliability(pe.P, pe.active_label), "c_rel": vz.reliability(pe.C, pe.active_label)}

    D = {"name": "real statements", "today": "the day after each payday (2 people, 9 paydays each)",
         "verdicts": [{"verdict": k, "before": int(r.before), "after": int(r.after)} for k, r in vt.iterrows()],
         "correct": {"model": float(km.verdict.str.startswith("correct").mean()), "questions": float(kq.verdict.str.startswith("correct").mean()),
                     "model_money": money_ok(km), "questions_money": money_ok(kq), "rows": int(len(km))},
         "by_age": [{"label": f"{c['id']}", "customers": 1, "bills": int(len(km[km.snapshot == s])),
                     "model": float(km[km.snapshot == s].verdict.str.startswith("correct").mean()), "questions": float(kq[kq.snapshot == s].verdict.str.startswith("correct").mean()),
                     "model_money": money_ok(km[km.snapshot == s]), "questions_money": money_ok(kq[kq.snapshot == s])}
                    for c, s in zip(custs, [c["id"] for c in custs])],
         "money": M, "nonrec_month": {"median": float(nm.median()), "mean": float(nm.mean()), "p10": float(nm.quantile(.1)), "p90": float(nm.quantile(.9))},
         "questions_per_customer": nq / len(custs), "pacm": pacm, "customers": custs}
    data = json.dumps(D, ensure_ascii=False, default=float).replace("</", "<" + "\\" + "/")
    html = open("viz_v5_template.html", encoding="utf-8").read().replace("/*DATA*/null", data)
    for a, b2 in [("SafeBelt v6", "SafeBelt · real statements" + (" · v8 model" if SUF else "")), ("synthetic data ${D.name}", "${D.name}"),
                  ("Bills fully correct by how long the customer has been with the bank", "Bills fully correct on each payday"),
                  ('History = months from the first account opening to today. Buckets are cumulative: "0 – 6 months" includes every customer with up to 6 months of history.',
                   "One row per payday: the model is run on the day after that salary, using only the history before it."),
                  ("<th>History</th><th>Customers</th>", "<th>Payday</th><th>Paydays</th>"),
                  ("${idx.length} customers", "${idx.length} paydays (both people)"), ('<label for="cust" class="note">Customer</label>', '<label for="cust" class="note">Payday</label>'),
                  ('<option value="id">customer ID (low → high)</option><option value="id_desc">customer ID (high → low)</option>', '<option value="id">payday (oldest first)</option><option value="id_desc">payday (newest first)</option>'),
                  ("Customers short > ฿", "Paydays short > ฿"), ("Customers overall short", "Paydays overall short"), ("(median customer)", "(median payday)"),
                  ("10% of customers below", "10% of paydays below"), ("(average per customer)", "(average per payday)"),
                  ('<span class="n">3</span>One customer</h2>', '<span class="n">3</span>One payday</h2>'),
                  ('${c.split === "heldout" ? "held-out" : "seen"} · ', ""),
                  ("Synthetic customers, each checked on their own payday.", "Two real bank statements (Nov 2025 – Oct 2026): mine (salary) and a friend's (no salary: the largest allowance each month counts as payday). Each is checked on the day after each of 9 paydays. Truth = the owner's own answers about which payees are bills; the questions are the ones the app asked at the time, answered by the owner (my friend's relayed by me). My friend's card purchases have no merchant names, so card bills cannot be seen or scored."),
                  ("about ${D.questions_per_customer.toFixed(1)} questions per customer this month", "about ${D.questions_per_customer.toFixed(1)} questions per payday")]:
        html = html.replace(a, b2)
    # payday ids sort by date, not by text
    html = html.replace('(a, b) => D.customers[b].id.localeCompare(D.customers[a].id) : (a, b) => D.customers[a].id.localeCompare(D.customers[b].id));',
                        '(a, b) => b - a : (a, b) => a - b);')
    # per payday: the questions the app asked and model only vs + questions
    card = """    <div class="grid2">
      <div class="card"><h3>Questions the app asked for this payday (your answers)</h3><div class="qlist" id="qcards"></div></div>
      <div class="card"><h3>This payday: model only vs + questions</h3><div class="tbl" id="qcompare"></div>
        <p class="note">Checked against what you really paid until the next salary and your truth table. Questions only add a bill the model had no prediction for (moment 1), keep a small very regular payment as a bill (moment 5b), or set the amount of a bill it was unsure about (moments 3 and 4).</p></div>
    </div>
"""
    html = html.replace('    <div class="card"><h3>Bills per payee', card + '    <div class="card"><h3>Bills per payee', 1)
    css = """.qlist { display: grid; gap: 10px; } .qcard { border: 1px solid var(--ring); border-radius: 8px; padding: 10px 12px; display: grid; gap: 6px; }
.qcard.changed { border-left: 4px solid var(--accent); } .qhead { display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
.qtag { font: 600 11.5px var(--mono); color: var(--accent); } .qtext { font-weight: 500; }
.qopts { display: flex; flex-wrap: wrap; gap: 6px; } .qopt { font-size: 12.5px; padding: 3px 9px; border-radius: 99px; border: 1px solid var(--ring); color: var(--muted); }
.qopt.on { background: var(--accent); border-color: var(--accent); color: var(--surface); font-weight: 600; }
</style>"""
    html = html.replace("</style>", css, 1)
    html = html.replace("function render() {", open("real_payday_panel.js", encoding="utf-8").read() + "\nfunction render() {", 1)
    html = html.replace("calc(c); shortfall(c); bills(c); }", "calc(c); shortfall(c); bills(c); paydayPanel(c); }", 1)
    open(f"{OUT}/safebelt_real{SUF}.html", "w", encoding="utf-8").write(html)
    print(f"wrote {OUT}/safebelt_real{SUF}.html ({len(html) / 1e6:.1f} MB, {len(custs)} paydays)")
    print(json.dumps({k: D["correct"][k] for k in D["correct"]}, indent=1), json.dumps(M, indent=1, default=float))


if __name__ == "__main__":
    main()
