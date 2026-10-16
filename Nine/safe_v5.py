#!/usr/bin/env python3
"""
Safe-to-Spend - a separate rule step after the model (pacm_v3.py) and the questions (hitl_v5.py). Changes neither.

Today = each customer's own validation day from the model (the day after their payday; see pacm_v3 --today payday).
Window = today .. next payday (from the customer's payroll credits: last payday + usual gap; no regular salary -> 30 days).

  Upcoming bills (detected bills with P >= 0.5), day unit for every bill (user decision 2026-10-14):
    max(A_P90 / floor(cycle days) x (days until next payday + 1),  A_P90 if due before payday)      weekly 120 x 30/7 -> 514; rent 6,500 x 30/30.4 -> 6,414; 6,000 / 91 d x 30 d -> 1,978
  Spendable      = balance today - upcoming bills
  Safety buffer  = 20% x spendable (0 when spendable is negative)          (user decision 2026-10-13)
  Safe to spend  = spendable - safety buffer

Check (truth table, today .. the REAL next payday): each real bill payment is covered by the reservation for the same bill (repeating
bills: the reserved payments; longer-cycle bills: the pot saved since their previous payment = A_P90); bills with no
reservation are not covered.

Usage: python safe_v5.py   -> results_v6/safe_to_spend_payday.csv (per customer), safe_lines_payday.csv, safe_short_payday.csv
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd

import pacm_v3 as pm
import timing_v3 as tm

BUFFER_SHARE = 0.2        # of spendable money
LONG = 1.5


def next_payday(pay_days, S):
    """(next payday, salary cycle days) from payroll credit dates before today, or None"""
    d = np.sort(pay_days[pay_days < S])
    if len(d) < 2:
        return None
    gap = float(np.median(np.diff(d[-6:])))
    if not 10 <= gap <= 40:
        return None
    nxt = d[-1] + round(gap)
    while nxt <= S:
        nxt += round(gap)
    return int(nxt), gap


def bill_lines(rows, S, cycle):
    lines = []
    for r in rows.itertuples():
        end, L = cycle[r.customer_id]
        left = end - S[r.customer_id]
        per = float(r.median_gap) if r.median_gap == r.median_gap and r.median_gap > 0 else 30.4
        per = max(1.0, float(np.floor(per)))             # whole days: 30.4 -> 30 (user decision 2026-10-14)
        daily = r.a_p90 / per
        # day unit for every bill (user decision 2026-10-14): A_P90 / cycle days x days until next payday.
        # Missed or overdue cycles cannot stack up; a weekly bill over 30 days -> ~4.3 payments, rent -> ~1.
        days = left + 1                                  # days until next payday + 1 (user decision 2026-10-14)
        amt = daily * days
        how = f"฿{r.a_p90:,.0f} every {per:.0f} days = ฿{daily:,.1f}/day × ({left} + 1) days = ฿{amt:,.0f}"
        # due before payday (T window starts before payday, overdue included) -> at least one full payment
        if r.t_p10 < left and amt < r.a_p90:
            amt = r.a_p90
            how = f"due before payday (T {r.t_p50:+.0f} days): one full payment ฿{amt:,.0f} (day unit would be ฿{daily * days:,.0f})"
        # "long" = longer than the salary cycle: the check treats its pot as saved since the previous payment
        lines.append((r.customer_id, r.gkey, r.payee, "long" if per > LONG * L else "repeat", per, round(days / per, 2), daily, r.a_p90, amt, how))
    return pd.DataFrame(lines, columns=["customer_id", "gkey", "payee", "kind", "period", "count", "daily", "a_p90", "reserve", "how"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v6")
    ap.add_argument("--res", default="results_v6")
    ap.add_argument("--today", default="payday", help="label used by pacm_v3 / hitl_v5 output files")
    a = ap.parse_args()
    st, ut, tx, day = pm.load_data(a.data)
    names = pm.payee_names(tx)
    full = pm.tm.load(a.data, "transactions", usecols=["customer_id", "account_id", "posted_datetime", "amount", "balance_after", "direction", "txn_type"])
    full["o"] = (pd.to_datetime(full.posted_datetime).dt.normalize() - pd.Timestamp("1970-01-01")).dt.days + pm.tm.EPOCH

    p = pd.read_csv(f"{a.res}/pacm_predictions.csv.gz", low_memory=False)
    p = p[(p.split == "test") & p.is_today.astype(bool)].copy()
    S = p.groupby("customer_id").snapshot.first().map(lambda v: date.fromisoformat(v).toordinal()).to_dict()
    custs = sorted(S)
    pays = {c: g.o.unique() for c, g in full[(full.direction == "CR") & (full.txn_type == "payroll_in")].groupby("customer_id")}
    cycle = {}
    for c in custs:
        np_ = next_payday(np.asarray(pays.get(c, [])), S[c])
        cycle[c] = np_ if np_ else (S[c] + 30, 30.4)
    full["cso"] = full.customer_id.map(S)
    # balance = the customer's main (spending) account: the one receiving salary, else the one with the most payments
    main = full[(full.direction == "CR") & (full.txn_type == "payroll_in")].groupby("customer_id").account_id.agg(lambda v: v.value_counts().index[0])
    busiest = full[full.direction == "DR"].groupby("customer_id").account_id.agg(lambda v: v.value_counts().index[0])
    main = busiest.to_dict() | main.to_dict()
    past = full[(full.o < full.cso) & (full.account_id == full.customer_id.map(main))]
    bal = past.sort_values(["customer_id", "posted_datetime"]).groupby("customer_id").balance_after.last()

    p["payee"] = p.gkey.map(names)
    cols = ["customer_id", "payee", "gkey", "a_p90", "t_p10", "t_p50", "median_gap", "sid"]
    p["sid"] = p.dom_sid
    model_rows = p[p.detected.astype(bool) & (p.P >= 0.5)][cols]
    h = pd.read_csv(f"{a.res}/hitl_bills_{a.today}.csv", encoding="utf-8-sig", low_memory=False)
    h = h[(h.detected.astype(str) == "True") & (pd.to_numeric(h.P, errors="coerce") >= 0.5)]
    q_rows = h[cols].copy()
    for c in ("a_p90", "t_p10", "t_p50", "median_gap"):
        q_rows[c] = pd.to_numeric(q_rows[c], errors="coerce")

    # average monthly non-recurring spending (model view): outgoing payments outside the groups the model calls bills,
    # own-account transfers excluded, last 3 months (or since the account opened)
    cu = pm.tm.load(a.data, "customers")
    opened = dict(zip(cu.customer_id, pd.to_datetime(cu.first_account_open_date).map(lambda t: t.toordinal())))
    bill_groups = set(p[p.detected.astype(bool)].gkey)
    txc = tx[tx.customer_id.isin(custs)].copy()
    txc["cso"] = txc.customer_id.map(S)
    hist = txc[(txc.o < txc.cso) & (txc.o >= txc.cso - 90) & ~txc.gkey.isin(bill_groups)]
    nonrec_sum = hist.groupby("customer_id").amount.sum().reindex(custs, fill_value=0)
    nonrec_days = pd.Series({c: max(30, min(90, S[c] - opened[c])) for c in custs})
    nonrec_month = nonrec_sum / nonrec_days * 30.4

    # the check ends at the REAL next payday (the next salary credit in the data); the app itself only knows the estimate.
    # Bills paid after the real payday belong to the next cycle and are paid from the new salary.
    real_end = {}
    for c in custs:
        later = np.sort(np.asarray(pays.get(c, [])))
        later = later[later >= S[c]]
        # use the real next salary credit only when it is a normal 10-40 days away (a job change or stopped salary leaves a
        # long gap; then the estimated payday is used, as the app does)
        real_end[c] = int(later[0]) if len(later) and 10 <= later[0] - S[c] + 1 <= 40 else cycle[c][0]
    fut = tx[tx.customer_id.isin(custs)].copy()
    fut = fut[(fut.o >= fut.customer_id.map(S)) & (fut.o < fut.customer_id.map(real_end))]
    real_bills = fut[fut.is_exp & ~fut.gt_flags.fillna("").str.contains("duplicate_charge")].sort_values("o")
    real_non = fut[~fut.is_exp].groupby("customer_id").amount.sum()

    out = pd.DataFrame(index=pd.Index(custs, name="customer_id"))
    out["today"] = [date.fromordinal(S[c]).isoformat() for c in custs]
    out["next_payday"] = [date.fromordinal(cycle[c][0]).isoformat() for c in custs]
    out["days_to_payday"] = [cycle[c][0] - S[c] for c in custs]
    out["real_next_payday"] = [date.fromordinal(real_end[c]).isoformat() for c in custs]
    out["regular_salary"] = [c in pays and next_payday(np.asarray(pays[c]), S[c]) is not None for c in custs]
    out["balance"] = bal.reindex(custs)
    # the salary that arrived on the latest payday before today (all payroll credits that day)
    prl = full[(full.direction == "CR") & (full.txn_type == "payroll_in") & (full.o < full.cso)]
    last_day = prl.groupby("customer_id").o.max()
    prl = prl[prl.o == prl.customer_id.map(last_day)]
    out["salary_received"] = prl.groupby("customer_id").amount.sum().reindex(custs)
    out["salary_date"] = last_day.reindex(custs).map(lambda o: date.fromordinal(int(o)).isoformat() if o == o else None)
    out["real_bills"] = real_bills.groupby("customer_id").amount.sum().reindex(custs, fill_value=0)
    out["real_nonrecurring"] = real_non.reindex(custs, fill_value=0)
    out["nonrec_month_avg"] = nonrec_month
    out["nonrec_days"] = nonrec_days
    all_lines, all_short = [], []
    for tag, rows in (("model", model_rows), ("questions", q_rows)):
        ln = bill_lines(rows, S, cycle)
        ln["sid"] = rows.sid.values                      # the real bill behind each reserved line (truth, for checking only)
        ln["version"] = tag
        all_lines.append(ln)
        out[f"upcoming_{tag}"] = ln.groupby("customer_id").reserve.sum().reindex(custs, fill_value=0)
        out[f"spendable_{tag}"] = out.balance - out[f"upcoming_{tag}"]
        out[f"buffer_{tag}"] = BUFFER_SHARE * out[f"spendable_{tag}"].clip(lower=0)
        out[f"safe_{tag}"] = out[f"spendable_{tag}"] - out[f"buffer_{tag}"]
        # match each real payment to the reserve made for the SAME BILL (not the same payee account), so a landlord's new
        # account or the other amount group of one person still counts as covered
        bill_key = lambda c, sid, g: (c, sid) if isinstance(sid, str) else (c, "group:" + str(g))
        reserved_keys = {bill_key(c, sid, g) for c, sid, g in zip(ln.customer_id, ln.sid, ln.gkey)}
        left_money = {}
        for l in ln.itertuples():
            k = bill_key(l.customer_id, l.sid, l.gkey)
            left_money[k] = left_money.get(k, 0.0) + (l.a_p90 if l.kind == "long" else l.reserve)
        cover = []
        for r in real_bills.itertuples():
            k = bill_key(r.customer_id, r.gt_series_id, r.gkey)
            c = min(r.amount, left_money.get(k, 0.0))
            if k in left_money:
                left_money[k] -= c
            cover.append(c)
        rb = real_bills.assign(covered=cover, version=tag)
        rb["short"] = rb.amount - rb.covered
        rb["payee"] = rb.gkey.map(names)
        rb["bill"] = rb.gt_series_id.map(st.set_index("series_id").name)
        rb["date"] = rb.o.map(lambda o: date.fromordinal(int(o)).isoformat())
        rb["reserved_for_it"] = [bill_key(c, sid, g) in reserved_keys for c, sid, g in zip(rb.customer_id, rb.gt_series_id, rb.gkey)]
        all_short.append(rb[["customer_id", "version", "date", "payee", "bill", "amount", "covered", "short", "reserved_for_it"]])
        out[f"bill_short_{tag}"] = rb.groupby("customer_id").short.sum().reindex(custs, fill_value=0)
        # overall = safety buffer - bill shortfall: positive = the buffer covered the missed bills (what is left of it),
        # negative = still short after using the whole buffer (the overall shortfall)            (user decision 2026-10-14)
        out[f"overall_{tag}"] = out[f"buffer_{tag}"] - out[f"bill_short_{tag}"]
        rep = ln[ln.kind == "repeat"].groupby("customer_id").reserve.sum().reindex(custs, fill_value=0)
        used = rb[rb.reserved_for_it].groupby("customer_id").covered.sum().reindex(custs, fill_value=0)
        out[f"reserved_extra_{tag}"] = (rep - used).clip(lower=0)
    out = out.reset_index()
    out["user_split"] = out.customer_id.map(dict(zip(ut.customer_id, ut.user_split)))
    out.round(2).to_csv(f"{a.res}/safe_to_spend_{a.today}.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_lines).round(2).to_csv(f"{a.res}/safe_lines_{a.today}.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_short).round(2).to_csv(f"{a.res}/safe_short_{a.today}.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    rows = []
    for tag in ("model", "questions"):
        rows.append({"version": "model only" if tag == "model" else "model + questions",
                     "upcoming bills until payday (avg)": f"{out[f'upcoming_{tag}'].mean():,.0f}",
                     "bills really paid until payday (avg)": f"{out.real_bills.mean():,.0f}",
                     "bill shortfall (avg)": f"{out[f'bill_short_{tag}'].mean():,.0f}",
                     "customers short > 500": f"{(out[f'bill_short_{tag}'] > 500).mean():.0%}",
                     "customers short > 2,000": f"{(out[f'bill_short_{tag}'] > 2000).mean():.0%}",
                     "reserved above need (avg)": f"{out[f'reserved_extra_{tag}'].mean():,.0f}",
                     "overall shortfall after buffer (avg)": f"{(-out[f'overall_{tag}']).clip(lower=0).mean():,.0f}",
                     "customers overall short > 0": f"{(out[f'overall_{tag}'] < 0).mean():.0%}",
                     "customers overall short > 2,000": f"{(out[f'overall_{tag}'] < -2000).mean():.0%}",
                     "safety buffer (median)": f"{out[f'buffer_{tag}'].median():,.0f}",
                     "safe to spend (median)": f"{out[f'safe_{tag}'].median():,.0f}",
                     "safe to spend < 0": f"{(out[f'safe_{tag}'] < 0).mean():.0%}",
                     "non-recurring spending per month (median)": f"{out.nonrec_month_avg.median():,.0f}"})
    print(f"== today = day after payday: {len(out):,} customers; regular salary {out.regular_salary.mean():.0%}; "
          f"days to next payday median {out.days_to_payday.median():.0f}")
    print(pd.DataFrame(rows).set_index("version").T.to_string())
    sh = pd.concat(all_short)
    sh = sh[sh.version == "model"]
    if sh.short.sum() > 0:
        print("\nshortfall (model only) by cause: no reservation for that bill "
              f"{sh[~sh.reserved_for_it].short.sum() / sh.short.sum():.0%}, reserved but not enough {sh[sh.reserved_for_it].short.sum() / sh.short.sum():.0%}")


if __name__ == "__main__":
    main()
