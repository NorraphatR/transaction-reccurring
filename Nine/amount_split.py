#!/usr/bin/env python3
"""
Bill split for payments to one payee (e.g. landlord: rent ~3,700 + utilities ~400) - Bayesian model selection.

Hypothesis "k bills": the payee's payments are k bills; each bill has its own amount level and its own payment schedule.
  1. amounts: the best way to cut the sorted log amounts into k contiguous segments (exact dynamic programming),
     each segment ~ Normal(mu, sigma^2) with a Normal-Inverse-Gamma prior; the spread prior is learned on training bills
  2. timing: each segment's dates are scored by the Bayesian survival model (timing_v3, the same prior the T model uses):
     first date uniform over the 24-month history, every gap from the posterior-predictive of the segment's own schedule
  3. score(k) = log p(amounts | k) + log p(dates | k) - penalty x (k - 1); the best k wins (k = 1..4)
Part payments (rent paid as 60% + 40% a few days apart) are joined into one payment before the test (merge_parts).
A single bill whose amount varies is not split: its pieces would each have an irregular schedule, so the timing term
drops. Rent + utilities is split: together the gaps look irregular, apart each is a clean monthly series.
New payments go to the segment with the highest posterior-predictive amount density x segment share.

Truth (gt_series_id) is used only here, on training customers, to fit the spread prior and choose the penalty
(best mean of: 1-bill payees kept whole, mixed payees separated - every payee counts equally).

Usage: python amount_split.py [--data data_v7] [--out results_v7b]  -> tuning table + amount_split_prior.json
"""
import argparse
import json
import os
from math import lgamma

import numpy as np
import pandas as pd

import timing_v3 as tm

K_MAX = 4
K0 = 0.05                      # prior strength of a segment mean (vague: m0 is only the payee median)
NU_GRID = (1, 2, 4, 8, 16, 32)
PENALTY_GRID = (-4, -2, 0, 2, 4, 8)
HIST = 730
PART_DAYS, PART_TOL = 10, np.log(1.03)   # payment in parts: within 10 days, sum within +-3% of the usual amount
_lg = np.vectorize(lgamma)


# ---------------------------------------------------------------- amounts
def seg_loglik(x, a0, b0, m0=None, k0=K0):
    """log marginal likelihood of every contiguous segment [i, j) of sorted x -> (n+1) x (n+1) matrix (-inf where j <= i)"""
    n = len(x)
    m0 = float(np.median(x)) if m0 is None else m0
    c1 = np.concatenate([[0], np.cumsum(x)])
    c2 = np.concatenate([[0], np.cumsum(x * x)])
    i, j = np.triu_indices(n + 1, 1)
    cnt = (j - i).astype(float)
    s1, s2 = c1[j] - c1[i], c2[j] - c2[i]
    mean = s1 / cnt
    ss = np.maximum(s2 - s1 * mean, 0)
    kn, an = k0 + cnt, a0 + cnt / 2
    bn = b0 + ss / 2 + k0 * cnt * (mean - m0) ** 2 / (2 * kn)
    ll = _lg(an) - lgamma(a0) + a0 * np.log(b0) - an * np.log(bn) + 0.5 * (np.log(k0) - np.log(kn)) - cnt / 2 * np.log(2 * np.pi)
    out = np.full((n + 1, n + 1), -np.inf)
    out[i, j] = ll
    return out


def amount_partitions(x, a0, b0):
    """sorted log amounts x -> {k: (amount log-ml, cut indices)} for the best k-segment cut, k = 1..K_MAX"""
    n = len(x)
    S = seg_loglik(x, a0, b0)
    best = S[0].copy()
    backs = []
    out = {1: (float(best[n]), [0, n])}
    for k in range(2, min(K_MAX, n) + 1):
        cand = best[:, None] + S
        arg = cand.argmax(0)
        best = cand[arg, np.arange(n + 1)]
        backs.append(arg)
        cuts, j = [n], n
        for b in backs[::-1]:
            j = int(b[j])
            cuts.append(j)
        out[k] = (float(best[n]), [0] + cuts[::-1])
    return out


# ---------------------------------------------------------------- timing
def timing_ml(model, o, cat):
    """log p(dates) of one bill: first date uniform over the history, gaps from the survival model's marginal likelihood"""
    o = np.sort(np.asarray(o))
    if len(o) < 2:
        return -np.log(HIST)
    keep, tm.MAX_GAPS = tm.MAX_GAPS, 10 ** 6            # score every gap, not only the last 12
    try:
        li, lc = model.loglik(o)
    finally:
        tm.MAX_GAPS = keep
    pi, pc = model.log_prior(cat)
    return float(np.logaddexp(tm.logsumexp((li + pi).ravel()), tm.logsumexp((lc + pc).ravel()))) - np.log(HIST)


# ---------------------------------------------------------------- split
def merge_parts(o, amounts, days=PART_DAYS, tol=PART_TOL):
    """one payment made in parts (split rent): two payments within `days` whose sum is the payee's usual amount
    (median of the other payments, +-tol) count as one payment on the first date - for the split decision only"""
    o, amt = np.asarray(o), np.asarray(amounts, float)
    keep = np.ones(len(o), bool)
    out = amt.copy()
    for i in range(len(o) - 1):
        if not keep[i] or o[i + 1] - o[i] > days:
            continue
        rest = np.delete(amt, [i, i + 1])
        if len(rest) and abs(np.log((amt[i] + amt[i + 1]) / np.median(rest))) < tol and max(amt[i], amt[i + 1]) < np.median(rest):
            out[i] = amt[i] + amt[i + 1]
            keep[i + 1] = False
    return o[keep], out[keep]


def split(o, amounts, prior, model, cat):
    """history dates + amounts -> segments [(n, mean, ss)] of the chosen hypothesis (low to high amount)"""
    o, amounts = merge_parts(o, amounts)
    la = np.log(np.maximum(np.asarray(amounts, float), 1))
    n = len(la)
    if n < 2:
        return [(n, float(la.mean()), 0.0)] if n else []
    order = np.argsort(la, kind="stable")
    x, oo = la[order], np.asarray(o)[order]
    best = None
    for k, (aml, cuts) in amount_partitions(x, prior["a0"], prior["b0"]).items():
        tml = sum(timing_ml(model, oo[cuts[s]:cuts[s + 1]], cat) for s in range(k))
        score = aml + tml - prior["penalty"] * (k - 1)
        if best is None or score > best[0]:
            best = (score, cuts)
    cuts = best[1]
    return [(cuts[s + 1] - cuts[s], float(x[cuts[s]:cuts[s + 1]].mean()),
             float(((x[cuts[s]:cuts[s + 1]] - x[cuts[s]:cuts[s + 1]].mean()) ** 2).sum())) for s in range(len(cuts) - 1)]


def assign(amounts, segs, prior):
    """segment index per amount: highest posterior-predictive (Student-t) density x segment share"""
    la = np.log(np.maximum(np.asarray(amounts, float), 1))
    if len(segs) <= 1:
        return np.zeros(len(la), int)
    a0, b0 = prior["a0"], prior["b0"]
    tot = sum(s[0] for s in segs)
    m0 = sum(s[0] * s[1] for s in segs) / tot
    dens = []
    for n, mean, ss in segs:
        kn, an = K0 + n, a0 + n / 2
        mn = (K0 * m0 + n * mean) / kn
        bn = b0 + ss / 2 + K0 * n * (mean - m0) ** 2 / (2 * kn)
        sc2 = bn * (kn + 1) / (an * kn)
        df = 2 * an
        dens.append(np.log(n / tot) + lgamma((df + 1) / 2) - lgamma(df / 2) - 0.5 * np.log(df * np.pi * sc2)
                    - (df + 1) / 2 * np.log1p((la - mn) ** 2 / sc2 / df))
    return np.argmax(np.array(dens), 0)


def labels(o_hist, amt_hist, amounts, prior, model, cat):
    """split the history, then put every amount (history and later) into a segment"""
    return assign(amounts, split(o_hist, amt_hist, prior, model, cat), prior)


# ---------------------------------------------------------------- learning (training customers only)
def fit_spread(series_logs):
    """(a0, b0) of the inverse-gamma spread prior by maximum marginal likelihood over true bill series (2-D grid).
    b0 has a floor of (1% in log)^2: many bills have one fixed amount, which would otherwise push the prior to zero spread"""
    best = None
    for nu in NU_GRID:
        for s0 in np.exp(np.linspace(np.log(1e-4), np.log(0.25), 25)):
            a0, b0 = nu / 2, nu * s0 / 2
            ll = sum(seg_loglik(np.sort(x), a0, b0)[0, len(x)] for x in series_logs)
            if best is None or ll > best[0]:
                best = (ll, a0, b0, s0)
    return {"a0": best[1], "b0": best[2], "s0": best[3]}


def fit_timing(data, st, usplit):
    """the T model's prior, fitted exactly as in pacm_v3 (train customers, dates before the train cut)"""
    from backtest_v2 import EXPENSE_KINDS
    from generate_v2 import SPLIT_TRAIN_END
    socc, scat = tm.occurrences(data)
    model = tm.Model()
    cut = SPLIT_TRAIN_END.toordinal()
    tr = st[st.kind.isin(EXPENSE_KINDS) & st.customer_id.isin([c for c, s in usplit.items() if s == "train"])]
    lls, cats = [], []
    for sid in tr.series_id:
        d = socc.get(sid)
        if d is not None and (d < cut).sum() >= 2:
            lls.append(model.loglik(d[d < cut]))
            cats.append(scat[sid])
    model.fit_prior(lls, cats)
    return model


def evaluate(groups, fn):
    """bill groups only. whole = 1-bill groups left in one piece; split = groups with 2+ bills whose bills end up in
    different segments (pairs of payments of different bills never share a segment); pair F1 over bill payments"""
    tp = fp = fn_ = 0
    whole = [0, 0]
    sep = [0, 0]
    for o, amt, lab, cat in groups:
        pred = fn(o, amt, cat)
        bill = ~pd.Series(lab).str.startswith("x_").to_numpy()
        p, l = pred[bill], lab[bill]
        same_p, same_t = p[:, None] == p[None, :], l[:, None] == l[None, :]
        iu = np.triu_indices(len(p), 1)
        a, b = same_p[iu], same_t[iu]
        tp, fp, fn_ = tp + int((a & b).sum()), fp + int((a & ~b).sum()), fn_ + int((~a & b).sum())
        if len(set(l)) == 1:
            whole[0] += len(set(p)) == 1
            whole[1] += 1
        else:
            sep[0] += not (a & ~b).any()
            sep[1] += 1
    return np.array([tp, fp, fn_, whole[0], whole[1], sep[0], sep[1]])


def summary(c):
    tp, fp, fn_, w0, w1, s0, s1 = c
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn_)
    return {"pair precision": pr, "pair recall": rc, "pair F1": 2 * pr * rc / max(1e-9, pr + rc),
            "1-bill groups kept whole": w0 / max(1, w1), "2+-bill groups separated": s0 / max(1, s1), "n 1-bill": w1, "n 2+-bill": s1}


def main():
    from multiprocessing import Pool
    from pacm_v3 import load_data, clusters
    from generate_v2 import SPLIT_TRAIN_END
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v7")
    ap.add_argument("--out", default="results_v7b")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st, ut, tx, day = load_data(a.data)
    usplit = dict(zip(ut.customer_id, ut.user_split))
    model = fit_timing(a.data, st, usplit)
    cut = SPLIT_TRAIN_END.toordinal()
    day = day[day.gkey.str.split("#").str[1].str[0].eq("P") & (day.o < cut) & (day.o >= cut - HIST)].copy()
    day = day[day.gkey.str.split("#").str[0].map(usplit).eq("train")]
    # a bill payment's label is its true series; everything else is labelled x_ (not scored)
    day["lab"] = np.where(day.is_exp, day.sid.astype(str), "x_" + day.index.astype(str))
    logs = [np.log(np.maximum(g.amount.to_numpy(float), 1)) for _, g in day[day.is_exp].groupby("sid")]
    prior = fit_spread(logs)
    print(f"spread prior from {len(logs):,} train bills: a0={prior['a0']:.2f} b0={prior['b0']:.5f} "
          f"(typical bill amount spread {np.sqrt(prior['s0']):.3f} in log = +-{np.expm1(np.sqrt(prior['s0'])):.0%})", flush=True)
    groups = [(g.o.to_numpy(), g.amount.to_numpy(float), g.lab.to_numpy(object), g.txn_type.mode().iloc[0])
              for _, g in day.groupby("gkey") if len(g) >= 3 and g.is_exp.any()]
    methods = [("no split", None), ("amount peaks (current)", "peaks")] + [(f"bayes split, penalty {p}", p) for p in PENALTY_GRID]
    chunks = [groups[i::a.workers * 4] for i in range(a.workers * 4)]
    with Pool(a.workers, initializer=_init, initargs=(prior, model)) as pool:
        res = pool.map(_run, [(m, ch) for _, m in methods for ch in chunks])
    nc = len(chunks)
    tab = pd.DataFrame([summary(sum(res[i * nc:(i + 1) * nc])) for i in range(len(methods))], index=[n for n, _ in methods])
    pd.set_option("display.width", 200)
    print(tab.to_string(float_format="{:.3f}".format))
    bay = tab[tab.index.str.startswith("bayes")]
    # every payee counts equally (pair F1 is dominated by payees with many payments)
    bal = (bay["1-bill groups kept whole"] + bay["2+-bill groups separated"]) / 2
    prior["penalty"] = PENALTY_GRID[int(bal.to_numpy().argmax())]
    with open(f"{a.out}/amount_split_prior.json", "w") as f:
        json.dump(prior, f, indent=1)
    print(f"chosen penalty {prior['penalty']} (best mean of the two payee rates on train); saved {a.out}/amount_split_prior.json")


_CTX = None


def _init(prior, model):
    global _CTX
    _CTX = (prior, model)


def _run(task):
    from pacm_v3 import clusters
    m, groups = task
    prior, model = _CTX
    if m is None:
        fn = lambda o, amt, cat: np.zeros(len(amt), int)
    elif m == "peaks":
        fn = lambda o, amt, cat: np.searchsorted(clusters(amt), amt)
    else:
        p = dict(prior, penalty=m)
        fn = lambda o, amt, cat: labels(o, amt, amt, p, model, cat)
    return evaluate(groups, fn)


if __name__ == "__main__":
    main()
