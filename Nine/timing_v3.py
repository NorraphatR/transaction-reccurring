#!/usr/bin/env python3
"""
SafeBelt timing model v3 - dynamic-period Bayesian survival (grid posterior, no MCMC)
=====================================================================================

Hypotheses per series (all on fixed grids):
  interval  free period theta (3-400 d, log grid) x sigma (gap spread, log scale) x eps (skip rate)
  calendar  spikes 7 / 14 days and 1 / 3 / 6 / 12 months x tau (day noise) x eps
Prior = calendar spikes + free smooth part; spike weights, free-part shape, sigma / tau / eps priors are
learned per category (series txn_type, a standard field) by EM on train gaps only (train customers, < 2025-10-01).

Fix 1 (day-of-month evidence): month spikes are month-anchored - the expected gap is the calendar distance
to the same day of month k months later (28-31 days for monthly), not a fixed 30.4 days. A 28-day interval
series drifts ~2.4 days of month per cycle, which accumulates as residuals against every month hypothesis.
Fix 2 (skipped cycles): each gap is a mixture over m = 1, 2, 3 cycles with P(m) = (1-eps) eps^(m-1).

Prediction at a snapshot: predictive pmf of the next gap, conditioned on no event since the last occurrence
(survival); point = median, window = P10-P90.

Series grouping uses gt_series_id (oracle series builder, task 4 replaces it); only dates before the snapshot
are used. Evaluation: official test split, events observed, vs baseline B1. Synthetic data - not evidence.

Usage: python timing_v3.py --data data_v3 --out results_v3 [--no-dom] [--no-skip] [--limit N]
"""
import argparse
import os
import time
from calendar import monthrange
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from backtest_v2 import EXPENSE_KINDS, predict as baseline_predict
from generate_v2 import SPLIT_TRAIN_END

THETA = np.exp(np.linspace(np.log(3), np.log(400), 160))
LT = np.log(THETA)
SIG = np.array([0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.32, 0.45])
TAU = np.array([0.7, 1.5, 3.0, 6.0])
M = np.array([1, 2, 3])
ABS_NOISE = 0.6          # days: integer dates + card posting delay
P_OUT, D = 0.005, 900    # outlier mass (uniform over D days); predictive horizon in days
MAX_GAPS = 12
EPOCH = date(1970, 1, 1).toordinal()


def spikes(dom):
    if dom:
        return [("day", 7), ("day", 14), ("month", 1), ("month", 3), ("month", 6), ("month", 12)]
    return [("day", 7), ("day", 14), ("day", 30.4), ("day", 91.3), ("day", 182.6), ("day", 365.25)]   # ablation: fixed-length spikes


@lru_cache(maxsize=None)
def month_gap(o, k):
    """days from date ordinal o to the same day of month k months later (clipped to month end)"""
    d = date.fromordinal(o)
    y, m = divmod(d.month - 1 + k, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, monthrange(y, m)[1])).toordinal() - o


def logsumexp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0)
    return np.squeeze(np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m, axis=axis)


def lnorm(x, mu, s):
    return -0.5 * ((x - mu) / s) ** 2 - np.log(s) - 0.9189385332


class Model:
    def __init__(self, dom=True, skip=True):
        self.spk = spikes(dom)
        self.eps = np.array([0.005, 0.05, 0.15, 0.30]) if skip else np.array([1e-9])
        pm = (1 - self.eps[:, None]) * self.eps[:, None] ** (M[None, :] - 1)
        self.lpm = np.log(pm / pm.sum(1, keepdims=True))                                    # (E, M)
        # interval spread per (theta, sigma, m): CV of a sum of m gaps shrinks by sqrt(m); absolute noise added
        self.s_int = np.sqrt(SIG[None, :, None] ** 2 / M[None, None, :] + (ABS_NOISE / (THETA[:, None, None] * M[None, None, :])) ** 2)
        self.prior = {}
        self.max_cells = 400                # posterior cells kept for the predictive (speed vs detail)

    # ------------------------------------------------ likelihood of observed gaps
    def mix(self, ld):
        """ld: (..., M) log density per number of cycles -> (..., E) with skip mixture and outlier"""
        l = logsumexp(ld[..., None, :] + self.lpm, axis=-1)
        return np.logaddexp(np.log(1 - P_OUT) + l, np.log(P_OUT / D))

    def loglik(self, d):
        """d: occurrence ordinals (sorted). Returns (interval LL (T,S,E), calendar LL (K,TAU,E))."""
        g = np.diff(d)[-MAX_GAPS:].astype(float)
        prev = d[:-1][-MAX_GAPS:]
        E = len(self.eps)
        if len(g) == 0:
            return np.zeros((len(THETA), len(SIG), E)), np.zeros((len(self.spk), len(TAU), E))
        g = np.maximum(g, 0.5)
        lg = np.log(g)[:, None, None, None]
        li = lnorm(lg, LT[None, :, None, None] + np.log(M), self.s_int[None]) - lg            # (n, T, S, M)
        li = self.mix(li).sum(0)
        lc = []
        for unit, k in self.spk:
            mu = np.array([[month_gap(int(p), k * m) for m in M] for p in prev], float) if unit == "month" else np.outer(np.ones(len(g)), k * M)
            l = lnorm(g[:, None, None], mu[:, None, :], TAU[None, :, None])                  # (n, TAU, M)
            lc.append(self.mix(l).sum(0))
        return li, np.stack(lc)

    # ------------------------------------------------ prior
    def log_prior(self, cat):
        p = self.prior.get(cat, self.prior["_all"])
        li = p["w"][-1] + p["f"][:, None, None] + p["sig"][None, :, None] + p["eps"][None, None, :]
        lc = p["w"][:-1, None, None] + p["tau"][None, :, None] + p["eps"][None, None, :]
        return li, lc

    def fit_prior(self, lls, cats, iters=20, alpha=30.0):
        """Empirical Bayes by EM: lls = list of (LLi, LLc); cats = category per series."""
        K, E = len(self.spk), len(self.eps)
        uni = lambda n: np.full(n, -np.log(n))
        init = {"w": np.log(np.r_[np.full(K, 0.5 / K), 0.5]), "f": uni(len(THETA)), "sig": uni(len(SIG)), "tau": uni(len(TAU)), "eps": uni(E)}
        self.prior = {"_all": dict(init)}
        cats = np.asarray(cats)
        LI = np.stack([a for a, _ in lls]).astype(np.float32)
        LC = np.stack([b for _, b in lls]).astype(np.float32)
        groups = {c: np.where(cats == c)[0] for c in np.unique(cats)}
        kern = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
        kern /= kern.sum()
        for it in range(iters):
            stats = {}
            for c, idx in list(groups.items()) + [("_all", np.arange(len(cats)))]:
                pi, pc = self.log_prior(c)
                a, b = LI[idx] + pi, LC[idx] + pc
                z = np.logaddexp(logsumexp(a.reshape(len(idx), -1), 1), logsumexp(b.reshape(len(idx), -1), 1))
                ri, rc = np.exp(a - z[:, None, None, None]), np.exp(b - z[:, None, None, None])
                stats[c] = {"w": np.r_[rc.sum((0, 2, 3)), ri.sum()], "f": ri.sum((0, 2, 3)), "sig": ri.sum((0, 1, 3)),
                            "tau": rc.sum((0, 1, 3)), "eps": ri.sum((0, 1, 2)) + rc.sum((0, 1, 2))}
            new = {}
            g = stats["_all"]
            gp = {k: v / v.sum() for k, v in g.items()}
            for c, s in stats.items():
                q = {}
                for k, v in s.items():
                    v = v + (alpha * gp[k] if c != "_all" else 1e-3)
                    if k == "f":
                        v = np.convolve(v / v.sum(), kern, mode="same") + 1e-4 / len(v)
                    q[k] = np.log(v / v.sum())
                new[c] = q
            self.prior = new
        return {c: len(i) for c, i in groups.items()}

    # ------------------------------------------------ prediction
    def predictive(self, d, cat, ll=None):
        """pmf over next gap t = 0..D-1 (t days after the last occurrence), unconditioned on survival."""
        li, lc = ll if ll is not None else self.loglik(d)
        pi, pc = self.log_prior(cat)
        a, b = li + pi, lc + pc
        flat = np.r_[a.ravel(), b.ravel()]
        flat = np.exp(flat - flat.max())
        flat /= flat.sum()
        order = np.argsort(-flat)
        keep = order[: max(1, min(self.max_cells, int(np.searchsorted(np.cumsum(flat[order]), 0.995)) + 1))]
        t = np.arange(1, D, dtype=float)
        L = int(d[-1])
        dom = np.array([date.fromordinal(int(x)).day for x in d[-6:]])
        off = float(np.median((dom - dom[-1] + 15) % 31 - 15))                                # typical day of month vs last occurrence
        ni = a.size
        pm = np.exp(self.lpm)
        pmf = np.zeros(D - 1)
        ii, ic = keep[keep < ni], keep[keep >= ni] - ni
        if len(ii):
            ti, si, ei = np.unravel_index(ii, a.shape)
            s = self.s_int[ti, si]                                                           # (C, M)
            mu = LT[ti][:, None] + np.log(M)[None, :]
            dens = np.exp(lnorm(np.log(t)[None, None, :], mu[:, :, None], s[:, :, None])) / t
            dens = (dens / np.maximum(dens.sum(2, keepdims=True), 1e-300) * pm[ei][:, :, None]).sum(1)
            pmf += (flat[ii][:, None] * dens).sum(0)
        if len(ic):
            ki, tau_i, ei = np.unravel_index(ic, b.shape)
            for j in np.unique(ki):
                sel = ki == j
                unit, k = self.spk[j]
                mu = np.array([month_gap(L, k * m) + off for m in M]) if unit == "month" else k * M
                dens = np.exp(lnorm(t[None, None, :], mu[None, :, None], TAU[tau_i[sel]][:, None, None]))
                dens = (dens / np.maximum(dens.sum(2, keepdims=True), 1e-300) * pm[ei[sel]][:, :, None]).sum(1)
                pmf += (flat[ic[sel] + ni][:, None] * dens).sum(0)
        pmf = (1 - P_OUT) * pmf / pmf.sum() + P_OUT / (D - 1)
        return np.r_[0.0, pmf]

    @staticmethod
    def quantiles(pmf, elapsed, qs=(0.1, 0.5, 0.9)):
        """condition on no event in days 1..elapsed-1 (next >= elapsed), return quantiles of days from snapshot"""
        p = pmf.copy()
        p[: elapsed] = 0
        if p.sum() <= 0:
            return [0.0] * len(qs)
        c = np.cumsum(p / p.sum())
        return [float(np.searchsorted(c, q) - elapsed) for q in qs]


# ---------------------------------------------------------------- data
def load(d, name, **kw):
    f = f"{d}/{name}.csv.gz" if os.path.exists(f"{d}/{name}.csv.gz") else f"{d}/{name}.csv"
    return pd.read_csv(f, encoding="utf-8-sig", **kw)


def occurrences(data):
    tx = load(data, "transactions", usecols=["txn_id", "posted_datetime", "txn_type"])
    lb = load(data, "txn_labels")
    m = lb[lb.gt_series_id.notna()].merge(tx, on="txn_id")
    m = m[~m.gt_flags.fillna("").str.contains("duplicate_charge|split_part2")]    # same occurrence definition as the snapshots
    m["o"] = (pd.to_datetime(m.posted_datetime).dt.normalize() - pd.Timestamp("1970-01-01")).dt.days + EPOCH
    m = m.sort_values(["gt_series_id", "o"])
    occ = {s: g.o.to_numpy() for s, g in m.groupby("gt_series_id")}
    cat = m.groupby("gt_series_id").txn_type.last().to_dict()
    return occ, cat


def bucket_period(p):
    return pd.cut(p, [0, 10, 25, 35, 70, 120, 250, 1000], labels=["<=10d", "11-25d", "26-35d (monthly)", "36-70d", "71-120d (quarterly)",
                                                                  "121-250d (semiannual)", ">250d (annual)"])


def metrics(g):
    out = {"n": len(g)}
    for m in ("b1", "bs"):
        e = (g[f"{m}_days"] - g.duration_days).abs()
        out.update({f"{m}_MAE": e.mean(), f"{m}_3d": (e <= 3).mean(), f"{m}_7d": (e <= 7).mean()})
    out["bs_cover80"] = ((g.bs_p10 <= g.duration_days) & (g.duration_days <= g.bs_p90)).mean()
    out["bs_width"] = (g.bs_p90 - g.bs_p10).mean()
    return pd.Series(out)


def fmt(t):
    t = t.copy()
    for c in t.columns:
        if c == "n":
            t[c] = t[c].astype(int)
        elif c.endswith(("MAE", "width")):
            t[c] = t[c].map("{:.1f}".format)
        else:
            t[c] = t[c].map("{:.0%}".format)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v3")
    ap.add_argument("--out", default="results_v3")
    ap.add_argument("--no-dom", action="store_true", help="ablation: drop fix 1 (fixed-length spikes, no day-of-month)")
    ap.add_argument("--no-skip", action="store_true", help="ablation: drop fix 2 (no skipped-cycle mixture)")
    ap.add_argument("--limit", type=int, default=0, help="score a random subset of test events (speed)")
    a = ap.parse_args()
    tag = "full" + ("_nodom" if a.no_dom else "") + ("_noskip" if a.no_skip else "")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    occ, cat = occurrences(a.data)
    st = load(a.data, "series_truth")
    ut = load(a.data, "users_truth")
    model = Model(dom=not a.no_dom, skip=not a.no_skip)

    # ---- fit prior on train customers, gaps before the train cutoff only
    cut = pd.Timestamp(SPLIT_TRAIN_END).date().toordinal()
    train_users = set(ut.customer_id[ut.user_split == "train"])
    tr = st[st.kind.isin(EXPENSE_KINDS) & st.customer_id.isin(train_users)]
    lls, cats = [], []
    for sid in tr.series_id:
        d = occ.get(sid)
        if d is None:
            continue
        d = d[d < cut]
        if len(d) >= 2:
            lls.append(model.loglik(d))
            cats.append(cat[sid])
    n_cat = model.fit_prior(lls, cats)
    print(f"[{tag}] prior fitted on {len(lls):,} train series {n_cat} in {time.time() - t0:.0f}s")
    for c in sorted(n_cat):
        w = np.exp(model.prior[c]["w"])
        print(f"   {c:14s} spikes " + " ".join(f"{k}{u[0]}={x:.2f}" for (u, k), x in zip(model.spk, w[:-1])) + f" free={w[-1]:.2f}")

    # ---- test snapshots
    te = load(a.data, "snapshots_test", parse_dates=["snapshot_date", "last_date", "next_date"])
    te = te[te.truth_kind.isin(EXPENSE_KINDS) & (te.event_observed == 1)].reset_index(drop=True)
    if a.limit:
        te = te.sample(min(a.limit, len(te)), random_state=0).reset_index(drop=True)
    te = baseline_predict(te)
    snap_o = ((te.snapshot_date - pd.Timestamp("1970-01-01")).dt.days + EPOCH).to_numpy()
    cache, res, bad = {}, np.zeros((len(te), 3)), 0
    for i, (sid, so, n_prior) in enumerate(zip(te.series_id, snap_o, te.n_prior)):
        d = occ[sid]
        d = d[d < so]
        bad += len(d) != n_prior
        key = (sid, len(d))
        if key not in cache:
            cache = {key: model.predictive(d, cat[sid])} if len(cache) > 50 else {**cache, key: model.predictive(d, cat[sid])}
        res[i] = model.quantiles(cache[key], int(so - d[-1]))
        if (i + 1) % 5000 == 0:
            print(f"   {i + 1:,}/{len(te):,} scored ({time.time() - t0:.0f}s)", flush=True)
    assert bad == 0, f"{bad} snapshots with n_prior mismatch"
    te["bs_p10"], te["bs_days"], te["bs_p90"] = res[:, 0], res[:, 1], res[:, 2]
    te["hist"] = pd.cut(te.n_prior, [0, 1, 3, 6, 1000], labels=["1", "2-3", "4-6", "7+"])
    te["period"] = bucket_period(te.truth_period_days)

    tables = {"overall": metrics(te).to_frame("all").T,
              "by cycle_type": te.groupby("truth_cycle_type").apply(metrics, include_groups=False),
              "by true period": te.groupby("period", observed=True).apply(metrics, include_groups=False),
              "by n_prior": te.groupby("hist", observed=True).apply(metrics, include_groups=False),
              "by customer split": te.groupby("user_split").apply(metrics, include_groups=False),
              "interval series by true period (fix 1 check)": te[te.truth_cycle_type == "interval"].groupby(
                  pd.cut(te.truth_period_days, [0, 10, 20, 27, 31, 45, 60, 100]), observed=True).apply(metrics, include_groups=False)}
    mon = te[(te.truth_cycle_type == "calendar") & (te.truth_period_days == 30.4)]
    ann = te[te.truth_period_days == 365.25]
    once = te[te.n_prior == 1]
    targets = pd.DataFrame([
        ("seen-once MAE (days)", "60", f"{(once.b1_days - once.duration_days).abs().mean():.1f}", f"{(once.bs_days - once.duration_days).abs().mean():.1f}"),
        ("annual within 7 days", "40%", f"{((ann.b1_days - ann.duration_days).abs() <= 7).mean():.0%}", f"{((ann.bs_days - ann.duration_days).abs() <= 7).mean():.0%}"),
        ("calendar monthly within 7 days", "98% (must not drop)", f"{((mon.b1_days - mon.duration_days).abs() <= 7).mean():.0%}",
         f"{((mon.bs_days - mon.duration_days).abs() <= 7).mean():.0%}")], columns=["target", "v2 baseline target", "B1", "Bayes"])

    pd.set_option("display.width", 220)
    with open(f"{a.out}/timing_report_{tag}.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Timing model ({tag}) vs B1 - test {te.snapshot_date.min():%Y-%m}..{te.snapshot_date.max():%Y-%m}, events observed\n\n"
                 "Synthetic data: shows whether the method works as designed, not real-world accuracy. Series grouping = oracle (gt_series_id).\n"
                 "b1 = baseline B1, bs = Bayesian survival; 3d/7d = share within 3/7 days; cover80 = share inside the P10-P90 window.\n\n")
        for k, t in tables.items():
            fh.write(f"## {k}\n\n```\n{fmt(t).to_string()}\n```\n\n")
            print(f"\n== {k}\n{fmt(t).to_string()}")
        fh.write(f"## targets\n\n```\n{targets.to_string(index=False)}\n```\n")
    print(f"\n== targets\n{targets.to_string(index=False)}")
    te.drop(columns=["b0_days", "b0_amount", "b0_active"]).to_csv(f"{a.out}/timing_predictions_{tag}.csv.gz", index=False)
    print(f"\n[{tag}] scored {len(te):,} test events in {time.time() - t0:.0f}s -> {a.out}/timing_report_{tag}.md")


if __name__ == "__main__":
    main()
