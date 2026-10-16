#!/usr/bin/env python3
"""
SafeBelt PACM v3 - model only: transactions in, per recurring bill out (detected, P, A, T, C), validated on what happened.

Data preparation (not a decision rule): outgoing payments grouped per customer by payee (biller+ref1 / merchant /
counterparty account / txn_type), own-account transfers dropped, payments to people split into separate bills
(--split bayes: amount_split.py, amount + timing Bayesian model selection; --split peaks: amount peaks, clusters below).
Only the last 24 months before each snapshot are used.

Models
  detected  LightGBM on series features (labels = true bill series with period >= 7 d), trained on train snapshots
  T         Bayesian survival (timing_v3): next date = median, window = P10-P90, conditioned on no payment since the last
  A         Bayesian shrinkage on log amounts: location = median of the last 3, scale from a per-txn_type inverse-gamma prior
            (nu0, s0^2 learned on train series); predictive Student-t -> median and P90
  P         LightGBM P(active): the next payment comes within 2 x typical gap + 15 days; isotonic-calibrated on calib
  C         raw = P x P(T window) x P(amount <= A_P90) -> isotonic on calib (target: paid inside the window and <= A_P90)

Split: train customers, snapshots 2024-10, 2025-01, 2025-04, 2025-07 (outcomes censored at 2025-10-01) -> fit
       train customers, snapshots 2025-11, 2026-01 (outcomes censored at 2026-02-01) -> calibrate P and C
       all customers,   snapshots 2026-02, 2026-04 + each customer's June payday (+1 day) (outcomes to 2026-09-30) -> test
Baselines: B1 (calendar-snapped next date), amount = median of last 3.

Usage: python pacm_v3.py [--data data_v6] [--out results_v6] [--today payday | YYYY-MM-DD] [--workers 20] [--customers 80 = smoke test]
"""
import argparse
import json
import os
import pickle
import time
from datetime import date
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

import amount_split as asp
import timing_v3 as tm
from backtest_v2 import EXPENSE_KINDS, predict as b1_predict
from generate_v2 import SPLIT_TRAIN_END, SPLIT_CALIB_END, END

HIST_DAYS = 730
# what counts as a bill (truth labels only): obligations + recurring transfers with a cycle >= 7 days, excluding
# wallet top-ups and haircuts (user decision 2026-10-13: those are spending, they belong in the Safety Buffer)
NOT_BILL_CATEGORIES = {"wallet", "personal_care"}
SNAPS = ([("train", date(y, m, 1)) for y, m in [(2024, 10), (2025, 1), (2025, 4), (2025, 7)]]
         + [("calib", date(y, m, 1)) for y, m in [(2025, 11), (2026, 1)]]
         + [("test", date(2026, m, 1)) for m in (2, 4)])          # + each customer's own "today" (see today_dates)
MIN_GAP = 3          # groups paid more often than every 3 days on average cannot be bills (cycle >= 7 d): not scored
CENSOR = {"train": SPLIT_TRAIN_END.toordinal(), "calib": SPLIT_CALIB_END.toordinal(), "test": END.toordinal() + 1}
CODES = {"counterparty_bank_code": str, "ref1": str, "ref2": str, "mcc": str, "biller_id": str}
TXN_TYPES = ["bill_payment", "direct_debit", "card_purchase", "qr_payment", "promptpay_out", "transfer_out", "atm_withdrawal"]
D_FEATURES = ["n_prior", "log_median_gap", "gap_cv", "gap_max_ratio", "share_gaps_near_median", "overdue_ratio", "log_amount", "amount_cv",
              "share_amount_mode", "dom_spread", "key_type", "txn_type", "has_mandate", "auto_share", "confidence", "p30", "rel_window"]
P_FEATURES = ["overdue_ratio", "since_days", "log_median_gap", "gap_cv", "gap_max_ratio", "share_gaps_near_median", "n_prior",
              "confidence", "p30", "p_surv_window", "rel_window", "amount_cv", "log_amount", "key_type", "txn_type", "has_mandate",
              "auto_share", "dom_spread", "det_score"]
MODEL = None
SPLIT = None             # amount_split prior (a0, b0, penalty) or None = amount peaks


# ---------------------------------------------------------------- data preparation (standard fields only)
def build_groups(data):
    """outgoing payments, own-account transfers dropped, grouped per customer by payee"""
    tx = tm.load(data, "transactions", dtype=CODES)
    ac = tm.load(data, "accounts")
    cu = tm.load(data, "customers")
    tx = tx[tx.direction == "DR"].copy()
    name = dict(zip(cu.customer_id, cu.customer_name))
    own = tx.counterparty_account_id.isin(set(ac.account_id)) | (tx.counterparty_name == tx.customer_id.map(name))
    tx = tx[~own]
    key = np.where(tx.biller_id.notna(), "B|" + tx.biller_id.fillna("") + "|" + tx.ref1.fillna(""),
          np.where(tx.merchant_name.notna(), "M|" + tx.merchant_name.fillna(""),
          np.where(tx.counterparty_account_id.notna(), "P|" + tx.counterparty_account_id.fillna(""), "T|" + tx.txn_type)))
    tx["gkey"] = tx.customer_id + "#" + key
    tx["o"] = (pd.to_datetime(tx.posted_datetime).dt.normalize() - pd.Timestamp("1970-01-01")).dt.days + tm.EPOCH
    return tx


def load_data(data="data_v6"):
    """groups + one row per (group, day); truth columns are attached only for labels / validation, never as features"""
    st = tm.load(data, "series_truth")
    ut = tm.load(data, "users_truth")
    tx = build_groups(data)
    lb = tm.load(data, "txn_labels")
    tx = tx.merge(lb[["txn_id", "gt_series_id", "gt_flags"]], on="txn_id", how="left")
    real = st.set_index("series_id")
    tx["is_exp"] = (tx.gt_series_id.map(real.kind).isin(EXPENSE_KINDS) & (tx.gt_series_id.map(real.true_period_days) >= 7)
                    & ~tx.gt_series_id.map(real.category).isin(NOT_BILL_CATEGORIES))
    tx["auto"] = tx.channel.eq("AUTO")
    tx["mand"] = tx.mandate_id.notna()
    day = tx.sort_values("posted_datetime").groupby(["gkey", "o"]).agg(
        amount=("amount", "sum"), txn_id=("txn_id", "first"), txn_type=("txn_type", "first"), is_exp=("is_exp", "max"),
        sid=("gt_series_id", "first"), auto=("auto", "max"), mand=("mand", "max")).reset_index()
    return st, ut, tx, day


def today_dates(data, mode, custs, month=(2026, 6)):
    """per-customer validation day: the day after the customer's first payroll credit in the given month (standard
    fields: direction CR, txn_type payroll_in); customers without one use the 1st of the month"""
    first = date(month[0], month[1], 1)
    if mode != "payday":
        d = date.fromisoformat(mode)
        return {c: d for c in custs}
    t = tm.load(data, "transactions", usecols=["customer_id", "posted_datetime", "direction", "txn_type"])
    t = t[(t.direction == "CR") & (t.txn_type == "payroll_in")]
    t["d"] = pd.to_datetime(t.posted_datetime).dt.date
    nxt = date(month[0] + (month[1] == 12), month[1] % 12 + 1, 1)
    t = t[(t.d >= first) & (t.d < nxt)].groupby("customer_id").d.min()
    return {c: date.fromordinal(t[c].toordinal() + 1) if c in t.index else first for c in custs}


def payee_names(tx):
    """display name per group from standard fields: counterparty name / merchant / biller description"""
    desc = tx.description.str.replace(r" REF\d+| \d{6}$", "", regex=True).str[:26]
    name = np.where(tx.counterparty_name.notna(), tx.counterparty_name, np.where(tx.merchant_name.notna(), tx.merchant_name,
                    np.where(tx.biller_id.notna(), desc, tx.txn_type)))
    return pd.Series(name, index=tx.index).groupby(tx.gkey).agg(lambda v: v.value_counts().index[0])


def clusters(amounts, near=np.log(1.2), apart=np.log(1.45), min_share=0.1):
    """cut points between amount levels (geometric midpoints) from history amounts: levels are the amounts with the most
    payments within +-20%, at least 1.45x apart and holding >= 10% of payments (min 2)"""
    la = np.log(np.maximum(np.asarray(amounts, float), 1))
    if len(la) < 4:
        return np.array([])
    support = (np.abs(la[:, None] - la[None, :]) < near).sum(1)
    levels = []
    for j in np.argsort(-support, kind="stable"):
        if support[j] < max(2, min_share * len(la)):
            break
        if all(abs(la[j] - l) >= apart for l in levels):
            levels.append(la[j])
    levels = np.sort(levels)
    return np.exp((levels[:-1] + levels[1:]) / 2)


def dom_spread(o):
    dom = np.array([date.fromordinal(int(x)).day for x in o])
    ang = dom / 31 * 2 * np.pi
    r = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
    return float(np.sqrt(-2 * np.log(min(max(r, 1e-9), 1.0))) * 31 / (2 * np.pi))


def prf(pred, y):
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return p, r, 2 * p * r / max(1e-9, p + r), tp, fp, fn


def init(model, split=None):
    global MODEL, SPLIT
    MODEL, SPLIT = model, split


def candidates(task):
    """one row per (series candidate, snapshot), history = last 24 months; outcome = next payment of the same amount group"""
    rows = []
    o_all, amt_all = task["o"], task["amount"]
    for split in ("train", "calib", "test"):
        if split not in task["splits"]:
            continue
        dates = [s for sp, s in SNAPS if sp == split]
        if split == "test" and task.get("today") and task["today"] not in dates:
            dates = dates + [task["today"]]
        for sd in dates:
            so = sd.toordinal()
            lo, i = int(np.searchsorted(o_all, so - HIST_DAYS)), int(np.searchsorted(o_all, so))
            if i - lo < 2:
                continue
            idx_all = np.arange(len(o_all))
            lab = np.zeros(len(o_all), int)
            if task["key_type"] == "P":
                lab = (asp.labels(o_all[lo:i], amt_all[lo:i], amt_all, SPLIT, MODEL, task["cat"]) if SPLIT
                       else np.searchsorted(clusters(amt_all[lo:i]), amt_all))
            for c in np.unique(lab[lo:i]):
                h = idx_all[(lab == c) & (idx_all >= lo) & (idx_all < i)]
                if len(h) < 2:
                    continue
                ho, ha = o_all[h], amt_all[h]
                gaps = np.diff(ho)
                med = float(np.median(gaps))
                since = so - int(ho[-1])
                if since > min(400, 1.5 * med + 30) or med < MIN_GAP:
                    continue
                hh = h[-(tm.MAX_GAPS + 1):]
                pmf = MODEL.predictive(o_all[hh], task["cat"])
                p = pmf.copy()
                p[:since] = 0
                if p.sum() <= 0:
                    continue
                p /= p.sum()
                cdf = np.cumsum(p)
                q10, q50, q90 = (int(np.searchsorted(cdf, q)) for q in (0.1, 0.5, 0.9))
                fut = idx_all[(lab == c) & (idx_all >= i)]
                nx = fut[0] if len(fut) else None
                nxo = int(o_all[nx]) if nx is not None else None
                la = np.log(np.maximum(ha, 1))
                vals, cnt = np.unique(np.round(ha), return_counts=True)
                sids = task["sid"][h][task["is_exp"][h]]
                lim = so + int(2 * med + 15) if since <= 2 * med + 15 else so + int(med + 15)
                rows.append({
                    "gkey": task["gkey"], "cand": int(c), "customer_id": task["customer_id"], "user_split": task["user_split"], "split": split,
                    "snapshot": sd.isoformat(), "n_prior": len(h), "log_median_gap": np.log(max(med, 1)), "median_gap": med,
                    "gap_cv": float(gaps.std() / max(gaps.mean(), 1)) if len(gaps) > 1 else np.nan, "gap_max_ratio": float(gaps.max() / max(med, 1)),
                    "share_gaps_near_median": float((np.abs(gaps - med) <= 0.2 * med + 1).mean()), "overdue_ratio": since / max(med, 1),
                    "since_days": since, "log_amount": float(np.median(la)), "amount_cv": float(la.std()), "share_amount_mode": float(cnt.max() / len(ha)),
                    "dom_spread": dom_spread(ho), "key_type": task["key_type"], "txn_type": task["cat"],
                    "has_mandate": float(task["mandate"][h].mean()), "auto_share": float(task["auto"][h].mean()),
                    "confidence": float(p[max(since, q50 - 3): q50 + 4].sum()), "p30": float(p[since: since + 31].sum()),
                    "p_surv_window": float(p[since: since + int(2 * med + 16)].sum()), "rel_window": (q90 - q10) / max(med, 1),
                    "t_p10": q10 - since, "t_p50": q50 - since, "t_p90": q90 - since,
                    "last_date": date.fromordinal(int(ho[-1])).isoformat(),
                    "log_amounts": " ".join(f"{x:.5f}" for x in la[-12:]),
                    "next_days": (nxo - so) if nxo is not None and nxo < CENSOR[split] else None,
                    "next_amount": float(amt_all[nx]) if nxo is not None and nxo < CENSOR[split] else None,
                    "active_label": (1 if nxo is not None and nxo < min(lim, CENSOR[split]) else 0 if lim <= CENSOR[split] else None),
                    "is_today": bool(split == "test" and sd == task.get("today")),
                    "exp_share": float(task["is_exp"][h].mean()), "dom_sid": pd.Series(sids).value_counts().index[0] if len(sids) else None})
    return rows


# ---------------------------------------------------------------- A: Bayesian shrinkage amount model
def t_quantile(q_z, df):
    return q_z * (1 + (q_z ** 2 + 1) / (4 * df))


class AmountModel:
    """log amount ~ Student-t predictive; per-txn_type inverse-gamma prior on the series variance (nu0, s0^2)."""
    NU = (1.0, 2.0, 4.0, 8.0, 16.0)

    def fit(self, logs, cats):
        self.prior = {}
        data = pd.DataFrame({"cat": cats, "x": logs})
        for cat, g in list(data.groupby("cat")) + [("_all", data)]:
            s2 = np.array([np.var(x[:-1], ddof=1) if len(x) > 2 else np.nan for x in g.x])
            s0 = float(np.nanmedian(s2)) if np.isfinite(s2).any() else 0.01
            s0 = max(s0, 1e-4)
            best, best_ll = None, -np.inf
            for nu in self.NU:
                ll = 0.0
                for x in g.x:
                    if len(x) < 2:
                        continue
                    hist, y = x[:-1], x[-1]
                    loc, sc, df = self._post(hist, nu, s0)
                    z = (y - loc) / sc
                    ll += -0.5 * (df + 1) * np.log1p(z * z / df) - np.log(sc)
                if ll > best_ll:
                    best, best_ll = nu, ll
            self.prior[cat] = (best, s0)
        return self

    @staticmethod
    def _post(hist, nu0, s0):
        n = len(hist)
        loc = float(np.median(hist[-3:]))
        s2 = float(np.var(hist, ddof=1)) if n > 1 else 0.0
        sn2 = (nu0 * s0 + (n - 1) * s2) / (nu0 + n - 1)
        df = nu0 + n - 1
        return loc, float(np.sqrt(sn2 * (1 + 1 / min(n, 3)))), df

    def predict(self, hist, cat, q=0.8):
        nu0, s0 = self.prior.get(cat, self.prior["_all"])
        loc, sc, df = self._post(np.asarray(hist), nu0, s0)
        zq = {0.8: 0.8416, 0.9: 1.2816}[q]
        return np.exp(loc), np.exp(loc + t_quantile(zq, df) * sc), (loc, sc, df)

    @staticmethod
    def cdf_t(y, loc, sc, df):
        """P(log amount <= y) for the Student-t, by numeric integration (no scipy)"""
        z = (y - loc) / sc
        g = np.linspace(-12, min(z, 12), 400) if z > -12 else np.array([-12, -12])
        dens = (1 + g * g / df) ** (-(df + 1) / 2)
        full = np.linspace(-12, 12, 800)
        tot = np.trapezoid((1 + full * full / df) ** (-(df + 1) / 2), full)
        return float(np.trapezoid(dens, g) / tot)


# ---------------------------------------------------------------- isotonic calibration (pool adjacent violators)
class Isotonic:
    def fit(self, x, y, w=None):
        o = np.argsort(x)
        x, y = np.asarray(x)[o], np.asarray(y, float)[o]
        w = np.ones_like(y) if w is None else np.asarray(w)[o]
        blocks = [[y[i] * w[i], w[i], x[i], x[i]] for i in range(len(x))]
        out = []
        for b in blocks:
            out.append(b)
            while len(out) > 1 and out[-2][0] / out[-2][1] > out[-1][0] / out[-1][1]:
                s, ww, lo, hi = out.pop()
                out[-1] = [out[-1][0] + s, out[-1][1] + ww, out[-1][2], hi]
        self.x = np.array([(b[2] + b[3]) / 2 for b in out])
        self.y = np.array([b[0] / b[1] for b in out])
        return self

    def predict(self, x):
        return np.interp(x, self.x, self.y)


# ---------------------------------------------------------------- helpers
def frame(df, cols):
    X = df[cols].copy()
    if "key_type" in cols:
        X["key_type"] = pd.Categorical(X.key_type, categories=["B", "M", "P", "T"])
    if "txn_type" in cols:
        X["txn_type"] = pd.Categorical(X.txn_type, categories=TXN_TYPES)
    return X


def lgb_fit(X, y, rounds=600):
    return lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50, "feature_fraction": 0.9,
                      "bagging_fraction": 0.9, "bagging_freq": 1, "verbose": -1, "seed": 7}, lgb.Dataset(X, y), rounds)


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * (len(y) - n1)))


def reliability(p, y, bins=(0, .2, .4, .6, .8, .9, 1.0001)):
    b = pd.cut(p, bins, right=False)
    t = pd.DataFrame({"p": p, "y": y}).groupby(b, observed=True).agg(n=("y", "size"), says=("p", "mean"), happened=("y", "mean"))
    t["says"] = t.says.map("{:.0%}".format)
    t["happened"] = t.happened.map("{:.0%}".format)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--customers", type=int, default=0, help="smoke test: first N customers (separate cache)")
    ap.add_argument("--data", default="data_v6")
    ap.add_argument("--out", default="results_v6")
    ap.add_argument("--today", default="payday", help="'payday' = day after each customer's June 2026 payday, or one date for everyone")
    ap.add_argument("--split", default="bayes", choices=["bayes", "peaks"], help="bayes needs <out>/amount_split_prior.json (amount_split.py)")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    cache = f"{a.out}/pacm_cands.csv.gz" if not a.customers else f"{a.out}/pacm_cands_smoke{a.customers}.csv.gz"
    st, ut, tx, day = load_data(a.data)
    usplit = dict(zip(ut.customer_id, ut.user_split))
    today_map = today_dates(a.data, a.today, sorted(usplit))
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        tasks = []
        if a.customers:
            day = day[day.gkey.str.split("#").str[0].isin(sorted(usplit)[: a.customers])]
        for k, g in day.groupby("gkey"):
            if len(g) < 3:
                continue
            cid = k.split("#")[0]
            tasks.append({"gkey": k, "customer_id": cid, "user_split": usplit[cid], "key_type": k.split("#")[1][0], "cat": g.txn_type.mode().iloc[0],
                          "o": g.o.to_numpy(), "amount": g.amount.to_numpy(float), "is_exp": g.is_exp.to_numpy(bool), "sid": g.sid.to_numpy(object),
                          "auto": g.auto.to_numpy(float), "mandate": g.mand.to_numpy(float),
                          "splits": ("train", "calib", "test") if usplit[cid] == "train" else ("test",), "today": today_map[cid]})
        socc, scat = tm.occurrences(a.data)
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
        model.max_cells = 200
        split = json.load(open(f"{a.out}/amount_split_prior.json")) if a.split == "bayes" else None
        pickle.dump(model, open(f"{a.out}/timing_model.pkl", "wb"))
        print(f"T prior fitted; building candidates for {len(tasks):,} groups ({time.time() - t0:.0f}s)", flush=True)
        with Pool(a.workers, initializer=init, initargs=(model, split)) as pool:
            df = pd.DataFrame([r for rows in pool.imap_unordered(candidates, tasks, chunksize=40) for r in rows])
        df.to_csv(cache, index=False)
    df["y_bill"] = (df.exp_share >= 0.5) & ~df.dom_sid.map(st.set_index("series_id").category).isin(NOT_BILL_CATEGORIES)
    df = df.join(st.set_index("series_id")[["cycle_type", "true_period_days", "name"]], on="dom_sid")
    tr, ca, te = (df.split == s for s in ("train", "calib", "test"))
    print(f"{len(df):,} candidates: {df.split.value_counts().to_dict()} ({time.time() - t0:.0f}s)", flush=True)

    # ---- detected
    Xd = frame(df, D_FEATURES)
    det = lgb_fit(Xd[tr], df.y_bill[tr])
    df["det_score"] = det.predict(Xd)
    df["detected"] = df.det_score >= 0.5
    det.save_model(f"{a.out}/pacm_detect.txt")

    # ---- A: amount model
    df["logs"] = df.log_amounts.astype(str).str.split().map(lambda v: np.array(v, float))
    trs = df[tr & df.y_bill].drop_duplicates(["gkey", "cand"], keep="last")
    am = AmountModel().fit(list(trs.logs), list(trs.txn_type))
    pr = [am.predict(x, c, q=0.9) for x, c in zip(df.logs, df.txn_type)]
    df["a_med"] = [p[0] for p in pr]
    df["a_p90"] = [p[1] for p in pr]
    post = [p[2] for p in pr]
    df["p_amt"] = [AmountModel.cdf_t(np.log(r.a_p90), *ps) for r, ps in zip(df[["a_p90"]].itertuples(), post)]
    df["b_amt"] = df.logs.map(lambda x: float(np.exp(np.median(x[-3:]))))

    # ---- P: P(active)
    Xp = frame(df, P_FEATURES)
    ptr = tr & df.active_label.notna() & df.detected
    pmod = lgb_fit(Xp[ptr], df.active_label[ptr].astype(int), 400)
    df["p_raw"] = pmod.predict(Xp)
    pca = ca & df.active_label.notna() & df.detected
    iso_p = Isotonic().fit(df.p_raw[pca], df.active_label[pca])
    df["P"] = iso_p.predict(df.p_raw)

    # ---- C
    df["in_window"] = df.next_days.notna() & (df.next_days >= df.t_p10) & (df.next_days <= df.t_p90)
    df["amt_ok"] = df.next_amount.notna() & (df.next_amount <= df.a_p90 * 1.0001)
    df["c_target"] = (df.in_window & df.amt_ok).astype(int)
    df["c_observable"] = df.next_days.notna() | (df.active_label == 0) | ((df.snapshot.map(lambda s: date.fromisoformat(s).toordinal()) + df.t_p90)
                                                                          < df.split.map(CENSOR))
    df["c_raw"] = df.P * 0.8 * df.p_amt
    cca = ca & df.detected & df.c_observable
    iso_c = Isotonic().fit(df.c_raw[cca], df.c_target[cca])
    df["C"] = iso_c.predict(df.c_raw)

    # ---- B1 baseline date on the same rows
    b = df[["snapshot", "last_date", "median_gap", "since_days"]].rename(columns={"median_gap": "median_gap_prior", "since_days": "days_since_last"}).copy()
    b["snapshot_date"] = pd.to_datetime(b.snapshot)
    b["last_date"] = pd.to_datetime(b.last_date)
    b["amount_med3"] = b["last_amount"] = 0.0
    df["b1_days"] = b1_predict(b).b1_days.values
    df.drop(columns=["logs"]).to_csv(f"{a.out}/pacm_predictions.csv.gz", index=False)
    print(f"models fitted ({time.time() - t0:.0f}s)", flush=True)
    report(df[te].copy(), a.out)
    output_today(df[te].copy(), st, tx, a.out, a.today)


def report(t, out):
    pd.set_option("display.width", 220)
    L = ["# PACM v3 - model only, test 2026-02..2026-06 (synthetic data)", ""]

    def show(title, x):
        print(f"\n== {title}\n{x.to_string()}")
        L.extend([f"## {title}", "", "```", x.to_string(), "```", ""])

    rows = []
    for us, g in [("all", t)] + list(t.groupby("user_split")):
        p, r, f, *_ = prf(g.detected, g.y_bill)
        rows.append({"customers": us, "candidates": len(g), "precision": f"{p:.1%}", "recall": f"{r:.1%}", "F1": f"{f:.3f}"})
    show("DETECTED: is it a recurring bill?", pd.DataFrame(rows).set_index("customers"))

    d = t[t.detected & t.y_bill]
    ev = d[d.next_days.notna()]
    def tblock(g):
        e, e1 = (g.t_p50 - g.next_days).abs(), (g.b1_days - g.next_days).abs()
        return pd.Series({"n": len(g), "T MAE d": round(e.mean(), 1), "B1 MAE d": round(e1.mean(), 1), "T within 3d": f"{(e <= 3).mean():.0%}",
                          "B1 within 3d": f"{(e1 <= 3).mean():.0%}", "in P10-P90": f"{((g.next_days >= g.t_p10) & (g.next_days <= g.t_p90)).mean():.0%}"})
    x = pd.concat({"all": tblock(ev)}, axis=1).T
    x = pd.concat([x, ev.groupby("cycle_type").apply(tblock, include_groups=False), ev.groupby("user_split").apply(tblock, include_groups=False)])
    show("T: next date (detected real bills whose next payment was observed; target: 80% inside P10-P90)", x)

    ea = ev[ev.next_amount.notna()]
    def ablock(g):
        ape = (g.a_med - g.next_amount).abs() / g.next_amount
        bape = (g.b_amt - g.next_amount).abs() / g.next_amount
        return pd.Series({"n": len(g), "A median APE": f"{ape.median():.1%}", "baseline APE": f"{bape.median():.1%}", "A within 5%": f"{(ape <= .05).mean():.0%}",
                          "real <= A_P90": f"{(g.next_amount <= g.a_p90 * 1.0001).mean():.0%}", "P90 above median": f"{(g.a_p90 / g.a_med - 1).median():.1%}"})
    x = pd.concat({"all": ablock(ea)}, axis=1).T
    x = pd.concat([x, ea.groupby("txn_type").apply(ablock, include_groups=False)])
    show("A: amount (target: 90% of real amounts <= A_P90)", x)

    pe = t[t.detected & t.active_label.notna()]
    show(f"P: will it be paid again? AUC {auc(pe.active_label, pe.P):.3f}, Brier {((pe.P - pe.active_label) ** 2).mean():.3f}, "
         f"base rate {pe.active_label.mean():.0%} (reliability: says vs happened)", reliability(pe.P, pe.active_label))
    ce = t[t.detected & t.c_observable]
    show(f"C: paid inside the T window and <= A_P90? Brier {((ce.C - ce.c_target) ** 2).mean():.3f}, happened {ce.c_target.mean():.0%} "
         f"(reliability: says vs happened)", reliability(ce.C, ce.c_target))
    td = t[t.detected & t.is_today.astype(bool)]
    ex = td[td.customer_id == td.customer_id.iloc[0]].sort_values("t_p50").head(12)
    show("example output for one customer on " + ex.snapshot.iloc[0], ex[["gkey", "name", "P", "t_p10", "t_p50", "t_p90", "a_med", "a_p90", "C"]].round(2))
    with open(f"{out}/pacm_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def output_today(t, st, tx, out, today):
    """one row per payment group on `today`: detected / P / T / A_P90 / C, and what really happened per the truth table"""
    real = st.set_index("series_id")
    names = payee_names(tx)
    x = t[t.is_today.astype(bool)].copy()
    x["so"] = x.snapshot.map(lambda v: date.fromisoformat(v).toordinal())
    fmt = lambda base, d: date.fromordinal(int(base + d)).strftime("%d %b %Y")
    x["today"] = x.snapshot
    x["payee"] = x.gkey.map(names)
    x["truth"] = np.where(x.y_bill, x.dom_sid.map(real.name).fillna("?") + " [bill, " + x.dom_sid.map(real.status).fillna("?") + "]", "not a bill")
    x["T"] = [fmt(so, v) for so, v in zip(x.so, x.t_p50)]
    x["T_window"] = [f"{fmt(so, a)} - {fmt(so, b)}" for so, a, b in zip(x.so, x.t_p10, x.t_p90)]
    x["actual"] = [f"{fmt(so, d)} {a:,.0f}" if d == d else "no payment by 30 Sep 2026" for so, d, a in zip(x.so, x.next_days, x.next_amount)]
    x["date_ok"] = x.next_days.notna() & (x.next_days >= x.t_p10) & (x.next_days <= x.t_p90)
    x["amount_ok"] = x.next_amount.notna() & (x.next_amount <= x.a_p90 * 1.0001)
    active = x.P >= 0.5
    paid = x.next_days.notna()
    x["verdict"] = np.select(
        [x.detected & ~x.y_bill, ~x.detected & x.y_bill, ~x.detected & ~x.y_bill,
         active & ~paid, ~active & paid, ~active & ~paid,
         x.date_ok & x.amount_ok, ~x.date_ok & x.amount_ok, x.date_ok & ~x.amount_ok],
        ["WRONG: not a bill in truth", "WRONG: real bill not detected", "correct: not a bill",
         "WRONG: said active, no payment", "WRONG: said inactive, but paid", "correct: inactive, no payment",
         "correct", "WRONG: date outside window", "WRONG: amount above P90"], "WRONG: date and amount")
    # real bills paid in the next 60 days that never became a candidate (< 2 payments in a group, or grouping split them)
    lab = tx[tx.is_exp & ~tx.gt_flags.fillna("").str.contains("duplicate_charge|split_part2")]
    have = set(x.dom_sid.dropna())
    cust_so = x.groupby("customer_id").so.first()
    cust_split = x.groupby("customer_id").user_split.first()
    lab = lab[lab.customer_id.isin(cust_so.index)].copy()
    lab["cso"] = lab.customer_id.map(cust_so)
    nxt = lab[(lab.o >= lab.cso) & (lab.o < lab.cso + 60)].sort_values("o").groupby("gt_series_id").first()
    before = lab[lab.o < lab.cso].groupby("gt_series_id").size()
    rows = []
    for sid, r in nxt.iterrows():
        if sid in have:
            continue
        nb = int(before.get(sid, 0))
        rows.append({"customer_id": r.customer_id, "user_split": cust_split[r.customer_id], "today": date.fromordinal(int(r.cso)).isoformat(),
                     "payee": names.get(r.gkey, "?"),
                     "truth": f"{real.name[sid]} [bill]", "detected": False,
                     "verdict": "WRONG: real bill has no prediction (" + ("first payment ever" if nb == 0 else f"seen {nb}x in its group") + ")",
                     "actual": f"{date.fromordinal(int(r.o)).strftime('%d %b %Y')} {r.amount:,.0f}"})
    cols = ["customer_id", "user_split", "today", "payee", "truth", "detected", "det_score", "P", "T", "T_window", "a_med", "a_p90", "C", "actual",
            "date_ok", "amount_ok", "verdict"]
    full = pd.concat([x[cols], pd.DataFrame(rows)], ignore_index=True)
    full.to_csv(f"{out}/pacm_output_{today}.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    shown = full[full.verdict != "correct: not a bill"]
    v = shown.verdict.value_counts()
    print(f"\n== OUTPUT on {today} (per customer): {x.customer_id.nunique()} customers, every row checked against the truth table "
          f"({len(shown):,} rows about real or predicted bills; {(full.verdict == 'correct: not a bill').sum():,} non-bills correctly ignored)")
    print(pd.DataFrame({"rows": v, "share": (v / v.sum()).map("{:.1%}".format)}).to_string())
    lines = []
    for split in ("train", "heldout"):
        cid = x[x.user_split == split].customer_id.iloc[0]
        lines.append(f"\n-- {cid} ({'held-out' if split == 'heldout' else 'seen'} customer), today {today}")
        for r in full[(full.customer_id == cid) & (full.verdict != "correct: not a bill")].itertuples():
            mark = "OK " if r.verdict.startswith("correct") else "XX "
            if not r.detected and r.verdict.startswith("WRONG: real bill has no"):
                lines.append(f"  {mark}{r.payee} | truth: {r.truth} | no prediction | actual {r.actual} -> {r.verdict}")
                continue
            lines.append(f"  {mark}{r.payee} | truth: {r.truth} | model: {'bill' if r.detected else 'not a bill'} | "
                         f"P {'active' if r.P >= 0.5 else 'inactive'} {r.P:.2f} | T {r.T} ({r.T_window}) | A {r.a_p90:,.0f} (P90) | "
                         f"C {r.C:.2f} | actual {r.actual} -> {r.verdict}")
    print("\n".join(lines))
    with open(f"{out}/pacm_output_{today}_examples.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
