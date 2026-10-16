#!/usr/bin/env python3
"""
SafeBelt synthetic bank data generator - v2 (standard field schema)
====================================================================

Transactions use ONLY standard bank fields:
  txn_id, customer_id, account_id, posted_datetime, amount, direction, txn_type, channel,
  balance_after, description, counterparty_account_id, counterparty_bank_code,
  counterparty_name, promptpay_proxy_type, biller_id, ref1, ref2, mandate_id,
  mcc, merchant_id, merchant_name, original_currency, original_amount

Removed (not standard / derived): memo, is_own_account, is_payroll.
Future work (not generated): scheduled_transfer_flag.

Bank-side master data: customers.csv, accounts.csv.
Ground truth (never a model input): txn_labels, series_truth, users_truth.
Model-ready: snapshots_{train,calib,test}.csv.gz with leak-free censoring.

Dates and amounts come from calendar / business rules (day of month, business-day
shifts, paydays, deadline windows, pauses, cancellations, balance constraints),
not from the statistical model SafeBelt fits. Use for development and stress tests;
claim accuracy only on real data.

Usage: python generate_v2.py --users 1000 --seed 42 --out data_v2
"""
import argparse
import calendar
import csv
import gzip
import hashlib
import math
import os
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

START, END = date(2024, 1, 1), date(2026, 9, 30)
SPLIT_TRAIN_END = date(2025, 10, 1)     # train snapshots < this, labels censored here
SPLIT_CALIB_END = date(2026, 2, 1)      # calib snapshots < this, labels censored here
SPLIT_TEST_END = date(2026, 7, 1)       # test snapshots < this (outcomes observed to END)
OWN_BANK = "004"
OTHER_BANKS = ["014", "002", "006", "025", "011", "030"]

TX_COLS = ["txn_id", "customer_id", "account_id", "posted_datetime", "amount", "direction", "txn_type", "channel",
           "balance_after", "description", "counterparty_account_id", "counterparty_bank_code", "counterparty_name",
           "promptpay_proxy_type", "biller_id", "ref1", "ref2", "mandate_id", "mcc", "merchant_id", "merchant_name",
           "original_currency", "original_amount"]
CP_KEYS = TX_COLS[10:]
LB_COLS = ["txn_id", "gt_kind", "gt_series_id", "gt_flags", "gt_link_txn_id"]

# ---------------------------------------------------------------- calendar
_HOL = []
for y in (2024, 2025, 2026):
    _HOL += [(y, 1, 1), (y, 4, 6), (y, 4, 13), (y, 4, 14), (y, 4, 15), (y, 5, 1), (y, 5, 4), (y, 6, 3), (y, 7, 28),
             (y, 8, 12), (y, 10, 13), (y, 10, 23), (y, 12, 5), (y, 12, 10), (y, 12, 31)]
_HOL += [(2024, 2, 24), (2024, 5, 22), (2024, 7, 20), (2024, 7, 22), (2025, 2, 12), (2025, 5, 11), (2025, 7, 10),
         (2025, 7, 11), (2026, 3, 3), (2026, 5, 31), (2026, 7, 29), (2026, 7, 30)]
HOLIDAYS = {date(*t) for t in _HOL}


def is_bd(d):
    return d.weekday() < 5 and d not in HOLIDAYS


def shift(d, rule):
    step = {"next": 1, "prev": -1}.get(rule)
    while step and not is_bd(d):
        d += timedelta(days=step)
    return d


def dom_date(y, m, dom):
    return date(y, m, min(dom, calendar.monthrange(y, m)[1]))


def month_iter(a, b):
    y, m = a.year, a.month
    while (y, m) <= (b.year, b.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def h(*parts, n=8):
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:n].upper()


# ---------------------------------------------------------------- catalogs
# billers: key -> (biller_id, description variants, category, ref1 kind, has_period_ref2)
BILLERS = {
    "AIS": ("BL10001", ["AIS BILL PAYMENT", "ADVANCED INFO SERVICE", "ชำระบิล AIS"], "telecom", "phone", True),
    "TRUE": ("BL10002", ["TRUEMOVE-H BILL", "TRUE MOVE H", "ชำระบิล ทรูมูฟ เอช"], "telecom", "phone", True),
    "DTAC": ("BL10003", ["DTAC BILL", "TOTAL ACCESS COMM", "ชำระบิล ดีแทค"], "telecom", "phone", True),
    "AIS_FIBRE": ("BL10004", ["AIS FIBRE", "AWN FIBRE BILL", "ชำระบิล AIS ไฟเบอร์"], "telecom", "contract", True),
    "3BB": ("BL10005", ["3BB INTERNET", "TRIPLE T BROADBAND", "ชำระบิล 3BB"], "telecom", "contract", True),
    "MEA": ("BL20001", ["MEA ELECTRICITY", "การไฟฟ้านครหลวง", "MEA BILL PAYMENT"], "utilities", "ca", True),
    "PEA": ("BL20002", ["PEA ELECTRICITY", "การไฟฟ้าส่วนภูมิภาค", "PEA BILL"], "utilities", "ca", True),
    "MWA": ("BL20003", ["MWA WATER", "การประปานครหลวง", "MWA BILL"], "utilities", "ca", True),
    "KBANK_CARD": ("BL30001", ["KBANK CREDIT CARD", "ชำระบัตรเครดิต กสิกรไทย", "KCC PAYMENT"], "credit_card", "card", False),
    "AEON": ("BL30002", ["AEON THANA SINSAP", "อิออน ธนสินทรัพย์", "AEON INSTALLMENT"], "installment", "contract", False),
    "MTC": ("BL30003", ["MUANGTHAI CAPITAL", "เมืองไทย แคปปิตอล", "MTC LOAN"], "installment", "contract", False),
    "KRUNGSRI_AUTO": ("BL30004", ["KRUNGSRI AUTO", "กรุงศรี ออโต้", "KSA AUTO LOAN"], "installment", "contract", False),
    "PLOAN": ("BL30005", ["KTC PROUD", "KTC PERSONAL LOAN", "เคทีซี พราว"], "loan", "contract", False),
    "AIA": ("BL40001", ["AIA COMPANY LTD", "เอไอเอ ประกันชีวิต", "AIA THAILAND"], "insurance", "policy", False),
    "MTL": ("BL40002", ["MUANG THAI LIFE", "เมืองไทยประกันชีวิต", "MTL PREMIUM"], "insurance", "policy", False),
    "FWD": ("BL40003", ["FWD LIFE INSURANCE", "เอฟดับบลิวดี ประกันชีวิต", "FWD PREMIUM"], "insurance", "policy", False),
    "VIRIYAH": ("BL40004", ["VIRIYAH INSURANCE", "วิริยะประกันภัย", "VIRIYAH PREMIUM"], "insurance", "policy", False),
    "SLF": ("BL50001", ["กยศ. ชำระหนี้", "STUDENT LOAN FUND", "SLF REPAYMENT"], "government", "contract", False),
    "DLT": ("BL50002", ["กรมการขนส่งทางบก", "DLT VEHICLE TAX", "ภาษีรถ DLT"], "government", "plate", False),
    "JURISTIC": ("BL60001", ["นิติบุคคลอาคารชุด", "CONDO JURISTIC PERSON", "ค่าส่วนกลาง"], "housing", "unit", True),
    "KASSET": ("BL70001", ["K-ASSET DCA", "กองทุน K-ASSET", "KASSET FUND SUB"], "investment", "contract", False),
    "GYM": ("BL80001", ["FITNESS FIRST MEMBERSHIP", "FITNESS FIRST TH", "FF MEMBER FEE"], "fitness", "member", False),
}
# card / QR merchants: key -> (merchant_name, description variants, mcc, currency, category)
MERCH = {
    "NETFLIX": ("NETFLIX.COM", ["NETFLIX.COM", "NETFLIX INTERNATIONAL", "NETFLIX.COM 866-579"], "4899", "THB", "streaming"),
    "SPOTIFY": ("SPOTIFY", ["SPOTIFY P1A2B3", "SPOTIFY AB", "SPOTIFY STOCKHOLM"], "5815", "THB", "streaming"),
    "YOUTUBE": ("GOOGLE YOUTUBE", ["GOOGLE *YOUTUBEPREMIUM", "GOOGLE YOUTUBE PREMIUM", "YOUTUBEPREMIUM G.CO"], "5815", "THB", "streaming"),
    "DISNEY": ("DISNEY PLUS", ["DISNEY PLUS", "DISNEY+ HOTSTAR", "DISNEYPLUS TH"], "4899", "THB", "streaming"),
    "ICLOUD": ("APPLE.COM/BILL", ["APPLE.COM/BILL", "APPLE.COM BILL ITUNES.COM", "APPLE SERVICES"], "5818", "THB", "digital_services"),
    "GOOGLE_ONE": ("GOOGLE ONE", ["GOOGLE *GOOGLE ONE", "GOOGLE STORAGE", "GOOGLE ONE G.CO/HELPPAY"], "5818", "THB", "digital_services"),
    "CANVA": ("CANVA", ["CANVA* I0123456", "CANVA PTY LTD", "CANVA.COM"], "5818", "USD", "digital_services"),
    "CLEAN": ("BEEHIVE CLEANING", ["BEEHIVE CLEANING", "HOME CLEAN SERVICE", "BEEHIVE TH"], "7349", "THB", "home_services"),
    "BARBER": ("BARBER SHOP", ["BARBER SHOP", "ร้านตัดผม", "HAIR SALON"], "7230", "THB", "personal_care"),
    "COURT": ("SPORT CLUB", ["SPORT CLUB BOOKING", "BADMINTON COURT", "สนามแบด"], "7997", "THB", "sports"),
    "GYM_SHOP": ("FITNESS FIRST", ["FITNESS FIRST", "FITNESS FIRST TH", "FF CENTRAL WORLD"], "7997", "THB", "fitness"),
    "7ELEVEN": ("7-ELEVEN", ["7-ELEVEN", "7-11 CP ALL", "CP ALL PCL"], "5499", "THB", "convenience"),
    "CAFE": ("CAFE AMAZON", ["CAFE AMAZON", "AMAZON PTT", "CAFE AMAZON BKK"], "5814", "THB", "food_drink"),
    "GRAB": ("GRAB", ["GRAB*", "GRABTAXI", "GRAB TH"], "4121", "THB", "transport"),
    "LINEMAN": ("LINE MAN", ["LINE MAN", "LINEMAN WONGNAI", "LMWN"], "5812", "THB", "food_delivery"),
    "SHOPEE": ("SHOPEE", ["SHOPEEPAY", "SHOPEE TH", "SHOPEE ONLINE"], "5999", "THB", "ecommerce"),
    "AGODA": ("AGODA", ["AGODA.COM", "AGODA COMPANY", "AGODA BOOKING"], "4722", "THB", "travel"),
    "AIRASIA": ("AIRASIA", ["AIRASIA", "THAI AIRASIA", "AIRASIA.COM"], "4511", "THB", "travel"),
    "CLINIC": ("BANGKOK CLINIC", ["BANGKOK CLINIC", "คลินิกเวชกรรม", "MEDICAL CLINIC"], "8011", "THB", "medical"),
}
ONEOFF_SHOPS = [("MK RESTAURANT", "5812"), ("BAR B Q PLAZA", "5812"), ("SIZZLER", "5812"), ("UNIQLO", "5651"),
                ("CENTRAL DEPT", "5311"), ("BIG C", "5411"), ("LOTUS'S", "5411"), ("TOPS MARKET", "5411"),
                ("WATSONS", "5912"), ("BOOTS", "5912"), ("MAJOR CINEPLEX", "7832"), ("SF CINEMA", "7832"),
                ("SWENSEN'S", "5812"), ("AFTER YOU", "5812"), ("BANANA IT", "5732"), ("POWER BUY", "5732"),
                ("IKEA", "5712"), ("HOMEPRO", "5200"), ("B2S", "5942"), ("KINOKUNIYA", "5942"),
                ("SUKI TEE NOI", "5812"), ("SHABUSHI", "5812"), ("STARBUCKS", "5814"), ("PTT STATION", "5541"),
                ("SHELL", "5541"), ("MUJI", "5311"), ("ZARA", "5651"), ("DAISO", "5331")]
FIRST = ["สมชาย", "สมศรี", "วิชัย", "มาลี", "ประภา", "สุนทร", "อรุณ", "นภา", "กิตติ", "จันทร์เพ็ญ", "ธนา", "พิมพ์",
         "ณัฐ", "ปวีณา", "ศิริพร", "อนุชา", "กมล", "วรรณา"]
COMPANIES = ["SCG", "CP GROUP", "AGODA SERVICES", "LINE MAN WONGNAI", "BITKUB", "CENTRAL RETAIL", "TRUE DIGITAL",
             "SIAM PIWAT", "KBTG", "SHOPEE TH", "BOSCH TH", "DTGO", "TOYOTA TH"]


@dataclass
class Series:
    sid: str
    uid: str
    name: str
    key: str              # biller / merchant / person key
    category: str
    kind: str             # obligation | recurring_transfer | own_transfer | income | reimbursement
    period_type: str
    nominal_days: float
    essential: bool
    txn_type: str
    channel: str
    direction: str
    start: date
    end: date
    rule: dict
    amount: dict
    cp: dict = None       # fixed counterparty fields
    status: str = "active"
    quirks: list = field(default_factory=list)
    user_share: float = None
    linked_sid: str = None
    droppable: bool = False
    occ: list = field(default_factory=list)


class Gen:
    def __init__(self, n_users, seed, out):
        self.rng = np.random.default_rng(seed)
        self.n_users, self.out = n_users, out
        self.alt_ids = {k: self.rng.random() < 0.3 for k in MERCH}
        days = (END - START).days + 1
        walk = np.cumsum(self.rng.normal(0, 0.06, days))
        self.fx = 34.5 + 1.2 * np.sin(np.arange(days) / 365 * 2 * np.pi) + walk - walk.mean()
        self.series, self.customers, self.accounts, self.users_truth = [], [], [], []
        self.series_occ = []            # (sid, txn_id, date, amount, flags, link) for labeled series txns
        self.txn_no = 0
        os.makedirs(out, exist_ok=True)

    # ------------------------------------------------ small helpers
    def r(self):
        return self.rng.random()

    def ri(self, a, b):
        return int(self.rng.integers(a, b + 1))

    def ch(self, xs, p=None):
        return xs[int(self.rng.choice(len(xs), p=p))]

    def fx_rate(self, d):
        return float(self.fx[min(max((d - START).days, 0), len(self.fx) - 1)])

    @staticmethod
    def blank():
        return {k: None for k in CP_KEYS}

    def masked_name(self, title=None, first=None):
        return f"{title or self.ch(['นาย', 'นาง', 'น.ส.'])} {first or self.ch(FIRST)} {self.ch('กขคงจฉชซญฎฐณดตถทธนบปผพภมยรลวศสหอฮ')}****"

    def person_cp(self, uid, who, bank=None, name=None, proxy=None):
        bank = bank or (OWN_BANK if self.r() < 0.45 else self.ch(OTHER_BANKS))
        c = self.blank()
        c.update(counterparty_account_id="X" + h(uid, who, n=9), counterparty_bank_code=bank,
                 counterparty_name=name or self.masked_name(),
                 promptpay_proxy_type=None if proxy == "none" else (proxy or self.ch(["mobile", "citizen_id", None], [0.55, 0.15, 0.3])))
        return c

    def biller_cp(self, key, ref1, mandate=None):
        c = self.blank()
        c.update(biller_id=BILLERS[key][0], ref1=ref1, mandate_id=mandate)
        return c

    def ref1_for(self, kind):
        return {"phone": lambda: "08" + str(self.ri(10**7, 10**8 - 1)), "ca": lambda: str(self.ri(10**9, 10**10 - 1)),
                "card": lambda: f"4{self.ri(100, 999)}XXXXXXXX{self.ri(1000, 9999)}", "policy": lambda: f"P{self.ri(10**7, 10**8 - 1)}",
                "plate": lambda: f"{self.ri(1, 9)}กข{self.ri(1000, 9999)}", "unit": lambda: f"U{self.ri(100, 3999)}",
                "member": lambda: f"M{self.ri(10**6, 10**7 - 1)}", "contract": lambda: f"C{self.ri(10**8, 10**9 - 1)}"}[kind]()

    def merchant_cp(self, key, d=None, thb=None, mid_override=None):
        name, variants, mcc, cur, _ = MERCH[key]
        c = self.blank()
        mid = mid_override or ("M" + h(key, n=9))
        if self.alt_ids.get(key) and self.r() < 0.3 and not mid_override:
            mid = "M" + h(key, "alt", n=9)
        c.update(mcc=mcc, merchant_id=mid, merchant_name=name, original_currency=cur)
        return c

    def describe(self, variants, ref=False):
        d = variants[0] if self.r() < 0.55 else self.ch(variants)
        if ref and self.r() < 0.6:
            d += f" REF{self.ri(10**7, 10**8 - 1)}"
        elif self.r() < 0.3:
            d += f" {self.ri(100000, 999999)}"
        if self.r() < 0.12:
            d = d[: self.ri(12, 20)]
        if self.r() < 0.08:
            d = d.lower()
        return d

    # ------------------------------------------------ events (per user, finalized later)
    def ev(self, d, amt, direction, txn_type, channel, desc, cp, kind, sid=None, flags="", link=None, droppable=False, hour=None):
        if d < self.open_d or d > END or amt <= 0:
            return None
        self.eid += 1
        if hour is None:
            hour = {"payroll_in": self.ri(0, 5), "direct_debit": self.ri(0, 6)}.get(txn_type, self.ri(7, 23))
        self.events.append({"eid": self.eid, "d": d, "t": f"{hour:02d}:{self.ri(0, 59):02d}:{self.ri(0, 59):02d}",
                            "amount": round(float(amt), 2), "direction": direction, "txn_type": txn_type, "channel": channel,
                            "description": desc, "cp": cp, "kind": kind, "sid": sid, "flags": flags, "link": link,
                            "droppable": droppable})
        return self.eid

    # ------------------------------------------------ schedules & amounts
    def schedule(self, s, paydays):
        R, out, t = s.rule, [], s.rule["type"]
        if t == "monthly":
            k, a0 = R.get("every", 1), R.get("anchor", 0)
            for y, m in month_iter(s.start, s.end):
                if ((y * 12 + m) - a0) % k:
                    continue
                d = dom_date(y, m, R["dom"])
                if "jitter" in R:
                    d += timedelta(days=self.ri(*R["jitter"]))
                out.append(shift(d, R.get("shift", "none")))
        elif t == "lastbd":
            out = [shift(dom_date(y, m, 31), "prev") for y, m in month_iter(s.start, s.end)]
        elif t == "semimonthly":
            for y, m in month_iter(s.start, s.end):
                out += [shift(date(y, m, 15), "prev"), shift(dom_date(y, m, 31), "prev")]
        elif t == "weekly":
            d = s.start + timedelta(days=(R["weekday"] - s.start.weekday()) % 7)
            while d <= s.end:
                if self.r() >= R.get("skip", 0):
                    out.append(d)
                d += timedelta(days=7 * R.get("every", 1))
        elif t == "after_payday":
            out = [p + timedelta(days=self.ri(*R["lag"])) for p in paydays]
        elif t == "irregular":
            d = s.start + timedelta(days=self.ri(0, R["gap"][1]))
            while d <= s.end:
                out.append(d)
                d += timedelta(days=self.ri(*R["gap"]))
        elif t == "annual_window":
            out = [date(y, R["month"], R["day_lo"]) + timedelta(days=self.ri(0, R["width"])) for y in range(s.start.year, s.end.year + 1)]
        elif t == "after_series":
            out = [d0 + timedelta(days=self.ri(*R["lag"])) for d0, _, f0 in R["parent"].occ if f0 != "skipped"]
        return sorted(d for d in out if s.start <= d <= s.end)

    def amount(self, s, d, i):
        A = s.amount
        base = A["base"]
        for cd, nb in A.get("changes", []):
            if d >= cd:
                base = nb
        k = A.get("kind", "fixed")
        f = {3: 1.25, 4: 1.45, 5: 1.4, 6: 1.15}.get(d.month, 1.0)
        if k == "seasonal":
            v = base * f * self.rng.lognormal(0, 0.08)
        elif k == "base_plus_seasonal":
            v = base + A["var"] * f * self.rng.lognormal(0, 0.1)
        elif k == "variable":
            v = base * self.rng.lognormal(0, A.get("sd", 0.35))
        elif k == "usage":
            v = base + (self.ri(30, 250) if self.r() < 0.15 else 0)
        elif k == "yearly_step":
            v = base * (1.06 ** (d.year - s.start.year))
        else:
            v = base
        if i == 0 and "promo_first" in A:
            v = A["promo_first"]
        return round(v, 2) if A.get("satang") else round(v)

    def realize(self, s, paydays):
        dates = self.schedule(s, paydays)
        if "pause" in s.rule:
            p0, p1 = s.rule["pause"]
            dates = [d for d in dates if not (p0 <= d <= p1)]
        if "n_payments" in s.rule and len(dates) >= s.rule["n_payments"]:
            dates = dates[: s.rule["n_payments"]]
            if dates[-1] < END - timedelta(days=31):
                s.status, s.end = "completed", dates[-1]
        catch = False
        for i, d in enumerate(dates):
            amt = self.amount(s, d, i)
            flags = []
            if i == 0 and "promo_first" in s.amount:
                flags.append("promo")
            if d in HOLIDAYS:
                flags.append("on_holiday")
            cp = dict(s.cp) if s.cp else self.merchant_cp(s.key)
            if "pause" in s.rule and d > s.rule["pause"][1] and cp.get("mandate_id"):
                cp["mandate_id"] = s.rule["mandate2"]                # re-registered after pause
                if i and dates[i - 1] < s.rule["pause"][0]:
                    flags.append("resumed_after_pause_new_mandate")
            if s.txn_type in ("bill_payment", "promptpay_out", "transfer_out") and s.kind == "obligation" and self.r() < 0.02 and i < len(dates) - 1:
                s.occ.append((d, amt, "skipped"))
                catch = True
                continue
            post = d
            if s.txn_type == "card_purchase" and self.r() < 0.3:
                post = d + timedelta(days=self.ri(1, 2))
                flags.append("posting_delay")
            if catch:
                amt, catch = amt * 2, False
                flags.append("catch_up_double")
            if cp.get("biller_id") and BILLERS[s.key][4]:
                pm = (d.replace(day=1) - timedelta(days=1))
                cp["ref2"] = f"{pm.year}{pm.month:02d}"                   # billing period
            if cp.get("original_currency") == "USD":
                usd = s.amount["usd"]
                cp["original_amount"] = usd
                amt = round(usd * self.fx_rate(post), 2)
            elif cp.get("original_currency") == "THB":
                cp["original_amount"] = amt
            s.occ.append((post, amt, "|".join(flags)))
            desc = s.rule.get("desc") or self.describe(BILLERS[s.key][1] if s.key in BILLERS else MERCH[s.key][1],
                                                       ref=s.txn_type == "bill_payment")
            if s.txn_type in ("promptpay_out", "transfer_out") and s.kind == "obligation" and self.r() < 0.03:
                a1 = round(amt * self.ch([0.5, 0.6, 0.7]))
                e1 = self.ev(post, a1, s.direction, s.txn_type, s.channel, desc, cp, s.kind, s.sid, "|".join(flags + ["split_part1"]), droppable=s.droppable)
                self.ev(post + timedelta(days=self.ri(0, 3)), amt - a1, s.direction, s.txn_type, s.channel, desc, cp, s.kind, s.sid,
                        "|".join(flags + ["split_part2"]), e1, droppable=s.droppable)
                continue
            e = self.ev(post, amt, s.direction, s.txn_type, s.channel, desc, cp, s.kind, s.sid, "|".join(flags), droppable=s.droppable)
            if e and s.txn_type in ("card_purchase", "direct_debit") and self.r() < 0.012:
                dup = self.ev(post + timedelta(days=self.ri(0, 1)), amt, "DR", s.txn_type, s.channel, desc, cp, s.kind, s.sid, "duplicate_charge")
                rcp = dict(cp)
                self.ev(post + timedelta(days=self.ri(2, 7)), amt, "CR", "refund_in", s.channel, "REVERSAL " + desc[:20], rcp, "refund", None,
                        "reversal_of_duplicate", dup)

    # ------------------------------------------------ series factory
    def S(self, **kw):
        kw.setdefault("start", START)
        kw.setdefault("end", END)
        s = Series(sid=f"S{len(self.series) + 1:06d}", uid=self.uid, **kw)
        self.series.append(s)
        self.user_series.append(s)
        return s

    def maybe_cancel(self, s, p):
        if self.r() < p and (s.end - s.start).days > 120:
            s.end = s.start + timedelta(days=self.ri(90, (s.end - s.start).days - 30))
            s.status = "cancelled"

    def bill_series(self, key, name, kind, period_type, nominal, essential, txn_type, rule, amount, start=START, ref1=None):
        mandate = ("MD" + h(self.uid, key, name, n=8)) if txn_type == "direct_debit" else None
        cp = self.biller_cp(key, ref1 or self.ref1_for(BILLERS[key][3]), mandate)
        ch = "AUTO" if txn_type == "direct_debit" else self.ch(["KPLUS", "KPLUS", "KPLUS", "BRANCH"])
        return self.S(name=name, key=key, category=BILLERS[key][2], kind=kind, period_type=period_type, nominal_days=nominal,
                      essential=essential, txn_type=txn_type, channel=ch, direction="DR", rule=rule, amount=amount, cp=cp, start=start)

    # ------------------------------------------------ one customer
    def build_user(self, k):
        self.uid = uid = f"C{k:05d}"
        self.events, self.eid, self.user_series = [], 0, []
        persona = self.ch(["salaried", "salaried_semimonthly", "freelancer"], [0.75, 0.1, 0.15])
        u = self.r()
        self.open_d = START if u < 0.45 else (START + timedelta(days=self.ri(30, 600)) if u < 0.85 else END - timedelta(days=self.ri(20, 180)))
        salary = round(self.ri(15, 38) * 1000 + self.ri(0, 9) * 100)
        self.salary = salary
        my_name = self.masked_name()
        main_acct = "ACC" + h(uid, "main", n=9)
        accts = [(uid, main_acct, "savings", self.open_d.isoformat(), "main")]
        sav_acct = None
        if self.r() < 0.4:
            sav_acct = "ACC" + h(uid, "sav", n=9)
            accts.append((uid, sav_acct, "savings", self.open_d.isoformat(), "second"))
        ext_own = self.person_cp(uid, "own_ext", bank=self.ch(OTHER_BANKS), name=my_name, proxy=self.ch(["citizen_id", "none"]))
        self.customers.append((uid, my_name, self.open_d.isoformat()))
        self.accounts += accts
        split = "heldout" if self.r() < 0.15 else "train"

        # ---- income
        paydays, job_change = [], persona != "freelancer" and self.r() < 0.18
        if persona == "freelancer":
            for c in range(self.ri(2, 4)):
                comp = f"บจก. {self.ch(['สยาม', 'ไทย', 'กรุงเทพ', 'ดิจิทัล', 'ครีเอทีฟ'])} {h(uid, c, n=3)}"
                cp = self.person_cp(uid, f"client{c}", name=comp, proxy="none")
                s = self.S(name=f"freelance client {c + 1}", key="CLIENT", category="income", kind="income", period_type="irregular",
                           nominal_days=35, essential=False, txn_type="transfer_in", channel="KPLUS", direction="CR",
                           rule={"type": "irregular", "gap": (15, 60), "desc": f"TRANSFER FROM {comp}"},
                           amount={"base": salary / 2.5, "kind": "variable", "sd": 0.5}, cp=cp)
                self.maybe_cancel(s, 0.3)
        else:
            rule = {"type": "semimonthly"} if persona == "salaried_semimonthly" else self.ch(
                [{"type": "monthly", "dom": 25, "shift": "prev"}, {"type": "lastbd"}, {"type": "monthly", "dom": 28, "shift": "prev"},
                 {"type": "monthly", "dom": 1, "shift": "next"}], [0.45, 0.35, 0.1, 0.1])
            comp = self.ch(COMPANIES)
            amt = salary / 2 if persona == "salaried_semimonthly" else salary
            end1 = START + timedelta(days=self.ri(250, 750)) if job_change else END
            cp = self.person_cp(uid, comp, bank=self.ch([OWN_BANK] + OTHER_BANKS), name=f"บจก. {comp}", proxy="none")
            s = self.S(name="salary", key="PAYROLL", category="income", kind="income",
                       period_type="semimonthly" if rule["type"] == "semimonthly" else "monthly",
                       nominal_days=15.2 if rule["type"] == "semimonthly" else 30.4, essential=False, txn_type="payroll_in",
                       channel="AUTO", direction="CR", end=end1, rule={**rule, "desc": f"PAYROLL {comp}"},
                       amount={"base": amt, "kind": "fixed", "changes": [(date(y, 4, 1), round(amt * 1.05 ** (y - 2023))) for y in (2024, 2025, 2026)]}, cp=cp)
            if job_change:
                s.status = "switched"
                s.quirks.append("job_change")
                comp2 = self.ch([c for c in COMPANIES if c != comp])
                cp2 = self.person_cp(uid, comp2, bank=self.ch([OWN_BANK] + OTHER_BANKS), name=f"บจก. {comp2}", proxy="none")
                s2 = self.S(name="salary (new job)", key="PAYROLL", category="income", kind="income", period_type="monthly", nominal_days=30.4,
                            essential=False, txn_type="payroll_in", channel="AUTO", direction="CR", start=end1 + timedelta(days=self.ri(10, 40)),
                            rule={**self.ch([{"type": "monthly", "dom": 25, "shift": "prev"}, {"type": "lastbd"}]), "desc": f"PAYROLL {comp2}"},
                            amount={"base": round(salary * self.ch([1.1, 1.2, 1.35])), "kind": "fixed"}, cp=cp2)
                s2.quirks.append("job_change_new")
            if self.r() < 0.5:
                self.S(name="annual bonus", key="PAYROLL", category="income", kind="income", period_type="annual", nominal_days=365,
                       essential=False, txn_type="payroll_in", channel="AUTO", direction="CR", end=end1,
                       rule={"type": "annual_window", "month": self.ch([12, 4]), "day_lo": 20, "width": 8, "desc": f"BONUS {comp}"},
                       amount={"base": salary * self.ch([0.5, 1.0, 1.5, 2.0]), "kind": "variable", "sd": 0.2}, cp=cp)
        for s in list(self.user_series):
            self.realize(s, [])
            if s.name.startswith("salary"):
                paydays += [d for d, a, f in s.occ]
        paydays = sorted(paydays)
        if persona == "salaried_semimonthly":
            paydays = [p for p in paydays if p.day >= 20]

        # ---- own-account transfers (the main trap: recurring but NOT an obligation)
        if paydays and self.r() < 0.45:
            if sav_acct and self.r() < 0.7:
                cp = self.blank()
                cp.update(counterparty_account_id=sav_acct, counterparty_bank_code=OWN_BANK, counterparty_name=my_name)
                q = "own_account_same_bank"
            else:
                cp, q = dict(ext_own), "own_account_other_bank"
            s = self.S(name="move to own savings", key="OWN", category="own_transfer", kind="own_transfer", period_type="income_linked",
                       nominal_days=30.4, essential=False, txn_type="transfer_out" if cp["promptpay_proxy_type"] is None else "promptpay_out",
                       channel="KPLUS", direction="DR", rule={"type": "after_payday", "lag": (0, 3), "desc": f"TRANSFER TO {my_name}"},
                       amount={"base": self.ri(10, 60) * 100, "kind": "fixed"}, cp=cp)
            s.quirks += [q, "income_linked"]
            s.droppable = True

        # ---- housing
        housing = self.ch(["rent_dorm", "rent_condo", "family"], [0.5, 0.2, 0.3])
        if housing.startswith("rent"):
            rent = self.ri(30, 90) * 100 if housing == "rent_dorm" else self.ri(80, 150) * 100
            landlord = self.person_cp(uid, "landlord")
            combined = housing == "rent_dorm" and self.r() < 0.3
            linked = bool(paydays) and self.r() < 0.35
            rule = {"type": "after_payday", "lag": (0, 3)} if linked else {"type": "monthly", "dom": self.ri(1, 5), "jitter": (-2, 3)}
            rule["desc"] = f"PromptPay To: {landlord['counterparty_name']}" if landlord["promptpay_proxy_type"] else f"TRANSFER TO {landlord['counterparty_name']}"
            amount = {"base": rent, "var": self.ri(600, 1200), "kind": "base_plus_seasonal"} if combined else {"base": rent, "kind": "fixed"}
            if not combined and self.r() < 0.15:
                amount["changes"] = [(START + timedelta(days=self.ri(200, 800)), rent + 500)]
            ttype = "promptpay_out" if landlord["promptpay_proxy_type"] else "transfer_out"
            rs = self.S(name="rent+utilities" if combined else "rent", key="LANDLORD", category="p2p", kind="obligation",
                        period_type="income_linked" if linked else "monthly", nominal_days=30.4, essential=True, txn_type=ttype,
                        channel="KPLUS", direction="DR", rule=rule, amount=amount, cp=landlord)
            rs.quirks += (["rent_and_utilities_one_transfer"] if combined else []) + (["rent_increase"] if amount.get("changes") else []) + (["income_linked"] if linked else [])
            if housing == "rent_dorm" and not combined and self.r() < 0.5:
                us = self.S(name="dorm utilities", key="LANDLORD", category="p2p", kind="obligation", period_type="monthly", nominal_days=30.4,
                            essential=True, txn_type=ttype, channel="KPLUS", direction="DR",
                            rule={"type": "monthly", "dom": self.ri(6, 12), "jitter": (-1, 3), "desc": rule["desc"]},
                            amount={"base": self.ri(600, 1300), "kind": "seasonal"}, cp=landlord)
                us.quirks.append("same_counterparty_as_rent")
            if self.r() < 0.2:
                rs.quirks.append("shared_with_roommate")
                rm_cp = self.person_cp(uid, "roommate")
                rm = self.S(name="roommate pays back half", key="ROOMMATE", category="p2p", kind="reimbursement", period_type="monthly",
                            nominal_days=30.4, essential=False, txn_type="transfer_in", channel="KPLUS", direction="CR",
                            rule={"type": "after_series", "parent": rs, "lag": (0, 5), "desc": f"PromptPay From: {rm_cp['counterparty_name']}"},
                            amount={"base": rent / 2, "kind": "fixed"}, cp=rm_cp)
                rm.linked_sid, rs.user_share = rs.sid, rent / 2
            if housing == "rent_condo" and self.r() < 0.6:
                self.bill_series("JURISTIC", "condo common fee", "obligation", "semiannual", 182.6, True, "bill_payment",
                                 {"type": "monthly", "dom": self.ri(1, 15), "every": 6, "anchor": self.ri(0, 5), "shift": "next"},
                                 {"base": self.ri(18, 45) * 100, "kind": "fixed"})
        elif paydays and self.r() < 0.6:
            parent = self.person_cp(uid, "parent", name=self.masked_name(self.ch(["นาย", "นาง"]), my_name.split()[1]))
            s = self.S(name="support parents", key="PARENT", category="p2p", kind="recurring_transfer", period_type="income_linked",
                       nominal_days=30.4, essential=True, txn_type="promptpay_out" if parent["promptpay_proxy_type"] else "transfer_out",
                       channel="KPLUS", direction="DR", rule={"type": "after_payday", "lag": (0, 2), "desc": f"PromptPay To: {parent['counterparty_name']}"},
                       amount={"base": self.ri(2, 8) * 1000, "kind": "fixed"}, cp=parent)
            s.quirks += ["income_linked", "same_first_name_as_customer"]

        if housing != "rent_dorm":
            self.bill_series(self.ch(["MEA", "PEA"]), "electricity", "obligation", "monthly", 30.4, True, "bill_payment",
                             {"type": "monthly", "dom": self.ri(8, 20), "jitter": (0, 4), "shift": "next"},
                             {"base": self.ri(500, 1600), "kind": "seasonal", "satang": True})
            if self.r() < 0.5:
                self.bill_series("MWA", "water", "obligation", "monthly", 30.4, True, "bill_payment",
                                 {"type": "monthly", "dom": self.ri(10, 22), "jitter": (0, 4), "shift": "next"},
                                 {"base": self.ri(120, 350), "kind": "variable", "sd": 0.15, "satang": True})

        # ---- telecom (incl. second line at the same biller)
        if self.r() < 0.95:
            tel = self.ch(["AIS", "TRUE", "DTAC"])
            tt = self.ch(["bill_payment", "direct_debit"], [0.6, 0.4])
            s = self.bill_series(tel, "mobile postpaid", "obligation", "monthly", 30.4, True, tt,
                                 {"type": "monthly", "dom": self.ri(1, 28), "jitter": (0, 2), "shift": "next"},
                                 {"base": self.ch([299, 399, 499, 599, 699]), "kind": "usage"})
            if self.r() < 0.2:
                s.amount["changes"] = [(START + timedelta(days=self.ri(150, 850)), s.amount["base"] + self.ch([100, 200, -100]))]
                s.quirks.append("plan_change")
            if self.r() < 0.15:
                s2 = self.bill_series(tel, "mobile 2nd line (family)", "obligation", "monthly", 30.4, True, tt,
                                      {"type": "monthly", "dom": self.ri(1, 28), "jitter": (0, 2), "shift": "next"},
                                      {"base": self.ch([199, 299, 399]), "kind": "usage"}, start=START + timedelta(days=self.ri(0, 500)))
                s2.quirks.append("second_contract_same_biller")
        if self.r() < 0.35 and housing != "rent_dorm":
            self.bill_series(self.ch(["AIS_FIBRE", "3BB"]), "home internet", "obligation", "monthly", 30.4, True,
                             self.ch(["bill_payment", "direct_debit"]), {"type": "monthly", "dom": self.ri(1, 28), "shift": "next"},
                             {"base": self.ch([590, 690, 799, 899]), "kind": "fixed"})

        # ---- card subscriptions
        for key, prices, p in [("NETFLIX", [169, 349, 419], 0.45), ("SPOTIFY", [69, 139], 0.35), ("YOUTUBE", [179], 0.25),
                               ("DISNEY", [99, 289], 0.15), ("ICLOUD", [35, 99], 0.35), ("GOOGLE_ONE", [70], 0.15), ("CANVA", [270], 0.1)]:
            if self.r() >= p:
                continue
            st = START if self.r() < 0.6 else START + timedelta(days=self.ri(30, 900))
            base = self.ch(prices)
            amount = {"base": base, "kind": "fixed"}
            if key == "CANVA":
                amount["usd"] = self.ch([7.99, 12.99])
            s = self.S(name=f"{key.lower()} subscription", key=key, category=MERCH[key][4], kind="obligation", period_type="monthly",
                       nominal_days=30.4, essential=False, txn_type="card_purchase", channel="CARD", direction="DR", start=st,
                       rule={"type": "monthly", "dom": min(st.day, 28) if st != START else self.ri(1, 28)}, amount=amount)
            if key == "CANVA":
                s.quirks.append("fx_amount_usd")
            if self.r() < 0.3 and key in ("NETFLIX", "SPOTIFY", "YOUTUBE"):
                s.amount["changes"] = [(START + timedelta(days=self.ri(200, 900)), round(base * self.ch([1.1, 1.15, 1.2])))]
                s.quirks.append("price_increase")
            if st != START and self.r() < 0.3 and key != "CANVA":
                s.amount["promo_first"] = self.ch([1, 19, 49])
                s.quirks.append("promo_first_payment")
            if self.r() < 0.12 and key in ("SPOTIFY", "CANVA", "YOUTUBE"):
                sw = s.start + timedelta(days=self.ri(150, 500))
                if sw < END - timedelta(days=60):
                    s.end, s.status = sw, "switched"
                    s.quirks.append("switched_to_annual")
                    am = {"base": round(base * 10), "kind": "fixed"}
                    if key == "CANVA":
                        am["usd"] = round(amount["usd"] * 10, 2)
                    a = self.S(name=f"{key.lower()} annual plan", key=key, category=MERCH[key][4], kind="obligation", period_type="annual",
                               nominal_days=365.25, essential=False, txn_type="card_purchase", channel="CARD", direction="DR",
                               start=sw + timedelta(days=self.ri(1, 20)), rule={"type": "monthly", "dom": min(sw.day, 28), "every": 12, "anchor": sw.month % 12},
                               amount=am)
                    a.quirks.append("switched_from_monthly")
                    a.linked_sid = s.sid
                    continue
            self.maybe_cancel(s, 0.25)

        # ---- gym: biller membership (mandate) + card purchases at the same brand
        self.gym = None
        if self.r() < 0.25:
            g = self.bill_series("GYM", "gym membership", "obligation", "monthly", 30.4, False, "direct_debit",
                                 {"type": "monthly", "dom": self.ri(1, 28)}, {"base": self.ch([990, 1290, 1590, 1990]), "kind": "fixed"},
                                 start=START + timedelta(days=self.ri(0, 500)))
            g.quirks.append("same_brand_has_card_purchases")
            if self.r() < 0.25:
                p0 = g.start + timedelta(days=self.ri(90, 300))
                g.rule["pause"] = (p0, p0 + timedelta(days=self.ri(45, 100)))
                g.rule["mandate2"] = "MD" + h(uid, "gym2", n=8)
                g.status = "paused_resumed"
                g.quirks.append("pause_resume_new_mandate")
            else:
                self.maybe_cancel(g, 0.35)
            self.gym = g

        # ---- debt
        if self.r() < 0.45:
            self.bill_series("KBANK_CARD", "credit card bill", "obligation", "monthly", 30.4, True, "bill_payment",
                             {"type": "monthly", "dom": self.ri(5, 20), "jitter": (-3, 2), "shift": "next"},
                             {"base": self.ri(15, 80) * 100, "kind": "variable", "sd": 0.45, "satang": True}).quirks.append("variable_amount_hides_merchants")
        for key, n, base, p in [("AEON", self.ch([10, 12]), self.ri(8, 25) * 100, 0.3), ("MTC", 36, self.ri(25, 40) * 100, 0.08),
                                ("KRUNGSRI_AUTO", 48, self.ri(55, 90) * 100, 0.05)]:
            if self.r() < p:
                st = START + timedelta(days=self.ri(-200, 700))
                if st < START:
                    n = max(2, n - (START - st).days // 30)
                self.bill_series(key, f"installment {key.lower()}", "obligation", "monthly", 30.4, True, "direct_debit",
                                 {"type": "monthly", "dom": self.ri(1, 28), "shift": "next", "n_payments": n},
                                 {"base": base, "kind": "fixed"}, start=max(st, START)).quirks.append("fixed_term_installment")
        if self.r() < 0.1:
            self.bill_series("PLOAN", "personal loan", "obligation", "monthly", 30.4, True, "bill_payment",
                             {"type": "monthly", "dom": self.ri(1, 28), "shift": "next"}, {"base": self.ri(15, 50) * 100, "kind": "fixed"})

        # ---- insurance & government (non-monthly), incl. 2 policies at one insurer
        ins = self.ch(["AIA", "MTL", "FWD"])
        if self.r() < 0.22:
            self.bill_series(ins, "insurance quarterly", "obligation", "quarterly", 91.3, True, "direct_debit",
                             {"type": "monthly", "dom": self.ri(1, 28), "every": 3, "anchor": self.ri(0, 2), "shift": "next"},
                             {"base": self.ri(20, 50) * 100, "kind": "fixed"})
        if self.r() < 0.25:
            s = self.bill_series(ins, "insurance annual", "obligation", "annual", 365.25, True, self.ch(["direct_debit", "bill_payment"]),
                                 {"type": "monthly", "dom": self.ri(1, 28), "every": 12, "anchor": self.ri(0, 11), "shift": "next"},
                                 {"base": self.ri(80, 250) * 100, "kind": "fixed"})
            if any(x.name == "insurance quarterly" for x in self.user_series):
                s.quirks.append("second_contract_same_biller")
        if self.r() < 0.12:
            self.bill_series("VIRIYAH", "motor insurance semiannual", "obligation", "semiannual", 182.6, True, "bill_payment",
                             {"type": "monthly", "dom": self.ri(1, 28), "every": 6, "anchor": self.ri(0, 5), "shift": "next"},
                             {"base": self.ri(25, 60) * 100, "kind": "fixed"})
        if self.r() < 0.3:
            self.bill_series("SLF", "student loan (กยศ)", "obligation", "annual", 365.25, True, "bill_payment",
                             {"type": "annual_window", "month": 6, "day_lo": 15, "width": 20},
                             {"base": self.ri(15, 45) * 100, "kind": "yearly_step"}).quirks.append("deadline_window_jul5")
        if self.r() < 0.2:
            self.bill_series("DLT", "vehicle tax", "obligation", "annual", 365.25, True, "bill_payment",
                             {"type": "annual_window", "month": self.ri(1, 11), "day_lo": 1, "width": 40},
                             {"base": self.ri(500, 1200), "kind": "variable", "sd": 0.05, "satang": True})

        # ---- savings product & lifestyle
        if paydays and self.r() < 0.4:
            s = self.bill_series("KASSET", "fund DCA", "recurring_transfer", "income_linked", 30.4, False, "direct_debit",
                                 {"type": "after_payday", "lag": (0, 2)}, {"base": self.ri(1, 5) * 1000, "kind": "fixed"})
            s.quirks.append("income_linked")
            self.maybe_cancel(s, 0.2)
        if self.r() < 0.15:
            s = self.S(name="weekly badminton court", key="COURT", category="sports", kind="obligation", period_type="weekly", nominal_days=7,
                       essential=False, txn_type="qr_payment", channel="KPLUS", direction="DR",
                       rule={"type": "weekly", "weekday": self.ch([5, 6, 2]), "skip": 0.2}, amount={"base": self.ch([150, 200, 250]), "kind": "fixed"})
            s.quirks.append("weekly_with_random_skips")
            s.droppable = True
            self.maybe_cancel(s, 0.3)
        if self.r() < 0.08:
            self.S(name="biweekly cleaning", key="CLEAN", category="home_services", kind="obligation", period_type="biweekly", nominal_days=14,
                   essential=False, txn_type="card_purchase", channel="CARD", direction="DR", start=START + timedelta(days=self.ri(0, 300)),
                   rule={"type": "weekly", "weekday": self.ri(0, 5), "every": 2}, amount={"base": self.ch([450, 550, 650]), "kind": "fixed"})
        if self.r() < 0.6:
            s = self.S(name="haircut", key="BARBER", category="personal_care", kind="obligation", period_type="irregular", nominal_days=37,
                       essential=False, txn_type="qr_payment", channel="KPLUS", direction="DR", rule={"type": "irregular", "gap": (24, 50)},
                       amount={"base": self.ch([150, 250, 400]), "kind": "variable", "sd": 0.1})
            s.quirks.append("irregular_gap")
            s.droppable = True
        if self.r() < 0.5:
            wcp = self.blank()
            wcp.update(counterparty_account_id="W" + h(uid, "wallet", n=9), counterparty_bank_code=None,
                       counterparty_name=f"TRUEMONEY - {my_name}", promptpay_proxy_type="ewallet_id")
            s = self.S(name="wallet top-up", key="WALLET", category="wallet", kind="obligation", period_type="irregular", nominal_days=9,
                       essential=False, txn_type="promptpay_out", channel="KPLUS", direction="DR",
                       rule={"type": "irregular", "gap": (3, 15), "desc": f"PromptPay To: TRUEMONEY {my_name}"},
                       amount={"base": 300, "kind": "variable", "sd": 0.6}, cp=wcp)
            s.quirks += ["wallet_hides_merchants", "wallet_in_customer_name"]
            s.droppable = True

        self.extra_series(uid, my_name, paydays)
        for s in self.user_series:
            if s.kind != "income" and s.rule["type"] != "after_series":
                self.realize(s, paydays)
        for s in self.user_series:
            if s.rule["type"] == "after_series":
                self.realize(s, paydays)

        self.discretionary(uid, paydays, sav_acct, ext_own, my_name)
        self.finalize(uid, main_acct, salary, sav_acct, ext_own, my_name)
        self.users_truth.append((uid, persona, salary, housing, job_change, split,
                                 "cold_start" if self.open_d > END - timedelta(days=200) else ("partial" if self.open_d > START else "full")))

    def extra_series(self, uid, my_name, paydays):
        """Hook for subclasses (v3 adds interval series); must not draw random numbers here."""

    # ------------------------------------------------ discretionary spending & distractors
    def discretionary(self, uid, paydays, sav_acct, ext_own, my_name):
        rng = self.rng
        lam = rng.uniform(1.3, 2.8) * min(1.3, max(0.7, self.salary / 25000))
        branches = ["M" + h("7ELEVEN", "branch", uid, i, n=9) for i in range(self.ri(3, 6))]
        pdset = {p + timedelta(days=k) for p in paydays for k in range(4)}
        cats = ["7ELEVEN", "CAFE", "LINEMAN", "GRAB", "SHOPEE", "oneoff", "atm", "p2p_small"]
        w = np.array([0.33, 0.12, 0.13, 0.1, 0.06, 0.18, 0.03, 0.05]); w /= w.sum()
        mu = {"7ELEVEN": 85, "CAFE": 65, "LINEMAN": 220, "GRAB": 140, "SHOPEE": 420}
        d = self.open_d
        while d <= END:
            n = rng.poisson(lam * (1.4 if d in pdset else 1.0) * (1.15 if d.weekday() >= 4 else 1.0))
            for _ in range(n):
                c = cats[int(rng.choice(len(cats), p=w))]
                if c == "CAFE" and d.weekday() >= 5:
                    c = "7ELEVEN"
                if c in mu:
                    amt = round(max(15, rng.lognormal(np.log(mu[c]), 0.55)))
                    tt = self.ch(["card_purchase", "qr_payment"])
                    cp = self.merchant_cp(c, mid_override=self.ch(branches) if c == "7ELEVEN" else None)
                    cp["original_amount"] = amt
                    self.ev(d, amt, "DR", tt, "CARD" if tt == "card_purchase" else "KPLUS", self.describe(MERCH[c][1]), cp, "habitual", droppable=True)
                elif c == "oneoff":
                    shop, mcc = self.ch(ONEOFF_SHOPS)
                    amt = round(max(40, rng.lognormal(np.log(450), 0.8)))
                    cp = self.blank()
                    cp.update(mcc=mcc, merchant_id="M" + h(shop, n=9), merchant_name=shop, original_currency="THB", original_amount=amt)
                    tt = self.ch(["card_purchase", "qr_payment"])
                    self.ev(d, amt, "DR", tt, "CARD" if tt == "card_purchase" else "KPLUS", shop + (f" {self.ri(1, 99):02d}" if self.r() < 0.4 else ""),
                            cp, "one_off", droppable=True)
                elif c == "atm":
                    self.ev(d, self.ch([500, 1000, 1000, 2000]), "DR", "atm_withdrawal", "ATM", f"ATM WITHDRAWAL {self.ri(1000, 9999)}", self.blank(),
                            "one_off", droppable=True)
                else:
                    cp = self.person_cp(uid, f"friend{self.ri(0, 9)}")
                    self.ev(d, round(rng.lognormal(np.log(250), 0.7)), "DR", "promptpay_out", "KPLUS", f"PromptPay To: {cp['counterparty_name']}",
                            cp, "one_off", droppable=True)
            if self.gym and self.gym.start <= d <= self.gym.end and self.r() < 0.05:
                amt = self.ch([65, 90, 120, 150])
                cp = self.merchant_cp("GYM_SHOP")
                cp["original_amount"] = amt
                self.ev(d, amt, "DR", "card_purchase", "CARD", self.describe(MERCH["GYM_SHOP"][1]), cp, "one_off", flags="same_brand_as_recurring",
                        droppable=True)
            if self.r() < 0.012:
                total = round(rng.uniform(1500, 6000))
                shop, mcc = self.ch([("MK RESTAURANT", "5812"), ("SHABUSHI", "5812"), ("BAR B Q PLAZA", "5812")])
                cp = self.blank()
                cp.update(mcc=mcc, merchant_id="M" + h(shop, n=9), merchant_name=shop, original_currency="THB", original_amount=total)
                bill = self.ev(d, total, "DR", "card_purchase", "CARD", shop, cp, "one_off", flags="shared_bill_paid_in_full", droppable=True)
                k = self.ri(2, 4)
                for j in range(k):
                    fcp = self.person_cp(uid, f"friend{j}")
                    self.ev(d + timedelta(days=self.ri(0, 3)), round(total / (k + 1)), "CR", "transfer_in", "KPLUS",
                            f"PromptPay From: {fcp['counterparty_name']}", fcp, "reimbursement", flags="shared_bill_payback", link=bill)
            if (d.month in (4, 12) and self.r() < 0.02) or self.r() < 0.002:
                key = self.ch(["AGODA", "AIRASIA"])
                amt = round(rng.uniform(1500, 12000))
                cp = self.merchant_cp(key)
                cp["original_amount"] = amt
                self.ev(d, amt, "DR", "card_purchase", "CARD", self.describe(MERCH[key][1]), cp, "one_off", flags="large_seasonal", droppable=True)
            if self.r() < 0.004:
                amt = round(rng.uniform(400, 4000))
                cp = self.merchant_cp("CLINIC")
                cp["original_amount"] = amt
                self.ev(d, amt, "DR", "card_purchase", "CARD", self.describe(MERCH["CLINIC"][1]), cp, "one_off", flags="unexpected_expense")
            d += timedelta(days=1)

    # ------------------------------------------------ balance pass + write
    def finalize(self, uid, main_acct, salary, sav_acct, ext_own, my_name):
        ev = sorted(self.events, key=lambda e: (e["d"], e["t"], e["eid"]))
        bal = round(salary * float(self.rng.uniform(0.5, 2.0)), 2)
        dropped, eid2tid, rows, labels = set(), {}, [], []
        cushion = max(300.0, 0.3 * salary)

        def write(e, bal_after):
            self.txn_no += 1
            tid = f"T{self.txn_no:09d}"
            eid2tid[e["eid"]] = tid
            cp = e["cp"] or self.blank()
            rows.append([tid, uid, main_acct, f"{e['d'].isoformat()} {e['t']}", f"{e['amount']:.2f}", e["direction"], e["txn_type"],
                         e["channel"], f"{bal_after:.2f}", e["description"]] + [cp.get(k) for k in CP_KEYS])
            labels.append([tid, e["kind"], e["sid"], e["flags"], eid2tid.get(e["link"]) if e["link"] else None])
            if e["sid"]:
                self.series_occ.append((e["sid"], tid, e["d"], e["amount"], e["flags"], eid2tid.get(e["link"]) if e["link"] else None))

        for e in ev:
            if e["link"] in dropped and e["kind"] == "reimbursement":
                dropped.add(e["eid"])
                continue
            if e["direction"] == "CR":
                bal += e["amount"]
                write(e, bal)
                continue
            floor = cushion if e["kind"] in ("habitual", "one_off") else 300.0   # cut discretionary spend first, keep recurring needs
            if e["droppable"] and bal - e["amount"] < floor:
                dropped.add(e["eid"])
                continue
            if bal < e["amount"]:
                need = math.ceil((e["amount"] - bal + self.ri(500, 3000)) / 100) * 100
                src = self.blank()
                if sav_acct:
                    src.update(counterparty_account_id=sav_acct, counterparty_bank_code=OWN_BANK, counterparty_name=my_name)
                else:
                    src = dict(ext_own)
                self.eid += 1
                top = {"eid": self.eid, "d": e["d"], "t": e["t"], "amount": need, "direction": "CR",
                       "txn_type": "transfer_in" if src.get("promptpay_proxy_type") is None else "promptpay_in", "channel": "KPLUS",
                       "description": f"TRANSFER FROM {my_name}", "cp": src, "kind": "own_transfer", "sid": None,
                       "flags": "topup_when_short", "link": None}
                bal += need
                write(top, bal)
            bal -= e["amount"]
            write(e, bal)
        self.tx_writer().writerows(rows)
        self.lb_writer.writerows(labels)

    def tx_writer(self):
        part = (int(self.uid[1:]) - 1) // self.part_size + 1
        if part != self.cur_part:
            if self.tx_fh:
                self.tx_fh.close()
            self.cur_part = part
            lo, hi = (part - 1) * self.part_size + 1, min(part * self.part_size, self.n_users)
            self.tx_fh = gzip.open(f"{self.out}/transactions_part{part}_C{lo:05d}-C{hi:05d}.csv.gz", "wt", newline="", encoding="utf-8")
            self.tx_w = csv.writer(self.tx_fh)
            self.tx_w.writerow(TX_COLS)
        return self.tx_w

    def run(self, part_size):
        self.part_size, self.cur_part, self.tx_fh = part_size, 0, None
        lbf = gzip.open(f"{self.out}/txn_labels.csv.gz", "wt", newline="", encoding="utf-8")
        self.lb_writer = csv.writer(lbf)
        self.lb_writer.writerow(LB_COLS)
        for k in range(1, self.n_users + 1):
            self.build_user(k)
            if k % 100 == 0:
                print(f"  {k} customers done, {self.txn_no:,} txns", flush=True)
        self.tx_fh.close()
        lbf.close()
        self.write_truth_and_snapshots()

    def write_truth_and_snapshots(self):
        out = self.out
        pd.DataFrame(self.customers, columns=["customer_id", "customer_name", "first_account_open_date"]).to_csv(f"{out}/customers.csv", index=False)
        pd.DataFrame(self.accounts, columns=["customer_id", "account_id", "account_type", "open_date", "role"]).drop(columns="role").to_csv(
            f"{out}/accounts.csv", index=False)
        ut = pd.DataFrame(self.users_truth, columns=["customer_id", "persona", "base_monthly_salary", "housing", "job_change", "user_split", "history_class"])
        ut.to_csv(f"{out}/users_truth.csv", index=False)
        split = dict(zip(ut.customer_id, ut.user_split))
        opened = {c[0]: date.fromisoformat(c[2]) for c in self.customers}

        occ = pd.DataFrame(self.series_occ, columns=["sid", "tid", "d", "amount", "flags", "link"])
        p2 = occ[occ["flags"].str.contains("split_part2", na=False)].groupby("link").amount.sum()
        occ = occ[~occ["flags"].str.contains("duplicate_charge|split_part2", na=False)].copy()
        occ["amount"] = occ.amount + occ.tid.map(p2).fillna(0)
        occ_by = {k: g.sort_values("d") for k, g in occ.groupby("sid")}

        rows, snaps = [], []
        for s in self.series:
            g = occ_by.get(s.sid)
            n = 0 if g is None else len(g)
            rows.append((s.sid, s.uid, s.name, s.key, s.category, s.kind, s.period_type, s.nominal_days, s.essential, s.txn_type,
                         s.direction, s.status, ";".join(s.quirks), s.user_share, s.linked_sid, n,
                         g.d.iloc[0] if n else None, g.d.iloc[-1] if n else None, s.amount["base"]))
            if not n:
                continue
            dates, amts = g.d.tolist(), g.amount.tolist()
            ended = s.status in ("cancelled", "completed", "switched")
            darr = np.array(dates, dtype="datetime64[D]")
            for y, mo in month_iter(dates[0], END):
                sd = date(y, mo, 1)
                if sd <= dates[0] or sd < opened[s.uid]:
                    continue
                i = int(np.searchsorted(darr, np.datetime64(sd), side="left"))
                if (sd - dates[i - 1]).days > max(3 * s.nominal_days, 90):
                    continue
                prior = dates[max(0, i - 6):i]                         # last N=6 occurrences
                gaps = np.diff([x.toordinal() for x in prior])
                nxt = dates[i] if i < len(dates) else None
                snaps.append((s.uid, s.sid, sd, i, (sd - dates[i - 1]).days, float(np.median(gaps)) if len(gaps) else None,
                              float(np.std(gaps) / np.mean(gaps)) if len(gaps) > 1 else None, dates[i - 1], amts[i - 1],
                              float(np.median(amts[max(0, i - 3):i])), (nxt - sd).days if nxt else (END - sd).days, 1 if nxt else 0,
                              nxt, amts[i] if nxt else None, 1 if (nxt or not ended) else 0, s.period_type, s.essential, s.kind, split[s.uid]))
        pd.DataFrame(rows, columns=["series_id", "customer_id", "name", "counterparty_key", "category", "kind", "period_type",
                                    "nominal_period_days", "essential", "txn_type", "direction", "status", "quirks", "user_share_amount",
                                    "linked_series_id", "n_occurrences", "first_date", "last_date", "base_amount"]).to_csv(f"{out}/series_truth.csv", index=False)
        sn = pd.DataFrame(snaps, columns=["customer_id", "series_id", "snapshot_date", "n_prior", "days_since_last", "median_gap_prior",
                                          "gap_cv_prior", "last_date", "last_amount", "amount_med3", "duration_days", "event_observed",
                                          "next_date", "next_amount", "active_label", "truth_period_type", "truth_essential", "truth_kind", "user_split"])
        sn["snapshot_date"] = pd.to_datetime(sn.snapshot_date)
        sn["next_date"] = pd.to_datetime(sn.next_date)

        def censor(df, cut):
            df = df.copy()
            cut = pd.Timestamp(cut)
            late = df.next_date.isna() | (df.next_date >= cut)
            df.loc[late, "duration_days"] = (cut - df.loc[late, "snapshot_date"]).dt.days
            df.loc[late, "event_observed"] = 0
            df.loc[late, ["next_amount", "next_date"]] = None
            # not seen again before the cutoff and not truly ended -> still-alive status is unknown at the cutoff
            df.loc[late & (df.active_label == 1), "active_label"] = np.nan
            return df

        tr = sn[(sn.user_split == "train") & (sn.snapshot_date < pd.Timestamp(SPLIT_TRAIN_END))]
        ca = sn[(sn.user_split == "train") & (sn.snapshot_date >= pd.Timestamp(SPLIT_TRAIN_END)) & (sn.snapshot_date < pd.Timestamp(SPLIT_CALIB_END))]
        te = sn[(sn.snapshot_date >= pd.Timestamp(SPLIT_CALIB_END)) & (sn.snapshot_date < pd.Timestamp(SPLIT_TEST_END))]
        censor(tr, SPLIT_TRAIN_END).to_csv(f"{out}/snapshots_train.csv.gz", index=False)
        censor(ca, SPLIT_CALIB_END).to_csv(f"{out}/snapshots_calib.csv.gz", index=False)
        te.to_csv(f"{out}/snapshots_test.csv.gz", index=False)
        print(f"series={len(rows):,} snapshots train={len(tr):,} calib={len(ca):,} test={len(te):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data_v2")
    ap.add_argument("--part-size", type=int, default=250)
    a = ap.parse_args()
    g = Gen(a.users, a.seed, a.out)
    g.run(a.part_size)
    print(f"customers={a.users} txns={g.txn_no:,}")


if __name__ == "__main__":
    main()
