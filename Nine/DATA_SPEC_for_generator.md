# SafeBelt synthetic bank data — generator specification

You are asked to build a **synthetic transaction dataset** for SafeBelt, a "safe to spend" feature for a Thai mobile banking app
(customers: Thai first-jobbers and young workers). A separate model will read the transactions, detect each customer's
**recurring bills**, and predict for every bill: P (still active?), T (next date), A (amount), C (confidence). Your data is
used to train and validate that model, so it must have **realistic transactions** plus a **truth table** that says what
every transaction really is.

Do not hand-write rows. **Write a Python 3 script** (pandas + numpy only) that generates the files below from a random seed,
so the same seed always gives the same data. Target size: **3,000 customers** (about 2.5–3 million transactions). Make
`--users`, `--seed` and `--out` command-line options.

---

## 1. Files to produce (all required)

All files are UTF-8 (Thai text allowed), comma-separated, with a header row. `.csv` or `.csv.gz` are both fine.

| File | What it is | Seen by the model? |
|---|---|---|
| `transactions.csv.gz` | every transaction of every customer | **yes** (inputs) |
| `customers.csv` | one row per customer | **yes** |
| `accounts.csv` | one row per bank account | **yes** |
| `txn_labels.csv.gz` | the truth for every transaction (1 row per transaction) | no — validation only |
| `series_truth.csv` | the truth for every recurring series | no — validation only |
| `users_truth.csv` | the truth for every customer | no — validation only |

Truth must **never** leak into the three input files (no column in them may reveal labels).

---

## 2. Time frame and split (fixed)

- Data period: **2024-01-01 00:00:00 to 2026-09-30 23:59:59**.
- Each customer's account opens on a random date **1 month to 24 months before 2026-09-30**; no transactions before it.
- About **15% of customers are "heldout"** (never used for training), the rest "train" (`users_truth.user_split`).
- The model trains on dates before 2025-10-01, calibrates on 2025-10-01 .. 2026-01-31, and is tested from 2026-02-01, with
  each customer's **June 2026 payday** as the validation day (see 2b); outcomes are read until 2026-09-30. So the data
  must continue normally (bills, salary, spending) through 2026-09-30.
- Salaried customers need at least 2 salary credits before their June 2026 payday, so their account should open no
  later than early April 2026 (a few later-opening customers are fine; they use the 30-day fallback window).

---

## 2b. Validation on payday (important)

The model is validated **on each customer's own payday**, not on a fixed date:

- **Validation day ("today") = the day after the customer's first salary credit in June 2026.** The salary credit is found
  only from the standard fields: `direction = CR` and `txn_type = payroll_in`.
- **Window = from today until the customer's next payday**, where next payday = last payday + the customer's usual gap
  between salary credits (the median gap of their last few `payroll_in` credits; must be 10–40 days).
- On that day the app computes: balance today (after the salary) − upcoming bills until next payday = spendable;
  safety buffer = 20% of spendable; safe to spend = spendable − buffer. It is checked against the bills really paid
  in the window.
- Customers with no `payroll_in` in June 2026 (freelancers) use 2026-06-01 and a 30-day window.

So the data must satisfy:

1. Every salaried customer has **one `payroll_in` credit per pay date**, on a regular schedule (monthly or twice a month),
   description `PAYROLL <company>`, `channel = AUTO`, posted early in the day (00:00–06:00) so the balance that day already
   includes the salary. Bonuses are separate `payroll_in` credits on other dates.
2. Every salaried customer has a **June 2026 salary credit** and **at least 2 salary credits before it**
   (accounts opened by early April 2026 at the latest for salaried customers, or they fall back to the 30-day window).
3. Salary **continues normally after June** (July, August, September 2026), and so do bills and spending, so the next
   payday and the window's real bills exist in the data.
4. Paydays follow realistic Thai rules: 25th or the business day before, last business day, 28th, or 1st (next business
   day); twice-a-month pay on the 15th and the last business day. Some customers change job (new employer and payday).
5. `balance_after` on the salary credit must include the salary (it is processed like any other transaction in time order).

---

## 3. `transactions.csv.gz` — exactly these 23 columns, in this order

| # | Column | Type / format | Rules |
|---|---|---|---|
| 1 | `txn_id` | string, unique | e.g. `T000000001` |
| 2 | `customer_id` | string | e.g. `C00001`; must exist in customers.csv |
| 3 | `account_id` | string | must be one of this customer's accounts in accounts.csv |
| 4 | `posted_datetime` | `YYYY-MM-DD HH:MM:SS` | sorted per account |
| 5 | `amount` | number > 0, 2 decimals | always positive; the sign is in `direction` |
| 6 | `direction` | `DR` or `CR` | DR = money out, CR = money in |
| 7 | `txn_type` | enum (below) | |
| 8 | `channel` | `KPLUS`, `CARD`, `AUTO`, `ATM`, `BRANCH` | AUTO for payroll and direct debit |
| 9 | `balance_after` | number, 2 decimals | **= previous balance_after ± amount** for that account; **never negative** |
| 10 | `description` | free text | what a bank statement shows (Thai or English), e.g. `PAYROLL SCG`, `PromptPay To: นาย สมชาย ก****`, `NETFLIX.COM`, `MEA ELECTRICITY REF12345678` |
| 11 | `counterparty_account_id` | string or empty | the other account for transfers/PromptPay |
| 12 | `counterparty_bank_code` | 3-digit **string** or empty | e.g. `"004"`, `"014"` — keep leading zeros |
| 13 | `counterparty_name` | string or empty | masked Thai name like `นาย สมชาย ก****`, or company `บจก. ...` |
| 14 | `promptpay_proxy_type` | `mobile`, `citizen_id`, `ewallet_id` or empty | |
| 15 | `biller_id` | string or empty | for bill payments / direct debits, e.g. `BL20001` |
| 16 | `ref1` | **string** or empty | contract / account number at the biller; **stable per contract** |
| 17 | `ref2` | **string** or empty | billing period `YYYYMM` for monthly utility/telecom bills |
| 18 | `mandate_id` | string or empty | for direct debits; stable per mandate |
| 19 | `mcc` | 4-digit **string** or empty | merchant category code for card/QR purchases, e.g. `"5411"` |
| 20 | `merchant_id` | string or empty | |
| 21 | `merchant_name` | string or empty | e.g. `NETFLIX.COM`, `7-ELEVEN`, `SHOPEE` |
| 22 | `original_currency` | `THB`, `USD` or empty | |
| 23 | `original_amount` | number or empty | amount in the original currency |

**`txn_type` values** (use exactly these):

| Money in (CR) | Money out (DR) |
|---|---|
| `payroll_in` — salary and bonus from employer | `bill_payment` — bill paid in app (biller_id + ref1) |
| `transfer_in` — transfer from a person / client / own other account | `direct_debit` — automatic debit (biller_id + ref1 + mandate_id) |
| `promptpay_in` — PromptPay received | `card_purchase` — card payment at a merchant |
| `refund_in` — refund / reversal | `qr_payment` — QR payment at a merchant |
| | `promptpay_out` — PromptPay to a person / wallet |
| | `transfer_out` — bank transfer to a person |
| | `atm_withdrawal` — cash withdrawal |

**How fields are filled per kind of payment** (the model groups payments by these, so be consistent):
- Bill payment / direct debit → `biller_id` + `ref1` (+ `mandate_id` for direct debit); merchant and counterparty empty.
- Card / QR purchase → `merchant_name`, `merchant_id`, `mcc`, `original_currency`, `original_amount`; counterparty empty.
- Transfer / PromptPay → `counterparty_account_id`, `counterparty_bank_code`, `counterparty_name`, `promptpay_proxy_type`.
- Salary → `txn_type = payroll_in`, `channel = AUTO`, description `PAYROLL <company>`, counterparty = the employer.
- A transfer to the customer's **own** account must use one of their `account_id`s as `counterparty_account_id`
  (same bank) or their own `customer_name` as `counterparty_name` (other bank). These are not bills.

---

## 4. `customers.csv` and `accounts.csv`

- `customers.csv`: `customer_id, customer_name, first_account_open_date` (date `YYYY-MM-DD`; name masked like `น.ส. พิมพ์ ก****`).
- `accounts.csv`: `customer_id, account_id, account_type, open_date` (`account_type` e.g. `savings`). Most customers have one
  main account; about 40% also have a second savings account that receives own-account transfers.

---

## 5. `txn_labels.csv.gz` — one row per transaction

| Column | Values |
|---|---|
| `txn_id` | same ids as transactions (1:1, same count) |
| `gt_kind` | `obligation` (a bill), `recurring_transfer` (regular transfer that is a commitment, e.g. support parents, fund DCA), `income`, `own_transfer`, `habitual` (everyday spending), `one_off` (one-time purchase), `habit` (regular-looking spending that is NOT a bill, e.g. weekly groceries), `reimbursement` (someone pays you back), `refund` |
| `gt_series_id` | id of the recurring series this transaction belongs to (e.g. `S000123`), empty if none |
| `gt_flags` | `\|`-separated tags, empty if none. **Must use these exact tags when they apply:** `duplicate_charge` (a duplicated charge that is later refunded), `split_part1` / `split_part2` (one bill paid in two transfers; part2 is the second piece), `topup_when_short` (money moved in from own savings because the balance was too low). Other tags are free, e.g. `posting_delay`, `paid_late`, `paid_at_counter`, `catch_up_double`, `on_holiday` |
| `gt_link_txn_id` | linked txn_id (e.g. the refund's original charge, split part2 → part1), else empty |

Every transaction that belongs to a recurring series (bills, salary, recurring transfers, own-account savings
transfers, regular habits) must carry its `gt_series_id`.

---

## 6. `series_truth.csv` — one row per recurring series

Required columns (others welcome):

| Column | Values |
|---|---|
| `series_id` | e.g. `S000123` (matches `gt_series_id`) |
| `customer_id` | |
| `name` | readable name: `rent`, `electricity`, `netflix subscription`, `insurance annual`, `medicine refill`, `weekly groceries` … |
| `kind` | `obligation`, `recurring_transfer`, `income`, `own_transfer`, `reimbursement`, `habit` |
| `category` | e.g. `p2p`, `utilities`, `telecom`, `streaming`, `insurance`, `installment`, `credit_card`, `government`, `groceries`, `pharmacy`, `pets`, … **use `wallet` for e-wallet top-ups and `personal_care` for haircuts** (these two are treated as "not a bill") |
| `status` | `active`, `cancelled`, `completed` (installment finished), `switched` (moved to another plan/job), `paused_resumed` |
| `cycle_type` | `calendar` (same date each period), `interval` (every N days, not tied to the calendar), `income_linked` (right after payday), `irregular` |
| `true_period_days` | the real cycle in days: 7, 14, 30.4, 91.3, 182.6, 365.25, or any N for interval series (e.g. 28, 45); the mean gap for irregular series |
| `essential` | `True` / `False` |
| `quirks` | `;`-separated notes, e.g. `price_increase;payee_account_changed` |

A **bill**, for validation, is a series with `kind` in (`obligation`, `recurring_transfer`), `true_period_days >= 7`,
and `category` not in (`wallet`, `personal_care`).

---

## 7. `users_truth.csv` — one row per customer

Required: `customer_id`, `user_split` (`train` or `heldout`).
Recommended: `persona` (`salaried`, `salaried_semimonthly`, `freelancer`), `base_monthly_salary`, `housing`,
`spending_profile`, `history_class`.

---

## 8. What the data should contain (realism)

**Income**
- Salary ฿15,000–100,000 a month (log-normal, median about ฿28,000). About 75% monthly salary (paid on the 25th or the
  previous business day, the last business day, the 28th, or the 1st), 10% paid twice a month, 15% freelancers with
  irregular client transfers. A 3–5% raise each April; some customers get a yearly bonus (December or April); some change
  job (new employer, new payday).

**Recurring bills** (each customer gets a random mix; bigger salary → bigger rent, card bill, insurance and more subscriptions)
- Monthly by date: rent (paid by PromptPay/transfer to a landlord), dorm utilities to the same landlord (smaller amount),
  electricity (MEA/PEA, seasonal: higher March–June), water, mobile postpaid (sometimes a second line), home internet,
  credit-card bill (varies a lot), personal loan, installments that end after N months, gym (direct debit, may pause),
  streaming/cloud subscriptions (Netflix, Spotify, YouTube, Disney+, iCloud, Google One; price increases; one in USD).
- Quarterly / 6-monthly / yearly: insurance (AIA, MTL, FWD), motor insurance, condo common fee, student loan (กยศ, June–July
  window), vehicle tax.
- Right after payday: support for parents, fund DCA, transfer to own savings (own_transfer — not a bill).
- Every N days (interval): drinking water delivery (10–20 days), pet food (18–35), medicine refill (28/30/56/84), contact
  lenses (30–60), cooking gas (35–60), car wash (12–25), laundry (6–10). Add some skipped cycles.
- Irregular: haircut (24–50 days, category `personal_care`), e-wallet top-ups (3–15 days, category `wallet`).
- Weekly / biweekly: badminton court booking (with skips), house cleaning.

**Hard cases** (important — they test the model)
- A bill sometimes paid at the 7-Eleven counter (merchant `7-ELEVEN`, mcc `5499`) instead of the biller.
- Shopee PayLater installment (fixed monthly) at merchant `SHOPEE`, mixed with normal Shopee shopping.
- Rotating savings club (แชร์): fixed monthly PromptPay to a friend.
- Landlord changes bank account mid-history; rent due day moves (e.g. 1st → 5th).
- Late payers (some bills 4–15 days late), occasional skipped month followed by a double payment, payments split into
  two transfers, duplicate card charges with a refund 2–7 days later, card posting delay of 1–2 days.
- Regular-looking habits that are **not** bills (`kind = habit`): weekly groceries at the same supermarket, shopping
  right after payday, repeated transfers to the same close friend.
- Cancelled subscriptions, finished installments, a monthly plan switched to yearly.

**Everyday spending (not recurring)**
- Daily habits: 7-Eleven, café, food delivery (LINE MAN), Grab, Shopee, small transfers to friends, ATM withdrawals,
  one-off shops (restaurants, supermarkets, clothes, electronics, cinema), seasonal travel (Agoda, AirAsia in April and
  December), unexpected clinic visits, a restaurant bill paid in full and friends paying back their share.
- Give customers different spending styles (frugal, foodie, shopper, commuter, café lover, traveller, family supporter,
  heavy spender) so spending amounts and mixes differ.

**Balance rule**
- Start each account with a balance of 0.5–2 months of salary. Process transactions in time order; when everyday spending
  would push the balance under a small cushion, skip that spending; when a bill would make it negative, first add a
  `transfer_in` from the customer's own savings account (flag `topup_when_short`, `gt_kind = own_transfer`).
  `balance_after` must never be negative.

---

## 9. Self-check before handing over (please run and report)

1. transactions has exactly the 23 columns in the order above; `txn_id` unique; `amount > 0`.
2. `balance_after` = previous `balance_after` + amount (CR) or − amount (DR) for every account (±0.01), and never < 0.
3. Every `account_id` in transactions belongs to that customer in accounts.csv.
4. txn_labels has exactly one row per transaction; every `gt_series_id` exists in series_truth.
5. No truth column (`gt_*`, `truth_*`, `kind`, `series_id`, labels) appears in transactions, customers or accounts.
6. Codes (`counterparty_bank_code`, `ref1`, `ref2`, `mcc`) are strings (leading zeros kept).
7. ~15% of customers are `heldout`; data runs to 2026-09-30.
8. **Payday check:** for every non-freelancer, find the first `payroll_in` in June 2026 and the next one after it. Report
   the share of salaried customers that have (a) a June 2026 salary credit, (b) at least 2 salary credits before it,
   (c) a next salary credit 10–40 days later — all three should be close to 100%. Report the median days between the
   June payday and the next payday (about 30 for monthly pay, about 15 for twice-a-month pay).
9. Print a summary: customers, transactions, series by `kind` × `cycle_type`, salary distribution, history length.

Deliver the generator script and the six files in one folder (e.g. `data_v7/`).
