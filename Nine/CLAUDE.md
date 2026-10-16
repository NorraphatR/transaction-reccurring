# SafeBelt / Safe-to-Spend — project context for Claude Code

Competition project (KBTG). Predictive feature for K PLUS, target users: Thai first jobbers.
Goal: show "how much can I safely spend" = Balance − Upcoming Obligations − Smart Reserve − Safety Buffer.

## Main goal and output contract
The model detects upcoming recurring bills **as of today** and outputs P, T, A, C per bill. Reserving money is a separate,
rule-based layer that uses only these columns (explainable, tunable without retraining, tested separately).

One row per upcoming bill:
| column | meaning |
|---|---|
| detected | this payment group is a recurring bill (period >= 7 d; shorter cycles go to Safety Buffer). Bill = obligations + recurring transfers + regular needs (water, gas, medicine, pet food, lenses, car wash); NOT wallet top-ups or haircuts (user decision 2026-10-13) |
| P | P(active): the bill still exists and will be paid (handles cancellations) |
| T | next date (median) + window (P10-P90) |
| A | amount: median and A_P90 (user choice, was P80) |
| C | calibrated chance the bill is paid inside the T window at or below A_P90 (one number combining P, T, A) |
| bucket | Upcoming / Smart Reserve / Safety Buffer / ask user - set by the rule layer, not the model |

Not C: the "+-3 days around the guess" probability. It measures day precision, not whether money is needed; using it to
scale reserves (amount x confidence) increased shortfall in tests.

## Settled decisions (do not redesign without asking)
- Flow: Predict → Prepare → Protect → Spend.
- Per recurring series predict (P, T, A, C): still active?, next date + window, amount (median + P80), confidence.
- Models: series builder (rules on standard fields) → P(active) (LightGBM) → Bayesian survival for timing → Bayesian shrinkage for amount → router (baseline for long, regular monthly/quarterly series) → isotonic calibration → decision layer.
- Bayesian survival uses a **dynamic period**: continuous period θ (3–400 days, log grid) with a mixed prior = calendar spikes (7/14/30.4/91.3/182.6/365.25) + free smooth part; weights per category learned from train data; σ per series is also estimated. Use a grid, not MCMC.
- Known fixes still required in the timing model:
  1. Add day-of-month drift as evidence (calendar series keep the same day; interval series drift). Without it, a 28-day cycle is pulled to 30.4 with few gaps.
  2. Handle skipped cycles (gap ≈ 2× or 3× period) instead of treating them as noise.
- Decision layer: each obligation goes in exactly one bucket (due before next payday → Upcoming; later → Smart Reserve spread over cycles). Essential + P ≥ 0.8 → reserve A_P80; optional → P × A; uncertain + large → ask user (1-tap). Cycles < 7 days (e.g. wallet top-ups) go to Safety Buffer, not obligations. Own-account transfers are never obligations.
- Future work (do not build now): scheduled_transfer_flag, neural TPP.

## Data
Synthetic only. Never present accuracy from it as evidence; real data comes later.

Transaction fields (standard only, exact names):
txn_id, customer_id, account_id, posted_datetime, amount, direction, txn_type, channel, balance_after, description,
counterparty_account_id, counterparty_bank_code, counterparty_name, promptpay_proxy_type, biller_id, ref1, ref2,
mandate_id, mcc, merchant_id, merchant_name, original_currency, original_amount
(Removed as non-standard: memo, is_own_account, is_payroll.)

Read codes as text: `dtype={"counterparty_bank_code": str, "ref1": str, "ref2": str, "mcc": str}`.

Model inputs: transactions, customers.csv, accounts.csv.
Ground truth only (never features): txn_labels (gt_kind, gt_series_id, gt_flags, gt_link_txn_id), series_truth (incl. cycle_type, true_period_days, true_sigma in v3), users_truth, and any `truth_*` column.

Rules for features: raw IDs and names are grouping keys only, never features. balance_after is for the Safe-to-Spend engine, not the recurrence model.

## Files
Pipeline (run in this order; all default to data_v6 / results_v6):
1. `generate_v5.py --users 3000 --seed 45 --out data_v6` — data. Imports `generate_v4.py` -> `generate_v3.py` -> `generate_v2.py`
   (keep all four in the same folder). v3 adds every-N-days purchases; v4 adds hard cases (counter-service bill payments,
   Shopee PayLater, savings club, landlord account change, rent due-day change, late payers, regular-looking habits that are
   not bills); v5 adds 1-month-to-2-year histories, salary 15k-100k with bills scaling with income, 8 spending profiles.
2. `check_v3.py data_v6` — schema / leakage / balance / censoring checks.
3. `pacm_v3.py` — model only: transactions -> detected, P, A (median + P90), T, C per bill; validation report
   (`pacm_report.md`) and one row per bill with a truth-table verdict (`pacm_output_payday.csv`). Uses `timing_v3.py`
   (T model, dynamic Bayesian survival) and `backtest_v2.py` (baseline B1). Set OPENBLAS_NUM_THREADS=1 and --workers 16.
4. `hitl_v5.py` — questions, separate from the model: moments 1 (new payment >= 1,000), 3 (variable amount), 4 (overdue),
   5 (unsure detection); answers from the truth table; labels every bill with verdict before / after (+ `hitl_app.html`
   from `hitl_app_template.html`).
5. `safe_v5.py` — Safe-to-Spend until next payday (payday from payroll credits): repeating bills = payments due before payday
   x A_P90; longer-cycle bills = A_P90 / cycle days x days to payday; spendable = balance - upcoming bills;
   safety buffer = 20% x spendable (0 if negative); safe to spend = spendable - buffer (user decision 2026-10-13).
6. `viz_v5.py` + `viz_v5_template.html` — main page `results_v6/safebelt_v6.html` (correctness, P/A/T/C, per-customer view).

- Validation day ("today") = each customer's own payday + 1 day in June 2026 (from payroll credits; no payroll -> 1 Jun):
  `pacm_v3.py --today payday` (default); outputs are named *_payday.csv in results_v6/.
- `data_v6/` — the only dataset kept (older v3/v4/v5 data and results were deleted 2026-10-13).
- `DATA_SPEC_for_generator.md` — spec given to another AI to generate data in the same format (incl. payday validation).
- `data_v7/` — data from another AI (2,500 customers, its own generator + README inside; no snapshot files). Results in
  `results_v7/` (page `safebelt_v7.html`). safe_v5 uses the main account (salary account) balance and reports average
  monthly non-recurring spending (outside detected bills, own transfers excluded, last 3 months).
  New data from it goes in e.g. `data_v7/`, then run steps 2-6 with `--data data_v7 --out results_v7`.

## Evaluation rules
- Split by time: train < 2025-10-01, calib 2025-10..2026-01, test 2026-02..2026-06; plus held-out customers in test.
- Censor train/calib labels at their cutoff (no future outcomes). active_label is empty when unknown at the cutoff.
- Always compare against baseline B1 and break results down by cycle_type, period, n_prior bucket (1, 2–3, 4–6, 7+), held-out vs seen customers.
- Targets from the v2 baseline: seen-once MAE 60 days, annual ±7 days 40%, cancellation precision 10% / recall 50%, monthly ±7 days 98% (must not get worse).
- Product metric (later): simulated shortfall rate and wrongly reserved money.

## Status (2026-10-13): restarted, model only
The user asked to start over with **only the model**: transactions in, P / A / T / C per recurring bill out, using at most
the last 24 months of history, validated against what really happened. No reserve rules, no human-in-the-loop questions,
no Smart Reserve for now. Old exploration scripts and results_v3/ were deleted.

Kept from before:
- Task 1 data: `data_v3` (validated by `check_v3.py`).
- T: `timing_v3.py` (dynamic Bayesian survival, fixes 1+2). Test MAE 3.3 d vs B1 5.0; all three v2 targets met.
  Ablation: fix 2 carries most of the gain; fix 1 mainly helps calendar bills.
- Grouping lessons (now inside `pacm_v3.py`): group by biller+ref1 / merchant / counterparty account; drop own accounts;
  split payments to people by amount peaks (`clusters`), so rent and utilities to one landlord become two series.
- Earlier detection runs scored 99-100% precision/recall: the synthetic data is probably too easy (bills vs habits separate
  too cleanly); treat detection scores with suspicion until the data is made harder.

Findings to remember for a later rule layer (not used now): reserve = amount x (+-3-day confidence) increased shortfall;
a "when will you pay this again?" question for new payments >= 1,000 THB helped (skip one-off merchant categories);
holding the full amount when P(next within 30 d) >= 0.1 and saving gradually for later bills reduced shortfall.

## Next tasks, in order
1. Run `pacm_v3.py` on data_v6, validate each output on test: detection precision/recall, T MAE + P10-P90 coverage vs B1,
   A error + P90 coverage vs median-of-3, P AUC + reliability, C reliability; per-bill verdicts vs the truth table.
   Output format the user wants per bill: "rent (นาย ประกร) | P active 0.99 | T 15 Oct 2026 (window) | A 6,500 (P90) | C 0.98",
   plus whether it is truly recurring and whether the prediction came true.
2. Fix whatever validation shows is wrong (coverage far from 80%, P or C not calibrated), re-validate.
3. Then improve the rules / grouping where the verdicts show errors.

## Working style
- Respond to the user in Thai unless they write in English.
- Be honest and critical; the user explicitly asked not to be agreed with by default.
- Verify outputs by running them (schema, leakage, balance never negative) before reporting.
- Small, runnable scripts; print a short summary table at the end of each run.
