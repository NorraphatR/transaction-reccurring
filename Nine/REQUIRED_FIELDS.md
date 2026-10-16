# SafeBelt: data fields the prediction pipeline needs

## Context

SafeBelt predicts each customer's upcoming recurring bills from bank transactions, then computes how much is safe to spend until the next payday. For every bill it outputs:

- **P**: chance the bill is still active.
- **T**: next payment date, with a P10–P90 window.
- **A**: amount, as a median and P90.
- **C**: calibrated chance the bill is paid inside the window at or below A P90.

The pipeline has 3 steps:

1. **Model** (`pacm_v3.py`): LightGBM detector and P(active), Bayesian survival model for T, Bayesian amount model for A, and isotonic calibration for P and C.
2. **Questions** (`hitl_v5.py`, optional): asks the customer a few questions in the app.
3. **Safe to spend** (`safe_v5.py`): rule step. Safe to spend = (balance − upcoming bills until next payday) − 20% safety buffer.

The data is grouped into 3 tiers:

- **Tier 1**: needed by the model.
- **Tier 2**: needed by the other steps.
- **Tier 3**: needed only for training labels and validation; never used as a feature.

Rules:

- Raw IDs and names are only used to group payments. They are never model features.
- Truth columns are never features.

## Tier 1: needed by the model

### `transactions.csv.gz`: one row per transaction, all accounts of all customers

| Field | Type / format | Why the model needs it |
|---|---|---|
| `txn_id` | string, unique | joins to the labels (Tier 3) |
| `customer_id` | string, e.g. `C00001` | whose transaction |
| `account_id` | string | own-account transfer detection; main account for the balance |
| `posted_datetime` | `YYYY-MM-DD HH:MM:SS` | payment dates, which drive T and the cycle features |
| `amount` | number > 0, 2 decimals | always positive; the sign is in `direction`; drives A and the bill split |
| `direction` | `DR` (money out) / `CR` (money in) | only `DR` payments are scored as bills |
| `txn_type` | enum, see below | feature, and the group used for the timing and amount priors |
| `channel` | `KPLUS`, `CARD`, `AUTO`, `ATM`, `BRANCH` | feature: share of `AUTO` payments |
| `mandate_id` | string or empty | feature: has a direct-debit mandate |
| `biller_id` | string or empty | payee key for bill payments / direct debits (with `ref1`) |
| `ref1` | **string** or empty, keep leading zeros | contract number at the biller; must be **stable per contract** |
| `merchant_name` | string or empty | payee key for card / QR payments, e.g. `NETFLIX.COM` |
| `counterparty_account_id` | string or empty | payee key for transfers / PromptPay to people |
| `counterparty_name` | string or empty, masked, e.g. `นาย สมชาย ก****` | display name; own-name transfer detection |
| `description` | free text | display name only |

`txn_type` values:

- **Money out (DR):** `bill_payment`, `direct_debit`, `card_purchase`, `qr_payment`, `promptpay_out`, `transfer_out`, `atm_withdrawal`
- **Money in (CR):** `payroll_in`, `transfer_in`, `promptpay_in`, `refund_in`

How payments are grouped into one payee:

1. `biller_id + ref1`, if a biller is present;
2. otherwise `merchant_name`;
3. otherwise `counterparty_account_id`;
4. otherwise `txn_type`.

Payments to people (`counterparty_account_id`) are then split into separate bills (e.g. rent + utilities to one landlord) by `amount_split.py`. It uses only `amount`, `posted_datetime` and `txn_type`.

### `accounts.csv`

| Field | Why |
|---|---|
| `customer_id`, `account_id` | a transfer to one of the customer's own accounts is dropped; it isn't a bill |
| `account_type`, `open_date` | not used by the model |

### `customers.csv`

| Field | Why |
|---|---|
| `customer_id` | key |
| `customer_name` | a transfer to someone with the customer's own name is dropped as an own transfer |
| `first_account_open_date` | Tier 2, used by the questions step |

## Tier 2: needed by the other steps

| Step | Fields | Used for |
|---|---|---|
| Today / payday | `txn_type = payroll_in` (CR) + `posted_datetime` | validation day = the day after the customer's payday; next payday comes from the payroll history (no salary → +30 days) |
| Questions | `mcc` (4-digit **string**) | no questions about travel, medical or restaurant merchants |
| Questions | `customers.first_account_open_date` | history length per customer |
| Safe to spend | `account_id`, `balance_after` (2 decimals; equals the previous `balance_after` ± `amount` per account, never negative) | balance today on the main account (the payroll account, else the busiest account) |
| Safe to spend | `payroll_in` credits | salary received and next payday |

Fields in the full spec that **no step uses**: `counterparty_bank_code`, `promptpay_proxy_type`, `ref2`, `merchant_id`, `original_currency`, `original_amount`. They can be empty.

## Tier 3: training labels and validation only (never features)

| File | Fields | Used for |
|---|---|---|
| `txn_labels.csv.gz` | `txn_id`, `gt_series_id` (true recurring series), `gt_flags` (`duplicate_charge`, `split_part1`, `split_part2`, …) | training labels (is this group a bill?); scoring P / A / T / C against what happened |
| `series_truth.csv` | `series_id`, `customer_id`, `name`, `kind`, `category`, `cycle_type`, `true_period_days`, `status` | which series are bills, and the timing prior fit |
| `users_truth.csv` | `customer_id`, `user_split` (`train` / `heldout`) | train / held-out customer split |

A **bill** in the truth table is a series that meets all three conditions:

- `kind` is `obligation` or `recurring_transfer`;
- `true_period_days` is ≥ 7;
- `category` is not `wallet` or `personal_care`.

On real bank data, Tier 3 doesn't exist. Instead, the model would be retrained from labelled history, and validation would come from what customers actually paid.

## Minimum to run a prediction on new data

These, plus a trained model from a previous run:

- **transactions:** `customer_id`, `account_id`, `posted_datetime`, `amount`, `direction`, `txn_type`, `channel`, `mandate_id`, `biller_id`, `ref1`, `merchant_name`, `counterparty_account_id`, `counterparty_name`, `description`, `balance_after`, `mcc`
- **accounts:** `customer_id`, `account_id`
- **customers:** `customer_id`, `customer_name`, `first_account_open_date`

Data requirements:

- **History:** up to 24 months before "today". A bill needs at least 2 past payments to the same payee before it can be predicted.
- **Payroll:** at least 2 `payroll_in` credits before today, so the next payday can be estimated.
- **Time frame for the current pipeline:**
  - data runs from 2024-01 to 2026-09-30;
  - train cut 2025-10-01, calibration to 2026-01-31, test from 2026-02-01;
  - validation day = each customer's June 2026 payday + 1 day.

The full generator spec, with all 23 transaction columns, realism rules and self-checks, is in `DATA_SPEC_for_generator.md`.
