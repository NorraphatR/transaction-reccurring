# SafeBelt: recurring bills and safe to spend (Nine)

SafeBelt reads a person's bank transactions and predicts their recurring bills. From those bills it works out how much they can safely spend until the next payday.

For every bill the model gives four numbers:

| Output | Meaning |
|---|---|
| **P** | Chance the bill is still active |
| **T** | Next payment date, with a P10–P90 window |
| **A** | Amount: median and P90 |
| **C** | Calibrated chance the bill is paid inside the T window, at or below A P90 |

The work runs in three separate steps:

1. **Model** (`pacm_v3.py`). Steps 2 and 3 never change it.
2. **Questions** (`hitl_v5.py`). The app asks the customer a few questions, and the answers adjust the bill list.
3. **Safe to spend** (`safe_v5.py`).

Everything is checked against a truth table and what really happened.

All data here is either synthetic (`data_v8`) or anonymised real statements (`real_data`, `real_data_2`). The synthetic numbers show the method works. They are not evidence about real customers.

## Folder

| Path | What it is |
|---|---|
| `generate_v8.py` | Synthetic data generator, v8 (seed 20261014, 2,500 customers) |
| `generate_v2.py` | Shared constants (time split) and older generator; imported by the pipeline |
| `check_v3.py` | Data checks: schema, balances, leakage, payday rules |
| `amount_split.py` | Splits payments to one payee into separate bills (amount + timing Bayesian model selection) |
| `timing_v3.py` | T: Bayesian survival model (grid posterior, day-of-month and skipped-cycle aware) |
| `pacm_v3.py` | The model: detector and P (LightGBM), T, A (Bayesian shrinkage, Student-t), isotonic calibration of P and C |
| `backtest_v2.py` | Baseline B1 (calendar-snapped next date) |
| `hitl_v5.py` | Questions (moments 1, 3, 4, 5), answered from the truth table for synthetic customers |
| `safe_v5.py` | Safe to spend, buffer and shortfall check |
| `viz_v5.py`, `viz_v5_template.html` | Results page for a synthetic dataset |
| `predict_real.py` | Runs the trained model on a real statement, on the day after every payday |
| `real_validate.py` | Scores real paydays, model only and with the owner's answers |
| `viz_real.py`, `real_payday_panel.js` | Results page for the real statements: same layout, one entry per payday, with the questions asked |
| `predict_now.py` | Live forecast for "today" (no outcome yet) |
| `anonymize_real.py` | Replaces names and account numbers in a real statement with codes |
| `tools/` | Converters from KBank statements to this format (owner name is a placeholder) |
| `data_v8/` | Synthetic dataset: transactions, accounts, customers + truth (`txn_labels`, `series_truth`, `users_truth`) |
| `real_data/`, `real_data_2/` | Two real statements (Nov 2025 – Oct 2026), anonymised; `truth_by_user.csv` = the owner's own bill labels |
| `models/v8/` | Trained v8 model: detector, timing prior, bill-split prior, training candidates to refit P / A / C |
| `REQUIRED_FIELDS.md` | Which data fields each step needs |
| `DATA_SPEC_for_generator.md` | Full data format and generator rules |
| `CLAUDE.md` | Working notes and decisions for the project |

## How the pieces fit

1. **Group payments by payee.** The payee key is biller + reference, else merchant, else the counterparty account. Transfers to the customer's own accounts are dropped.
2. **Split mixed payees.** `amount_split.py` decides whether one payee hides several bills, for example rent + utilities to one landlord. It uses both amounts and payment dates. Part payments are joined first.
3. **Score each candidate bill** with the model at each validation day, using at most the last 24 months of history.
4. **Ask questions.** Each question is asked only when it can change a result:
   - **moment 1:** first payment of ฿1,000 or more to someone new, when the model has no prediction for that bill;
   - **moment 3:** amount varies;
   - **moment 4:** bill looks overdue;
   - **moment 5:** the model is unsure, or (5b) a small but very regular payment the model did not call a bill.
5. **Reserve money for each bill until the next payday:**
   `max(A_P90 / floor(cycle days) × (days to payday + 1), A_P90 if due before payday)`
6. **Safe to spend:**
   - spendable = balance − bills reserved;
   - safety buffer = 20% of spendable;
   - safe to spend = spendable − buffer.

Validation day = each customer's payday + 1 day. Real bill payments up to the real next payday are matched to the reserve made for the same bill.

## Run it

Python 3.13 with pandas, numpy and lightgbm.

```bash
# synthetic: train and validate on data_v8 (about 30 min with 16 workers)
python amount_split.py --data data_v8 --out results_v8
python pacm_v3.py      --data data_v8 --out results_v8 --workers 16
python hitl_v5.py      --data data_v8 --res results_v8
python safe_v5.py      --data data_v8 --res results_v8
python viz_v5.py       --data data_v8 --res results_v8      # -> results_v8/safebelt_v8.html

# real statement: model trained on data_v8, run on every payday
python predict_real.py --real real_data   --res models/v8 --data data_v8 --out results_real_v8
python predict_real.py --real real_data_2 --res models/v8 --data data_v8 --out results_real_2_v8
```

`real_validate.py` and `viz_real.py` also need each owner's answers to the app's questions (`real_answers.json` in the results folder). Those answers are not published.

On Windows, set `PYTHONIOENCODING=utf-8`. `OPENBLAS_NUM_THREADS=1` avoids memory errors with many workers.

## Results

### Synthetic data_v8 (2,500 customers, validated on each customer's June 2026 payday)

| | Model only | + questions |
|---|---|---|
| Bill shortfall (avg per customer) | ฿1,026 | ฿588 |
| Customers short > ฿2,000 | 17% | 8% |
| Reserved above need (avg) | ฿2,539 | ฿1,240 |
| Customers still short after the 20% buffer | 2% | 1% |

### Real statements (9 paydays each, model trained on data_v8)

| | Model only | + owner's answers |
|---|---|---|
| Statement 1: bill shortfall, all paydays | ฿13,718 | ฿6,245 |
| Statement 1: real bills found / wrong calls / missed | 22 / 12 / 0 | |
| Statement 2: real bills found / wrong calls / missed | 2 / 7 / 11 | |

v8 was trained with small shared subscriptions paid to friends. After that it catches a ฿41.50 monthly share it used to miss, so recall on statement 1 went from 55% to 100%.

## Known limits

- **Two real people.** This shows how the model behaves on real data. It is not an accuracy figure.
- **Statement 2 has no salary.** The largest allowance transfer each month is treated as payday. Its card purchases carry no merchant name, so card bills cannot be seen.
- **A P90 can explode** when a bill has only 2–4 payments with very different amounts (seen as ฿27,925 and ฿78,998 on real data). A cap is proposed but not applied.
- **Frequent transfers to friends are still often called bills on real data.** The synthetic data has no "friend paid back a large amount now and then" pattern.
