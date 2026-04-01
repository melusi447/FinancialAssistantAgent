# Financial Calculations Reference

## Compound Interest

### Future Value (lump sum)
FV = PV × (1 + r)^n

Where:
- FV = future value
- PV = present value (principal)
- r = annual interest rate (as decimal, e.g. 0.07 for 7%)
- n = number of years

Example: R10,000 at 7% for 20 years = 10,000 × (1.07)^20 = R38,697

### Future Value (with regular contributions)
FV = PV × (1 + r)^n + PMT × [((1 + r)^n − 1) / r]

Where PMT = regular periodic contribution

### Monthly Compounding
FV = PV × (1 + r/12)^(n×12)

Divide the annual rate by 12 for monthly compounding. This slightly increases the effective annual return.

## Loan / Mortgage Calculations

### Monthly Payment
M = P × [r(1+r)^n] / [(1+r)^n − 1]

Where:
- M = monthly payment
- P = loan principal
- r = monthly interest rate (annual rate ÷ 12)
- n = total number of payments (years × 12)

### Total Interest Paid
Total Interest = (M × n) − P

### Remaining Balance After k Payments
Balance = P × (1+r)^k − M × [(1+r)^k − 1] / r

## Retirement Calculations

### Required Retirement Portfolio (4% Rule)
Portfolio needed = Annual desired income × 25

### Required Retirement Portfolio (3% Rule — more conservative)
Portfolio needed = Annual desired income × 33.3

### Monthly Contribution Needed
PMT = FV × r / [(1 + r)^n − 1]

Where FV is your retirement target and r is monthly rate.

## Inflation Calculations

### Future Cost of Something
Future cost = Current cost × (1 + inflation rate)^years

Example: Something costing R1,000 today at 6% inflation for 10 years = R1,791

### Real Return (inflation-adjusted)
Real return ≈ Nominal return − Inflation rate

More precisely: Real return = (1 + nominal) / (1 + inflation) − 1

## Investment Return Metrics

### Compound Annual Growth Rate (CAGR)
CAGR = (Ending Value / Beginning Value)^(1/n) − 1

Where n = number of years

### Return on Investment (ROI)
ROI = (Gain from Investment − Cost of Investment) / Cost of Investment × 100

### Sharpe Ratio
Sharpe = (Portfolio Return − Risk-Free Rate) / Portfolio Standard Deviation

Interpretation:
- Below 0: worse than risk-free asset
- 0–1: acceptable
- 1–2: good
- Above 2: excellent

## Debt Metrics

### Debt-to-Income Ratio (DTI)
DTI = Monthly debt payments / Gross monthly income × 100

Benchmarks:
- Below 20%: excellent
- 20–35%: manageable
- 35–50%: concerning — focus on debt reduction
- Above 50%: critical — seek financial counselling

### Debt Payoff Time (no extra payments)
n = −ln(1 − (P × r / M)) / ln(1 + r)

Where n = months, P = balance, r = monthly rate, M = monthly payment

## Savings Rate

Savings Rate = (Income − Expenses) / Income × 100

A savings rate of 20% is the minimum target. Higher savings rates dramatically shorten the time to financial independence:

| Savings Rate | Years to Financial Independence (from zero) |
|---|---|
| 10% | ~40 years |
| 20% | ~37 years |
| 30% | ~28 years |
| 50% | ~17 years |
| 70% | ~8.5 years |

(Assumes 5% real investment return and 4% withdrawal rate)

## Emergency Fund Target
Months of expenses needed × Monthly essential expenses

Essential expenses = Housing + Food + Utilities + Insurance + Minimum debt payments + Transport to work
