# Fisher Scoring & MCMC for GLMs — from scratch

Frequentist vs. Bayesian estimation of **logistic** and **Poisson regression**, implemented from first principles in NumPy: no `statsmodels.fit()`, no `sklearn` for the estimation itself. Library implementations are only used afterwards to validate the results.

> Joint project with [Leonard Fischbach](https://github.com/LeoFischbach), Bocconi University.

---

## What's inside

| Part | Method | Implemented by hand |
|---|---|---|
| **1. Frequentist** | Fisher Scoring (≡ IRLS for canonical links) | Score vector, expected Fisher information, Newton-type update, convergence check, Wald standard errors & p-values, McFadden pseudo-R² |
| **2. Bayesian** | Random-walk Metropolis–Hastings | Log-posteriors with Gaussian prior, adaptive step size during burn-in, posterior summaries & 95% credible intervals, trace plots, ACF |

One generic `fisher_glm()` handles both models by switching the mean function and weights:

| Model | Mean μ | Weight W |
|---|---|---|
| Logistic | σ(Xβ) | μ(1 − μ) |
| Poisson | exp(Xβ) | μ |

Update step: β ← β + (XᵀWX)⁻¹ Xᵀ(y − μ), iterated until ‖β⁽ᵗ⁺¹⁾ − β⁽ᵗ⁾‖ < 10⁻⁶.

## Data

| Dataset | Model | Target | Features | n |
|---|---|---|---|---|
| `CreditCard.csv` (AER, Greene 2003) | Logistic | Credit card application accepted (yes/no) | reports, age, income, dependents, months, active, majorcards, owner, selfemp | 1,319 |
| `Hospital.csv` (azcabgptca, Hilbe 2014) | Poisson | Length of hospital stay (days) | age, procedure (CABG vs. PTCA), gender, type (emergency vs. elective) | 1,959 |

Modelling choices: continuous features are standardized, binary ones kept as 0/1 dummies. `share` and `expenditure` are dropped from the credit data (collinear with income), and `died` from the hospital data (endogeneity).

## Results

**Fisher Scoring matches the library implementations to ~3 decimals:**

| Hospital (Poisson) | own | `statsmodels` |
|---|---|---|
| procedure (CABG) | 1.122 | 1.122 |
| type (emergency) | 0.190 | 0.190 |
| age | 0.041 | 0.041 |
| gender (male) | −0.103 | −0.103 |

| CreditCard (Logistic) | own | `sklearn` (unpenalized) |
|---|---|---|
| reports | −2.356 | −2.354 |
| active | 0.834 | 0.833 |
| selfemp | −0.757 | −0.754 |
| income | 0.383 | 0.384 |

**MCMC posterior means agree with the MLEs** (weakly informative N(0, 5²) prior, 60k iterations, 12k burn-in):

| | Acceptance rate | McFadden R² (MLE) | McFadden R² (posterior mean) |
|---|---|---|---|
| CreditCard (Logistic) | 0.46 | 0.302 | 0.302 |
| Hospital (Poisson) | 0.44 | 0.280 | 0.280 |

**Interpretation highlights**
- Each additional derogatory report sharply lowers the odds of a credit card approval, the strongest effect in the model; higher income and more active accounts raise them.
- A CABG procedure is associated with roughly 3× longer hospital stays than PTCA (e^1.12 ≈ 3.1), and emergency admissions with ~21% longer stays.

The full analysis is in [`report/Report_Analysis.pdf`](report/Report_Analysis.pdf).

## Repository structure

```
├── data/
│   ├── CreditCard.csv
│   ├── Hospital.csv
│   └── Datasets explanation.txt
├── notebooks/
│   └── Code_Fisher_MCMC.ipynb   # Part 1: Fisher Scoring · Part 2: MCMC
├── report/
│   └── Report_Analysis.pdf
└── requirements.txt
```

## How to run

```bash
git clone https://github.com/ErdemB99/MCMC-Fisher_Scoring-Regression.git
cd MCMC-Fisher_Scoring-Regression
pip install -r requirements.txt
jupyter notebook notebooks/Code_Fisher_MCMC.ipynb
```

The notebook loads the CSV files from its working directory, so copy the files from `data/` next to the notebook (or adjust the paths in the "Load Data" cell).

## Tech

Python · NumPy · pandas · SciPy · Matplotlib · statsmodels & scikit-learn (validation only)
