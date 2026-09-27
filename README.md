# nonhomothetic-ces

Study note and Python estimation code for **non-homothetic CES (NH-CES) preferences**.

- Theory: Hanoch (1975, *Econometrica*); Comin, Lashkari and Mestieri (2021, *Econometrica*, "Structural Change with Long-run Income and Price Effects").
- Code: an independent Python re-implementation of Martí Mestieri's Stata Monte Carlo suite for estimating NH-CES preferences ([original code and step-by-step derivations](https://mestieri.github.io/), CC BY-SA 4.0). The original Stata files are **not** redistributed here.
- Note: `note/study-note-nhces.pdf` — a self-contained study note on the model.

## Repository layout

```
estimation/
    model.py            # NH-CES core: implicit aggregator, expenditure function,
                        # Marshallian/Hicksian demand, shares, elasticities
    simulate.py         # synthetic three-sector country panel (from simulate_...web.do)
    estimators.py       # seven estimators (GMM / SUR / OLS variants)
    run_montecarlo.py   # repeated simulations, bias/RMSE summary, CLI entry
tests/
    test_recovery.py    # known-answer checks: generated data must recover true parameters
note/
    study-note-nhces.pdf
results/check/          # a small example output (config, estimates, summary, status)
```

## Quick start

```bash
conda env create -f environment.yml
conda activate nhces

# 1. known-answer tests (must pass before trusting any change)
python -m pytest tests/test_recovery.py -v

# 2. run the Monte Carlo experiment (100 replications by default)
python -m estimation.run_montecarlo --simulations 100 --output results/monte_carlo
```

## Reading the output

- `estimates.csv` — every method × replication point estimate.
- `summary.csv` — bias/RMSE **conditional on successful finite estimates**.
- `status.csv` — success/failure counts per method; always read this together
  with `summary.csv`.
- Methods 1–3, 6–7 report structural parameters (or identifiable ratios /
  differences); methods 4–5 report proxy regression coefficients only, which
  are **not** preference parameters. See `estimation/estimators.py` docstring.

## License

Mixed licensing — see `LICENSE`. Original Python code by Wenzhuo Wang (MIT);
files adapted from Mestieri's Stata code are CC BY-SA 4.0 and marked with an
SPDX header.
