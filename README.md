# nonhomothetic-ces

Study notes and Python simulation and estimation code for nonhomothetic CES preferences.

**Wenzhuo Wang** · The University of Chicago · [wenzhuo_wang@outlook.com](mailto:wenzhuo_wang@outlook.com)

- **Theory:** [study note](note/study-note-nhces.pdf) — the aggregator, demand, expenditure elasticities, and a deterministic Euler equation.
- **Code:** [model](estimation/model.py), [simulation](estimation/simulate.py), [estimators](estimation/estimators.py), and [Monte Carlo runner](estimation/run_montecarlo.py).

The estimation experiment adapts Martí Mestieri's Stata sample code. The Python implementation covers the static model and estimation; the note's Euler equation is not implemented.

## Quick start

Run from the repository root with Python 3.11–3.13:

```bash
conda env create -f environment.yml
conda activate nhces
python -m pytest -v

# Quick check: five replications, all seven methods
python -m estimation.run_montecarlo --simulations 5 --output results/run_01

# Larger experiment
python -m estimation.run_montecarlo --simulations 100 --output results/run_02
```

Use a new output folder each time: existing result files are never overwritten. Without `--output`, the program uses `results/monte_carlo`, which already contains the reference run.

For VS Code, select the `nhces` interpreter, choose a Monte Carlo configuration under **Run and Debug**, and press **F5**. Enter a new result-folder name when prompted. Use the terminal command above to run tests.

## Estimators

| Method | Procedure | Estimated quantities |
|---|---|---|
| 1 | Log-linear GMM | Sigma and epsilon ratios |
| 2 | Nonlinear-price-index GMM | Sigma and epsilon ratios |
| 3 | Iterative SUR | Approximate sigma and epsilon ratios |
| 4 | Expenditure OLS | Proxy coefficients |
| 5 | Tornqvist SUR | Sigma and proxy coefficients |
| 6 | Hybrid SUR/GMM | Sigma and epsilon ratios |
| 7 | Known-consumption SUR | Sigma and epsilon differences |

`epsilon_m` is a supplied normalization, not an estimated parameter; it allows epsilon levels to be reported. Method 7 additionally uses the true consumption index as a simulation benchmark. Proxy coefficients are not epsilon levels.

The [tests](tests/test_recovery.py) check exact-sample recovery for methods 1, 2 and 7, normalization, and selected failure cases. The code provides point estimates; standard errors and a direct comparison with Stata output are not implemented.

## Example results

The supplied [100-replication run](results/monte_carlo/) contains:

- `config.json`: parameters, random seeds, and package versions.
- `estimates.csv`: estimates and diagnostics for each method and replication.
- `summary.csv`: means, bias, and RMSE, conditional on successful finite estimates.
- `status.csv`: success/failure counts and timings; read alongside the summary.

Blank truth/bias/RMSE entries for proxy coefficients mean that the comparison is unavailable. These files are a saved example, not regenerated automatically when the code changes.

## Reference

Comin, Lashkari and Mestieri (2021), *Structural Change With Long-Run Income and Price Effects*, **Econometrica 89(1), 311–374**. [Paper](https://doi.org/10.3982/ECTA16317) · [Author's code and derivations](https://mestieri.github.io/)

```bibtex
@article{comin2021structural,
  author  = {Comin, Diego and Lashkari, Danial and Mestieri, Mart{\'i}},
  title   = {Structural Change With Long-Run Income and Price Effects},
  journal = {Econometrica},
  year    = {2021},
  volume  = {89},
  number  = {1},
  pages   = {311--374},
  doi     = {10.3982/ECTA16317}
}
```

## Attribution and license

See [source and adaptation details](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE). Original model/tests use MIT, the three Stata adaptations use CC BY-SA 4.0, and the study note uses CC BY 4.0. Original Stata files are not redistributed; attribution does not imply endorsement.
