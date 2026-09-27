# nonhomothetic-ces

Study notes and a Python simulation/estimation toolkit for nonhomothetic CES preferences.

**Wenzhuo Wang** · The University of Chicago · [wenzhuo_wang@outlook.com](mailto:wenzhuo_wang@outlook.com)

- **Theory:** [study note](note/study-note-nhces.pdf), covering the implicit aggregator, demand, expenditure elasticities, and a deterministic Euler equation.
- **Computation:** [model](estimation/model.py), [simulation](estimation/simulate.py), [seven estimators](estimation/estimators.py), and [Monte Carlo runner](estimation/run_montecarlo.py).

The estimation experiment is a Python adaptation of **Martí Mestieri's Stata sample code**, linked from [his research page](https://mestieri.github.io/). The original Stata files are not redistributed. This is an educational implementation, not an official author release, a full replication of the empirical paper, or a HANK solver. The original author has not endorsed this project.

## Model and notation

The code follows the study note, without preference weights:

$$
\sum_i\left(\frac{c_i}{C^{\varepsilon_i}}\right)^{(\sigma-1)/\sigma}=1,
\qquad
E(p,C)=\left[\sum_i p_i^{1-\sigma}C^{(1-\sigma)\varepsilon_i}\right]^{1/(1-\sigma)}.
$$

All prices, quantities and nonhomotheticity parameters are strictly positive; $\sigma>0$, $\sigma\ne1$. The static model is implemented; the note's Euler equation is not part of this code.

The Stata expenditure equation uses an exponent $\delta_i$ directly on $C$. To keep the **same consumption index**, use $\varepsilon_i=\delta_i/(1-\sigma)$. Thus its baseline $\sigma=0.5$, $\delta=(0.1,1,1.25)$ becomes **`epsilon=(0.2, 2.0, 2.5)`** here. These are preference parameters, not expenditure elasticities.

Ordinary estimators identify parameter ratios, not the scale of $C$. `epsilon_m` is a supplied normalization. The Monte Carlo runner fixes it to the data-generating specification for reporting parameter levels; it is never counted as an estimated parameter. Method 7 additionally observes the true consumption index and requires the matching normalization.

## Installation and tests

Open the repository root in your terminal or VS Code. The code targets Python 3.11–3.13.

```bash
conda env create -f environment.yml
conda activate nhces
python -m pytest -v
```

For an existing Python environment, use `python -m pip install -r requirements.txt`. The dependency ranges are compatibility constraints, not an exact environment lock. Each experiment records the versions actually used.

Tests check parameter recovery for methods 1, 2 and 7 on exact samples, economic identities, normalization, simulation behavior, failure reporting, and Monte Carlo summaries. Approximate methods are not required to recover exact parameters. Passing tests does not prove empirical validity or consistency under measurement error.

## Run an experiment

```bash
# Five replications, all seven methods
python -m estimation.run_montecarlo --simulations 5

# Default: 100 replications, 10 countries, 50 periods
python -m estimation.run_montecarlo

# No measurement error: exact-model recovery experiment
python -m estimation.run_montecarlo --simulations 5 --methods 1 2 7 --measurement-sd 0
```

Each command creates a fresh directory such as `results/run-20260927-170000-123456`. To choose a directory explicitly, add `--output results/my_experiment`. Existing result files are never overwritten. `--help` lists the remaining controls.

For $\sigma>1$, methods 1 and 2 require bounds above one, for example `--sigma 1.4 --epsilon 0.4 0.9 1.8 --sigma-bounds 1.0001 3`. The default nonlinear search interval is $0<\sigma<1$.

In VS Code, select the intended Python interpreter, open **Run and Debug**, and choose **Monte Carlo - 5 simulations**, **Monte Carlo - 100 simulations**, or **Tests - all**. Press **F5**, or **Ctrl+F5** to run without debugging. Shared configurations are in `.vscode/`. Do not use “Run Python File” on the individual modules; they are designed to be imported as a package.

## Estimators and output

| Method | Procedure | Reported quantities |
|---|---|---|
| 1 | Log-linear two-step GMM | $\sigma$ and preference ratios |
| 2 | Nonlinear-price-index two-step GMM | $\sigma$ and preference ratios |
| 3 | Iterative constrained SUR | Approximate $\sigma$ and preference ratios |
| 4 | OLS on expenditure and prices | Proxy expenditure coefficients; no $\sigma$ estimate |
| 5 | Constrained SUR using a Tornqvist index | $\sigma$ and proxy consumption coefficients |
| 6 | SUR followed by GMM for scale | $\sigma$ and preference ratios; inherits first-stage approximation |
| 7 | Constrained SUR with known $C$ | $\sigma$, preference differences, and known-$C$ slopes |

The reported `relative_engel_slope` is $(\varepsilon_s-\varepsilon_m)/(\varepsilon_a-\varepsilon_m)$. For methods 4 and 5, the ratio of proxy slopes is evaluated as an approximation to that target. It is not reported in the summary when the target denominator is zero or numerically negligible.

| File | Contents |
|---|---|
| `estimates.csv` | One row per replication and method, including failures and diagnostic messages |
| `summary.csv` | Means, bias, RMSE, standard deviations, and Monte Carlo standard errors of means |
| `status.csv` | Success rates, failure counts, and estimator timings |
| `config.json` | Parameters, seed schedule, normalization, package versions, and source-file SHA-256 hashes |

Bias and RMSE condition on **successful, finite** estimates. `used` is the number used for each metric; always inspect `status.csv` as well. Proxy coefficients in methods 4 and 5 have no structural truth assigned, so their truth/bias/RMSE cells are blank. Fixed normalization parameters are not scored.

Measurements follow the Stata design: independent lognormal errors affect prices, expenditure, and shares; shares are capped at one separately and are **not renormalized**. Measured shares need not add to one. The growth shocks are observation-specific, not cumulative annual innovations. A matching seed reproduces NumPy draws in the same environment, not the original Stata draws.

GMM uses the source's instruments, an identity first-stage weight, and uncentered robust second-stage weights. SUR imposes a common relative-price coefficient. Unlike the source, this version checks convergence/rank, bounds nonlinear parameters, regularizes nearly singular covariance matrices, and bypasses covariance reweighting for exact fits. These changes are reported in result messages. The source's instruments are not automatically valid under measurement error. Standard errors, hypothesis tests, and a direct Stata-output comparison are not implemented.

## Included reference runs

`results/check/` and `results/monte_carlo/` preserve the supplied 5- and 100-replication runs. They are historical snapshots, not newly generated results of every subsequent code change. See [results/README.md](results/README.md) for provenance and interpretation. New timestamped runs are ignored by Git; the two reference runs and shared VS Code settings are tracked.

## References and attribution

The main reference is Comin, Lashkari and Mestieri (2021), *Structural Change With Long-Run Income and Price Effects*, **Econometrica 89(1), 311–374**, [doi:10.3982/ECTA16317](https://doi.org/10.3982/ECTA16317).

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

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source files and adaptation details. The code and notes use mixed licensing as specified in [LICENSE](LICENSE): original model/tests under MIT, the three Stata adaptations under CC BY-SA 4.0, and the study note under CC BY 4.0. This is not a blanket MIT license for the entire repository.
