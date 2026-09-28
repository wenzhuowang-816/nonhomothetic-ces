# SPDX-License-Identifier: CC-BY-SA-4.0
"""Run NH-CES Monte Carlo experiments and summarize estimator performance.

Adapted from Marti Mestieri's montecarlo_web.do:
https://mestieri.github.io/
Original license: https://creativecommons.org/licenses/by-sa/4.0/
Python adaptation: Wenzhuo Wang, The University of Chicago.
Adds configurable runs, CSV/JSON output, and bias/RMSE/failure summaries.

Methods share each simulated sample; only method 7 observes true consumption.
epsilon_m is a fixed normalization, not an estimated parameter.

Bias/RMSE use successful finite estimates; also check status.csv.
Methods 4–5 report proxy slopes with no individual structural truth.
Their ratio approximates (epsilon_s-epsilon_m)/(epsilon_a-epsilon_m),
which is undefined when epsilon_a=epsilon_m.

Run from the repository root: python -m estimation.run_montecarlo
"""

import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import scipy

from .model import NonhomotheticCES
from .simulate import simulate_panel
from .estimators import METHODS, EstimateResult, estimate


def _ratio(numerator, denominator):
    if abs(denominator) <= 1e-8 * max(1.0, abs(numerator)):
        return np.nan
    return float(numerator / denominator)


def _record(result, replication, seed, simulation_seconds, failure_stage):
    row = {
        "replication": replication,
        "seed": seed,
        "method": result.method,
        "method_name": METHODS[result.method],
        "success": result.success,
        "failure_stage": failure_stage,
        "seconds": result.seconds,
        "simulation_seconds": simulation_seconds,
        "iterations": result.iterations,
        "evaluations": result.evaluations,
        "objective": result.objective,
        "message": result.message,
        "sigma": result.sigma,
        "relative_engel_slope": np.nan,
    }
    for name, values in (("epsilon", result.epsilon),
                         ("epsilon_ratio", result.epsilon_ratios),
                         ("epsilon_difference", result.epsilon_differences)):
        for position, sector in ((0, "a"), (2, "s")):
            row[f"{name}_{sector}"] = np.nan if values is None else values[position]
    for i, sector in enumerate(("a", "s")):
        row[f"coefficient_{sector}"] = np.nan if result.coefficients is None else result.coefficients[i]
    if result.epsilon_ratios is not None:
        a, _, s = result.epsilon_ratios
        row["relative_engel_slope"] = _ratio(s-1, a-1)
    elif result.epsilon_differences is not None:
        a, _, s = result.epsilon_differences
        row["relative_engel_slope"] = _ratio(s, a)
    elif result.coefficients is not None:
        a, s = result.coefficients
        row["relative_engel_slope"] = _ratio(s, a)
    return row


def summarize(draws: pd.DataFrame, model: NonhomotheticCES):
    """Return (summary, status); unsupported parameters are not scored."""
    ea, em, es = model.epsilon
    contrast = _ratio(es-em, ea-em)
    targets = {"sigma": model.sigma, "epsilon_a": ea, "epsilon_s": es,
               "relative_engel_slope": contrast}
    summary, status = [], []
    for method, group in draws.groupby("method", sort=True):
        successful = group[group.success]
        attempted = group[group.failure_stage != "simulation"]
        status.append({
            "method": method, "method_name": METHODS[method], "runs": len(group),
            "successful": len(successful), "failed": len(group)-len(successful),
            "simulation_failures": int((group.failure_stage == "simulation").sum()),
            "estimation_failures": int((group.failure_stage == "estimation").sum()),
            "success_rate": len(successful)/len(group),
            "mean_seconds": attempted.seconds.mean(),
            "mean_success_seconds": successful.seconds.mean(),
        })
        metrics = ["sigma", "epsilon_a", "epsilon_s"]
        if method == 4:
            metrics = ["coefficient_a", "coefficient_s"]
        elif method == 5:
            metrics = ["sigma", "coefficient_a", "coefficient_s"]
        elif method == 7:
            metrics += ["coefficient_a", "coefficient_s"]
        if np.isfinite(contrast):
            metrics.append("relative_engel_slope")
        for metric in metrics:
            target = targets.get(metric, np.nan)
            if method == 7 and metric.startswith("coefficient_"):
                target = (1-model.sigma) * ((ea if metric.endswith("_a") else es)-em)
            values = pd.to_numeric(successful[metric], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            error = values-target
            n = len(values)
            sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
            summary.append({
                "method": method, "method_name": METHODS[method], "metric": metric,
                "runs": len(group), "successful": len(successful), "used": n,
                "truth": target,
                "mean": float(np.mean(values)) if n else np.nan,
                "bias": float(np.mean(error)) if n and np.isfinite(target) else np.nan,
                "rmse": float(np.sqrt(np.mean(error**2))) if n and np.isfinite(target) else np.nan,
                "sd": sd, "mcse_mean": sd/np.sqrt(n) if n > 1 else np.nan,
                "interpretation": "proxy coefficient" if metric.startswith("coefficient_") and method in (4, 5)
                                  else "approximation to structural ratio" if metric == "relative_engel_slope" and method in (4, 5)
                                  else "structural target",
            })
    return pd.DataFrame(summary), pd.DataFrame(status)


def run_monte_carlo(
    model: NonhomotheticCES,
    *,
    simulations: int = 100,
    seed: int = 1,
    methods: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    simulation_options: dict | None = None,
    estimation_options: dict | None = None,
    output: str | Path | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (draws, summary, status), optionally saving CSVs and config.

    Replication r uses seed+r-1; NumPy draws differ from Stata's.
    Options pass to simulate_panel/estimate, excluding runner-controlled
    arguments. Unknown DGP parameters are not used to initialize estimates.

    Simulation and estimation failures are recorded without replacement.
    Failed estimates stay in draws but are excluded from summary.
    Invalid API/configuration options stop execution.

    With output set, estimates are saved after each replication.
    Existing files are never overwritten; interrupted runs retain partial
    output but cannot resume automatically.
    """
    for name, value, minimum in (("simulations", simulations, 1), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be a nonempty sequence without duplicates.")
    if any(isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)) or m not in METHODS for m in methods):
        raise ValueError("Each method must be an integer from 1 through 7.")
    if len(model.epsilon) != 3:
        raise ValueError("The experiment requires three goods in a, m, s order.")
    sim = dict(simulation_options or {})
    est = dict(estimation_options or {})
    if {"model", "seed"} & sim.keys() or {"data", "method", "truth", "epsilon_m"} & est.keys():
        raise ValueError("Do not override model, seed, data, method, truth, or epsilon_m in options.")
    sim_bound = inspect.signature(simulate_panel).bind(model, seed=int(seed), **sim)
    sim_bound.apply_defaults()
    est_bound = inspect.signature(estimate).bind(pd.DataFrame(), method=methods[0], epsilon_m=model.epsilon[1], **est)
    est_bound.apply_defaults()
    # Validate estimator controls; the empty panel itself returns a failed result.
    estimate(pd.DataFrame(), method=methods[0], epsilon_m=model.epsilon[1], **est)
    lower, upper = est_bound.arguments["sigma_bounds"]
    if any(m in (1, 2) for m in methods) and not lower < model.sigma < upper:
        raise ValueError("The DGP sigma must be inside sigma_bounds for methods 1/2; adjust the bounds.")
    # Check deterministic simulation options without changing the actual draws.
    validation = dict(sim)
    validation.update(countries=sim.get("countries", 10), periods=sim.get("periods", 50))
    for key in ("countries", "periods"):
        value = validation[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    validation.update(countries=1, periods=1, growth_sd=0, measurement_sd=0)
    for key in ("growth_sd", "measurement_sd"):
        value = sim_bound.arguments[key]
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    simulate_panel(model, seed=0, **validation)

    config = {
        "model": asdict(model), "simulations": int(simulations), "seed": int(seed),
        "seed_rule": "seed + replication - 1", "methods": [int(m) for m in methods],
        "simulation_options": {k: v for k, v in sim_bound.arguments.items() if k not in ("model", "seed")},
        "estimation_options": {k: v for k, v in est_bound.arguments.items() if k not in ("data", "method", "truth")},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "notes": "epsilon_m is fixed, not estimated. Bias/RMSE condition on successful finite estimates. See status.csv.",
    }
    directory = Path(output) if output is not None else None
    if directory is not None:
        names = ("estimates.csv", "summary.csv", "status.csv", "config.json")
        if any((directory / name).exists() for name in names):
            raise FileExistsError("Output files already exist; choose a different output directory.")
        config_text = json.dumps(config, indent=2, allow_nan=False,
                                 default=lambda value: value.tolist())
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(config_text + "\n", encoding="utf-8")

    records = []
    for replication in range(1, simulations + 1):
        draw_seed = int(seed + replication - 1)
        started = perf_counter()
        simulation_error = None
        try:
            observed, truth = simulate_panel(model, seed=draw_seed, **sim)
        except (ValueError, FloatingPointError) as error:
            simulation_error = str(error)
        simulation_seconds = perf_counter()-started
        batch = []
        for method in methods:
            if simulation_error is not None:
                result = EstimateResult(method=method, message=simulation_error)
                stage = "simulation"
            else:
                result = estimate(observed, method, truth=truth if method == 7 else None,
                                  epsilon_m=model.epsilon[1], **est)
                stage = "" if result.success else "estimation"
            batch.append(_record(result, replication, draw_seed, simulation_seconds, stage))
        records.extend(batch)
        if directory is not None:
            path = directory / "estimates.csv"
            pd.DataFrame(batch).to_csv(path, mode="a", header=not path.exists(), index=False)
        if progress and (replication == 1 or replication % 10 == 0 or replication == simulations):
            failed = sum(not row["success"] for row in records)
            print(f"Completed {replication}/{simulations}; failed method-runs: {failed}/{len(records)}", flush=True)
    draws = pd.DataFrame(records)
    summary, status = summarize(draws, model)
    if directory is not None:
        summary.to_csv(directory / "summary.csv", index=False)
        status.to_csv(directory / "status.csv", index=False)
    return draws, summary, status


def main():
    parser = argparse.ArgumentParser(description="Run the NH-CES Monte Carlo experiment.")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--methods", type=int, nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, nargs=3, default=(0.2, 2.0, 2.5), metavar=("A", "M", "S"))
    parser.add_argument("--countries", type=int, default=10)
    parser.add_argument("--periods", type=int, default=50)
    parser.add_argument("--measurement-sd", type=float, default=0.01)
    parser.add_argument("--growth-sd", type=float, default=0.01)
    parser.add_argument("--sigma-bounds", type=float, nargs=2, default=(1e-4, 0.9999))
    parser.add_argument("--max-nfev", type=int, default=2000)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path("results/monte_carlo"))
    parser.add_argument("--quiet", action="store_true", help="Suppress replication progress messages.")
    args = parser.parse_args()
    try:
        model = NonhomotheticCES(args.sigma, tuple(args.epsilon))
        _, summary, status = run_monte_carlo(
            model, simulations=args.simulations, seed=args.seed, methods=tuple(args.methods),
            simulation_options={"countries": args.countries, "periods": args.periods,
                                "measurement_sd": args.measurement_sd, "growth_sd": args.growth_sd},
            estimation_options={"sigma_bounds": tuple(args.sigma_bounds),
                                "max_nfev": args.max_nfev, "max_iter": args.max_iter},
            output=args.output, progress=not args.quiet,
        )
    except (ValueError, FloatingPointError, FileExistsError) as error:
        parser.error(str(error))
    print("\nEstimation status:")
    print(status[["method", "runs", "successful", "failed", "mean_seconds"]].to_string(index=False))
    print("\nBias/RMSE conditional on successful finite estimates (blank = unavailable):")
    print(summary[["method", "metric", "used", "truth", "mean", "bias", "rmse"]]
          .to_string(index=False, float_format=lambda x: f"{x:.6g}", na_rep=""))
    print(f"\nResults saved in: {args.output.resolve()}")
    return 1 if (status.successful == 0).any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
