# SPDX-License-Identifier: CC-BY-SA-4.0
"""Seven NH-CES estimators adapted from Marti Mestieri's Stata sample code.

Source: simulation_countrypanel_web.do, https://mestieri.github.io/
Original license: https://creativecommons.org/licenses/by-sa/4.0/
Python adaptation: Wenzhuo Wang, The University of Chicago.

Methods: 1 log-linear GMM; 2 nonlinear-price-index GMM; 3 iterative SUR;
4 expenditure OLS; 5 Tornqvist SUR; 6 hybrid SUR/GMM; 7 known-C SUR.

Estimating equation (study-note notation):
    log(s_i/s_m) = (1-sigma)*log(p_i/p_m)
                   + (1-sigma)*(epsilon_i-epsilon_m)*log(C) + country_FE.

Identification: methods 1, 2, 3, 6 estimate ratios epsilon_i/epsilon_m;
method 7 estimates differences on the supplied C scale; methods 4 and 5
return proxy coefficients only. Point estimates only; no standard errors.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import logsumexp, softmax


METHODS = {
    1: "loglinear_gmm",
    2: "nonlinear_gmm",
    3: "iterative_sur",
    4: "expenditure_ols",
    5: "tornqvist_sur",
    6: "hybrid_gmm",
    7: "known_consumption_sur",
}


@dataclass
class EstimateResult:
    """Parameter triples use (a, m, s); coefficient pairs use (a, s).

    epsilon is reported only when epsilon_m was explicitly supplied.
    coefficients are slopes on log expenditure (4) or log consumption (5, 7).
    objective is a GMM criterion (1, 2, 6), a parameter-change norm (3), or
    residual mean square (4, 5, 7); do not compare it across methods.
    success describes numerical completion, not consistency or lack of bias.
    """

    method: int
    success: bool = False
    sigma: float | None = None
    epsilon_ratios: tuple[float, float, float] | None = None
    epsilon_differences: tuple[float, float, float] | None = None
    epsilon: tuple[float, float, float] | None = None
    coefficients: tuple[float, float] | None = None
    iterations: int = 0
    evaluations: int = 0
    objective: float | None = None
    seconds: float = 0.0
    message: str = ""


@dataclass
class _Panel:
    frame: pd.DataFrame
    p: np.ndarray
    s: np.ndarray
    e: np.ndarray
    y: np.ndarray
    x: np.ndarray
    fixed: np.ndarray
    first: np.ndarray


def _prepare(data):
    sectors = ("a", "m", "s")
    prices = [f"price_{s}" for s in sectors]
    shares = [f"share_{s}" for s in sectors]
    columns = ["country", "period", "expenditure"] + prices + shares
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = data[columns].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("The panel must be nonempty and have no missing values.")
    if frame.duplicated(["country", "period"]).any():
        raise ValueError("Country-period pairs must be unique.")
    values = frame[["period", "expenditure"] + prices + shares].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Periods, expenditure, prices, and shares must be finite and positive.")
    if (values[:, 0] != np.floor(values[:, 0])).any():
        raise ValueError("Periods must be integers.")
    if (frame[shares].to_numpy(float) > 1).any():
        raise ValueError("Individual shares must not exceed one; their sum may differ from one.")
    frame = frame.sort_values(["country", "period"]).reset_index(drop=True)
    p = np.log(frame[prices].to_numpy(float))
    s = frame[shares].to_numpy(float)
    e = np.log(frame.expenditure.to_numpy(float))
    fixed = pd.get_dummies(frame.country, dtype=float).to_numpy()
    first = np.r_[True, frame.country.to_numpy()[1:] != frame.country.to_numpy()[:-1]]
    return _Panel(frame, p, s, e, np.log(s[:, [0, 2]]) - np.log(s[:, [1]]),
                  p[:, [0, 2]] - p[:, [1]], fixed, first)


def _ols(x, y):
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1] or x.shape[0] <= x.shape[1]:
        raise np.linalg.LinAlgError("Insufficient independent variation or residual degrees of freedom.")
    return beta


def _whitener(covariance):
    values, vectors = np.linalg.eigh(covariance)
    largest = values.max()
    if not np.isfinite(largest) or largest <= 0:
        raise np.linalg.LinAlgError("Covariance matrix has no positive finite eigenvalue.")
    floor = largest * 1e-10
    regularized = bool(np.any(values < floor))
    # L.T @ L is the inverse covariance, up to an irrelevant scalar factor.
    return np.sqrt(largest / np.maximum(values, floor))[:, None] * vectors.T, regularized


def _sur(panel, log_c):
    """Two-equation SUR: one price slope, two C slopes, two sets of country FE."""
    n, k = panel.fixed.shape
    design = np.zeros((n, 2, 3 + 2*k))
    design[:, :, 0] = panel.x
    design[:, 0, 1] = log_c
    design[:, 1, 2] = log_c
    design[:, 0, 3:3+k] = panel.fixed
    design[:, 1, 3+k:] = panel.fixed
    beta = _ols(design.reshape(2*n, -1), panel.y.ravel())
    prediction = np.einsum("nij,j->ni", design, beta)
    if np.max(np.abs(panel.y - prediction)) < 1e-9:
        return beta, prediction, "Exact fit; SUR covariance reweighting skipped."
    errors = np.empty_like(panel.y)
    for j in range(2):
        x = np.column_stack((panel.x[:, j], log_c, panel.fixed))
        errors[:, j] = panel.y[:, j] - x @ _ols(x, panel.y[:, j])
    whitening, regularized = _whitener(errors.T @ errors / n)
    weighted_x = np.einsum("ab,nbk->nak", whitening, design).reshape(2*n, -1)
    weighted_y = (panel.y @ whitening.T).ravel()
    beta = _ols(weighted_x, weighted_y)
    prediction = np.einsum("nij,j->ni", design, beta)
    note = "SUR covariance regularized." if regularized else "Two-step SUR completed."
    return beta, prediction, note


def _tornqvist(panel):
    # Bilateral index relative to each country's first period, as in Stata.
    base = np.maximum.accumulate(np.where(panel.first, np.arange(len(panel.e)), 0))
    log_p = np.sum(0.5 * (panel.s + panel.s[base]) * (panel.p - panel.p[base]), axis=1)
    return panel.e - log_p


def _instruments(panel):
    # Intercept plus K-1 country dummies avoids the redundant full dummy set.
    common = np.column_stack((panel.e, panel.p[:, 1], np.log(panel.s[:, 1]),
                              np.ones(len(panel.e)), panel.fixed[:, 1:]))
    instruments = [np.column_stack((common, panel.p[:, j])) for j in (0, 2)]
    for z in instruments:
        if np.linalg.matrix_rank(z) < z.shape[1]:
            raise np.linalg.LinAlgError("The instrument matrix is rank deficient.")
    return instruments


def _gmm(panel, residual, start, bounds, max_nfev):
    z = _instruments(panel)

    def moments(theta):
        errors = residual(theta)
        return np.column_stack([z[j] * errors[:, [j]] for j in range(2)])

    def fit(initial, whitening):
        return least_squares(lambda theta: whitening @ moments(theta).mean(axis=0),
                             initial, bounds=bounds, x_scale="jac", max_nfev=max_nfev,
                             ftol=1e-11, xtol=1e-11, gtol=1e-11)

    first = fit(start, np.eye(sum(x.shape[1] for x in z)))
    fit_result = first
    evaluations = first.nfev
    stages = 1
    note = str(first.message)
    if first.success and np.max(np.abs(residual(first.x))) >= 1e-9:
        m = moments(first.x)
        whitening, regularized = _whitener(m.T @ m / len(m))
        fit_result = fit(first.x, whitening)
        evaluations += fit_result.nfev
        stages = 2
        note = str(fit_result.message)
        if regularized:
            note += " Moment covariance regularized."
    elif first.success:
        note += " Exact fit; GMM covariance reweighting skipped."
    identified = np.linalg.matrix_rank(fit_result.jac) == len(fit_result.x)
    interior = not np.any(fit_result.active_mask)
    success = bool(fit_result.success and identified and interior)
    if not identified:
        note += " Moment Jacobian is rank deficient."
    if not interior:
        note += " Parameter bound reached; inspect the specification or bounds."
    return fit_result.x, success, float(fit_result.fun @ fit_result.fun), stages, evaluations, note


def _linear_residual(panel, sigma, ratios, effects):
    q = 1 - sigma
    latent = np.log(panel.s[:, 1]) + q * (panel.e - panel.p[:, 1])
    return panel.y - q * panel.x - latent[:, None] * (ratios - 1) - effects


def _nonlinear_residual(panel, sigma, ratios, effects):
    q = 1 - sigma
    r = np.array([ratios[0], 1.0, ratios[1]])
    fe = np.column_stack((effects[:, 0], np.zeros(len(panel.e)), effects[:, 1]))
    log_terms = (q / r) * (fe + q * panel.p)
    log_terms += (1 - q / r) * (np.log(panel.s) + q * panel.e[:, None])
    log_p = logsumexp(log_terms, axis=1) / q
    return panel.y - q * panel.x - (panel.e - log_p)[:, None] * (ratios - 1) - effects


def _gmm_estimate(panel, method, result, sigma_bounds, max_nfev):
    k = panel.fixed.shape[1]
    if method in (1, 2):
        lower, upper = sigma_bounds
        start = np.r_[np.clip(0.75, lower + (upper-lower)*0.1, upper - (upper-lower)*0.1),
                      0.25, 1.75, np.zeros(2*k)]
        bounds = (np.r_[lower, 1e-6, 1e-6, np.full(2*k, -np.inf)],
                  np.r_[upper, np.inf, np.inf, np.full(2*k, np.inf)])
        equation = _linear_residual if method == 1 else _nonlinear_residual

        def residual(theta):
            effects = panel.fixed @ theta[3:].reshape(2, k).T
            return equation(panel, theta[0], theta[1:3], effects)

        theta, ok, obj, stages, evaluations, note = _gmm(panel, residual, start, bounds, max_nfev)
        result.sigma = float(theta[0])
        ratios = theta[1:3]
    else:
        beta, _, sur_note = _sur(panel, _tornqvist(panel))
        result.sigma = float(1 - beta[0])
        slopes = beta[1:3]
        result.coefficients = tuple(float(value) for value in slopes)
        # Require positive ratios 1 + scale*slopes, without fixing scale's sign.
        lower, upper = -np.inf, np.inf
        for slope in slopes:
            if slope > 0:
                lower = max(lower, (1e-6 - 1) / slope)
            elif slope < 0:
                upper = min(upper, (1e-6 - 1) / slope)
        initial = 1.0
        if initial <= lower or initial >= upper:
            initial = 0.0
        start = np.r_[initial, np.zeros(2*k)]
        bounds = (np.r_[lower, np.full(2*k, -np.inf)],
                  np.r_[upper, np.full(2*k, np.inf)])

        def residual(theta):
            effects = panel.fixed @ theta[1:].reshape(2, k).T
            return _linear_residual(panel, result.sigma, 1 + theta[0]*slopes, effects)

        theta, ok, obj, stages, evaluations, note = _gmm(panel, residual, start, bounds, max_nfev)
        ratios = 1 + theta[0]*slopes
        note = sur_note + " " + note
    result.epsilon_ratios = (float(ratios[0]), 1.0, float(ratios[1]))
    result.success, result.objective = ok, obj
    result.iterations, result.evaluations, result.message = stages, evaluations, note


def _iterative(panel, result, max_iter, tolerance):
    if np.any(np.diff(panel.frame.period.to_numpy())[~panel.first[1:]] != 1):
        raise ValueError("Method 3 requires consecutive periods within each country.")
    original = _tornqvist(panel)
    increments = np.r_[0.0, np.diff(original)]
    increments[panel.first] = 0.0
    log_c = original.copy()
    old = None
    any_regularized = False
    for iteration in range(1, max_iter + 1):
        beta, prediction, note = _sur(panel, log_c)
        any_regularized |= "regularized" in note
        q = beta[0]
        if abs(q) < 1e-8:
            raise ValueError("Method 3 has a near-zero relative-price coefficient.")
        ratios = np.array([1 + beta[1]/q, 1.0, 1 + beta[2]/q])
        if np.any(ratios <= 0):
            raise ValueError("Method 3 produced a nonpositive preference ratio.")
        current = np.r_[1-q, ratios[[0, 2]]]
        result.sigma = float(1-q)
        result.epsilon_ratios = tuple(float(value) for value in ratios)
        result.iterations = iteration
        if old is not None:
            result.objective = float(np.linalg.norm(current-old) / 3)
            if result.objective <= tolerance:
                result.success = True
                result.message = "Iterative SUR converged."
                break
        old = current
        predicted_shares = softmax(np.column_stack((prediction[:, 0],
                                                    np.zeros(len(original)), prediction[:, 1])), axis=1)
        lag = np.roll(predicted_shares, 1, axis=0)
        average = (0.5 * (predicted_shares + lag)) @ ratios
        changes = increments / average
        # Reset integration at the first observation of each country.
        for start, stop in zip(np.flatnonzero(panel.first),
                               np.r_[np.flatnonzero(panel.first)[1:], len(original)]):
            log_c[start:stop] = original[start] + np.cumsum(changes[start:stop])
    else:
        result.message = "Iterative SUR reached max_iter without convergence."
    if any_regularized:
        result.message += " SUR covariance regularized in at least one iteration."


def _known_consumption(panel, truth):
    if truth is None or not {"country", "period", "consumption"}.issubset(truth.columns):
        raise ValueError("Method 7 requires truth with country, period, consumption.")
    if len(truth) != len(panel.frame) or truth.duplicated(["country", "period"]).any():
        raise ValueError("Truth must have exactly the same unique country-period keys.")
    matched = panel.frame[["country", "period"]].merge(
        truth[["country", "period", "consumption"]], on=["country", "period"],
        how="left", validate="one_to_one", sort=False)
    c = matched.consumption.to_numpy(float)
    if not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("Truth keys must match, with finite positive consumption.")
    return np.log(c)


def estimate(
    data: pd.DataFrame,
    method: int = 1,
    *,
    truth: pd.DataFrame | None = None,
    epsilon_m: float | None = None,
    sigma_bounds: tuple[float, float] = (1e-4, 0.9999),
    max_nfev: int = 2000,
    max_iter: int = 250,
    tolerance: float = 1e-4,
) -> EstimateResult:
    """Estimate one method from simulated observed data.

    Only method 7 uses truth, matched by country-period keys.
    epsilon_m fixes the normalization; None returns ratios or differences.
    For method 7, it must match truth's consumption scale.

    sigma_bounds applies only to methods 1–2; use bounds above one
    for sigma > 1. Nonlinear ratios must exceed 1e-6.

    Invalid options raise ValueError. Data or estimation failures return
    success=False with a message and may retain the last estimates.
    """
    if isinstance(method, (bool, np.bool_)) or not isinstance(method, (int, np.integer)) or method not in METHODS:
        raise ValueError("method must be an integer from 1 through 7.")
    if epsilon_m is not None and (not np.isfinite(epsilon_m) or epsilon_m <= 0):
        raise ValueError("epsilon_m must be a finite positive normalization.")
    lower, upper = sigma_bounds
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 < lower < upper):
        raise ValueError("sigma_bounds must be finite, positive, and increasing.")
    if lower <= 1 <= upper:
        raise ValueError("sigma_bounds must lie entirely below or above one.")
    for name, value in (("max_nfev", max_nfev), ("max_iter", max_iter)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    result = EstimateResult(method=method)
    started = perf_counter()
    try:
        panel = _prepare(data)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if method in (1, 2, 6):
                _gmm_estimate(panel, method, result, sigma_bounds, max_nfev)
            elif method == 3:
                _iterative(panel, result, max_iter, tolerance)
            elif method == 4:
                x = np.column_stack((panel.p, panel.e, panel.fixed))
                beta = _ols(x, panel.y)
                result.coefficients = tuple(float(value) for value in beta[3])
                result.objective = float(np.mean((panel.y - x @ beta)**2))
                result.success, result.iterations = True, 1
                result.message = "OLS coefficients only; sigma and epsilon levels are not identified here."
            else:
                log_c = _known_consumption(panel, truth) if method == 7 else _tornqvist(panel)
                beta, prediction, note = _sur(panel, log_c)
                result.sigma = float(1-beta[0])
                result.coefficients = tuple(float(value) for value in beta[1:3])
                result.objective = float(np.mean((panel.y-prediction)**2))
                result.success, result.message = True, note
                result.iterations = 1 if note.startswith("Exact fit") else 2
                if method == 7:
                    if abs(beta[0]) < 1e-8:
                        raise ValueError("Cannot recover epsilon differences when sigma is near one.")
                    result.epsilon_differences = (float(beta[1]/beta[0]), 0.0, float(beta[2]/beta[0]))
                else:
                    result.message += " Tornqvist slopes are approximate coefficients, not epsilon levels."
            if epsilon_m is not None:
                if result.epsilon_ratios is not None:
                    result.epsilon = tuple(float(epsilon_m * value) for value in result.epsilon_ratios)
                elif result.epsilon_differences is not None:
                    result.epsilon = tuple(float(epsilon_m + value) for value in result.epsilon_differences)
            if result.sigma is not None and (not np.isfinite(result.sigma) or result.sigma <= 0 or abs(result.sigma-1) < 1e-8):
                result.success = False
                result.message += " Estimated sigma is outside the model domain."
            if result.epsilon is not None and np.any(np.array(result.epsilon) <= 0):
                result.success = False
                result.message += " Supplied normalization yields nonpositive epsilon."
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        result.success = False
        result.message = str(error)
    result.seconds = perf_counter() - started
    return result
