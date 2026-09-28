# SPDX-License-Identifier: CC-BY-SA-4.0
"""Simulate a three-sector country panel for the NH-CES model.

Adapted from Marti Mestieri's simulation_countrypanel_web.do, distributed
under CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/
Source website: https://mestieri.github.io/
Python adaptation: Wenzhuo Wang, The University of Chicago.

Changes: model calculations use model.py and the study notes' epsilon
definition; NumPy generates the draws; observed data and truth are separated.
The original growth processes and measurement-error specification are kept.
NumPy and Stata do not produce identical samples from the same seed.

Sector order is agriculture (a), manufacturing (m), services (s).
To match the author's baseline while keeping the same consumption index C,
pass NonhomotheticCES(sigma=0.5, epsilon=(0.2, 2.0, 2.5)).
The original exponents (0.1, 1, 1.25) equal (1-sigma)*epsilon.

Dependencies: NumPy, pandas, and the local model module (which uses SciPy).
"""

import numpy as np
import pandas as pd

from .model import NonhomotheticCES


def simulate_panel(
    model: NonhomotheticCES,
    *,
    countries: int = 10,
    periods: int = 50,
    consumption_growth: float = 0.02,
    price_growth: tuple[float, float, float] = (0.022, 0.017, 0.016),
    growth_sd: float = 0.01,
    measurement_sd: float = 0.01,
    initial_price: float = 100.0,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return observed and truth tables sorted by country and period.

    For each country-period pair, prices and consumption receive independent
    growth shocks following the original Stata equations. These shocks are
    observation-specific, not recursively accumulated.

    Observed prices, expenditure, and shares receive independent lognormal
    measurement errors. Shares are capped at one separately and are not
    renormalized, so they may not sum to one.

    The observed table is used by ordinary estimators. The truth table also
    contains consumption, price index, and quantities for validation and method 7.

    Set measurement_sd=0 for an exact model-consistent sample. The function
    does not write files or modify NumPy's global random state.
    """
    for name, count in (("countries", countries), ("periods", periods)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer.")
        if count <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if len(model.epsilon) != 3:
        raise ValueError("This simulation requires three goods, ordered a, m, s.")
    price_growth = np.asarray(price_growth, dtype=float)
    if price_growth.shape != (3,) or not np.all(np.isfinite(price_growth)):
        raise ValueError("price_growth must contain three finite rates.")
    if not np.isfinite(consumption_growth) or consumption_growth <= -1:
        raise ValueError("consumption_growth must be finite and greater than -1.")
    if np.any(price_growth <= -1):
        raise ValueError("Each price growth rate must be greater than -1.")
    for name, sd in (("growth_sd", growth_sd), ("measurement_sd", measurement_sd)):
        if not np.isfinite(sd) or sd < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if not np.isfinite(initial_price) or initial_price <= 0:
        raise ValueError("initial_price must be finite and strictly positive.")

    rng = np.random.default_rng(seed)
    size = countries * periods
    country = np.repeat(np.arange(1, countries + 1), periods)
    period = np.tile(np.arange(1, periods + 1), countries)

    consumption_base = 1 + consumption_growth + rng.normal(0, growth_sd, size)
    price_base = 1 + price_growth + rng.normal(0, growth_sd, (size, 3))
    if np.any(consumption_base <= 0) or np.any(price_base <= 0):
        raise ValueError("A growth draw gave a nonpositive base; reduce growth_sd.")

    # Work in logs to evaluate the original power formulas more stably.
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        log_c = np.log(np.log1p(country)) + period * np.log(consumption_base)
        log_prices = np.log(initial_price) - period[:, None] * np.log(price_base)
        consumption = np.exp(log_c)
        prices = np.exp(log_prices)

        expenditure = model.expenditure(prices, consumption)
        shares = model.shares(prices, consumption)
        quantities = model.hicksian_demand(prices, consumption)
        price_index = expenditure / consumption

        # Preserve Stata's independent measurement errors and share clipping.
        share_error = rng.normal(0, measurement_sd, (size, 3))
        price_error = rng.normal(0, measurement_sd, (size, 3))
        expenditure_error = rng.normal(0, measurement_sd, size)
        measured_shares = np.exp(np.minimum(np.log(shares) + share_error, 0.0))
        measured_prices = np.exp(log_prices + price_error)
        measured_expenditure = np.exp(np.log(expenditure) + expenditure_error)

    observed = pd.DataFrame({
        "country": country,
        "period": period,
        "expenditure": measured_expenditure,
    })
    truth = pd.DataFrame({
        "country": country,
        "period": period,
        "consumption": consumption,
        "expenditure": expenditure,
        "price_index": price_index,
    })
    for i, sector in enumerate(("a", "m", "s")):
        observed[f"price_{sector}"] = measured_prices[:, i]
        observed[f"share_{sector}"] = measured_shares[:, i]
        truth[f"price_{sector}"] = prices[:, i]
        truth[f"share_{sector}"] = shares[:, i]
        truth[f"quantity_{sector}"] = quantities[:, i]

    return observed, truth
