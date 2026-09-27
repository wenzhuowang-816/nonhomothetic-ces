"""Nonhomothetic CES preferences, using the notation of the study notes.

    sum_i (c_i / C**epsilon_i)**((sigma - 1) / sigma) = 1
    E(p, C) = [sum_i p_i**(1-sigma) C**((1-sigma)*epsilon_i)]**(1/(1-sigma))

There are no preference weights. Prices and quantities have goods on their
last axis; leading axes represent observations. C and E broadcast over those
leading axes. All inputs must be finite and strictly positive; sigma != 1.

Parameter convention matters: Mestieri's simulation_countrypanel_web.do uses
an exponent delta_i on C in the expenditure sum. To preserve the SAME C,
set epsilon_i = delta_i / (1-sigma). Thus its sigma=0.5 and delta=(0.1, 1, 1.25)
correspond to epsilon=(0.2, 2, 2.5) here. epsilon is not expenditure elasticity.

Theory: Comin, Lashkari and Mestieri (2021), Structural Change with Long-run
Income and Price Effects, Econometrica. Numerical implementation: Wenzhuo Wang.
Dependencies: NumPy and SciPy.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import logsumexp


def _positive(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must be finite and strictly positive.")
    return array


def _exp(value: ArrayLike) -> NDArray[np.float64]:
    # Report unrepresentable levels instead of silently returning zero or inf.
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = np.exp(value)
    if np.any(result == 0):
        raise FloatingPointError("Result underflowed; rescale the inputs.")
    return result


@dataclass(frozen=True)
class NonhomotheticCES:
    """A fixed preference specification: sigma > 0, sigma != 1, epsilon_i > 0.

    Supply epsilon in the same order as prices and quantities. Equal epsilon
    values give homothetic preferences; all ones give standard CES.
    The unweighted specification does not implement a sigma=1 limit.
    """

    sigma: float
    epsilon: tuple[float, ...]

    def __post_init__(self):
        sigma = float(self.sigma)
        if not np.isfinite(sigma) or sigma <= 0 or sigma == 1:
            raise ValueError("sigma must be positive, finite, and different from 1.")
        epsilon = _positive(self.epsilon, "epsilon")
        if epsilon.ndim != 1 or epsilon.size == 0:
            raise ValueError("epsilon must be a nonempty one-dimensional sequence.")
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "epsilon", tuple(epsilon.tolist()))

    def _goods(self, values: ArrayLike, name: str) -> NDArray[np.float64]:
        values = _positive(values, name)
        if values.ndim == 0 or values.shape[-1] != len(self.epsilon):
            raise ValueError(f"{name} must have {len(self.epsilon)} goods on its last axis.")
        return values

    def _inputs(self, prices, value, name):
        prices = self._goods(prices, "prices")
        value = _positive(value, name)
        shape = np.broadcast_shapes(prices.shape[:-1], value.shape)
        return (
            np.broadcast_to(prices, shape + (len(self.epsilon),)),
            np.broadcast_to(value, shape),
        )

    def _log_terms(self, prices, consumption):
        return (1 - self.sigma) * (
            np.log(prices) + np.log(consumption)[..., None] * self.epsilon
        )

    def _solve_index(self, log_terms, power, target):
        """Solve logsumexp(a_i + power*epsilon_i*x)/power = target for x=log C.

        The derivative is a weighted mean of epsilon, hence strictly positive.
        Its min/max bounds give a bracket without an arbitrary search range.
        """
        shape = log_terms.shape[:-1]
        target = np.broadcast_to(target, shape)
        epsilon = np.asarray(self.epsilon)
        roots = np.empty(target.size)
        rows = log_terms.reshape(-1, len(epsilon))

        for i, (row, goal) in enumerate(zip(rows, target.ravel())):
            def residual(x):
                return logsumexp(row + power * epsilon * x) / power - goal

            gap = -residual(0.0)
            bounds = (gap / epsilon.min(), gap / epsilon.max())
            roots[i] = brentq(
                residual, min(bounds) - 1.0, max(bounds) + 1.0,
                xtol=1e-12, rtol=1e-12,
            )
        return _exp(roots.reshape(shape))

    def expenditure(self, prices: ArrayLike, consumption: ArrayLike):
        """Minimum expenditure E(p, C) needed to attain consumption index C."""
        prices, consumption = self._inputs(prices, consumption, "consumption")
        log_e = logsumexp(self._log_terms(prices, consumption), axis=-1)
        return _exp(log_e / (1 - self.sigma))

    def consumption_index(self, prices: ArrayLike, expenditure: ArrayLike):
        """Recover C from prices and total expenditure by solving E(p, C)=E."""
        prices, expenditure = self._inputs(prices, expenditure, "expenditure")
        power = 1 - self.sigma
        return self._solve_index(power * np.log(prices), power, np.log(expenditure))

    def aggregate(self, quantities: ArrayLike):
        """Recover C directly from a goods bundle using the implicit aggregator."""
        quantities = self._goods(quantities, "quantities")
        rho = (self.sigma - 1) / self.sigma
        return self._solve_index(rho * np.log(quantities), -rho, 0.0)

    def shares(self, prices: ArrayLike, consumption: ArrayLike):
        """Expenditure shares s_i at prices p and consumption index C."""
        prices, consumption = self._inputs(prices, consumption, "consumption")
        terms = self._log_terms(prices, consumption)
        return _exp(terms - logsumexp(terms, axis=-1, keepdims=True))

    def hicksian_demand(self, prices: ArrayLike, consumption: ArrayLike):
        """Cost-minimizing quantities for a given C: c_i = s_i E(p,C) / p_i."""
        prices, consumption = self._inputs(prices, consumption, "consumption")
        terms = self._log_terms(prices, consumption)
        total = logsumexp(terms, axis=-1, keepdims=True)
        return _exp(terms - total + total / (1 - self.sigma) - np.log(prices))

    def demand(self, prices: ArrayLike, expenditure: ArrayLike):
        """Marshallian quantities at prices p and budget E; solve for C first."""
        consumption = self.consumption_index(prices, expenditure)
        return self.hicksian_demand(prices, consumption)

    def expenditure_elasticities(self, prices: ArrayLike, expenditure: ArrayLike):
        """eta_i = d log(c_i)/d log(E), holding prices fixed."""
        consumption = self.consumption_index(prices, expenditure)
        shares = self.shares(prices, consumption)
        average = np.sum(shares * self.epsilon, axis=-1, keepdims=True)
        return self.sigma + (1 - self.sigma) * np.asarray(self.epsilon) / average
