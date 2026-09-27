"""Known-answer checks for the NH-CES estimators.

Run from the repository root: python -m pytest tests/test_recovery.py -v

Generate data with known parameters and zero measurement error, then estimate
without passing the unknown parameters to the estimator. Growth shocks remain
active so prices and consumption vary across observations.

Only methods 1, 2, and 7 are required to recover the exact model parameters.
epsilon_m is a supplied normalization, not an estimated parameter. Method 7
additionally observes C; methods 1 and 2 see only prices, shares, expenditure,
and country-period identifiers. Methods 3-6 involve approximations or proxy
regressions and are not required to recover exact parameters in this test.

These checks validate implementation under designed conditions. They do not
establish consistency under measurement error or validity on real data.
Author: Wenzhuo Wang.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from estimation.model import NonhomotheticCES
from estimation.simulate import simulate_panel
from estimation.estimators import estimate


@pytest.fixture(scope="module", params=[
    pytest.param((0.5, (0.2, 2.0, 2.5), 1, (1e-4, 0.9999)), id="baseline"),
    pytest.param((0.3, (0.4, 0.9, 1.8), 17, (1e-4, 0.9999)), id="different_parameters"),
    pytest.param((1.4, (0.4, 0.9, 1.8), 17, (1.0001, 3.0)), id="sigma_above_one"),
])
def exact_sample(request):
    sigma, epsilon, seed, bounds = request.param
    model = NonhomotheticCES(sigma, epsilon)
    observed, truth = simulate_panel(model, seed=seed, measurement_sd=0)
    return model, observed, truth, bounds


@pytest.fixture(scope="module")
def baseline():
    model = NonhomotheticCES(0.5, (0.2, 2.0, 2.5))
    observed, truth = simulate_panel(model, seed=1, measurement_sd=0)
    return model, observed, truth


@pytest.mark.parametrize("method", (1, 2, 7))
def test_exact_parameter_recovery(exact_sample, method):
    model, observed, truth, bounds = exact_sample
    result = estimate(
        observed, method=method,
        truth=truth if method == 7 else None,
        epsilon_m=model.epsilon[1], sigma_bounds=bounds,
    )

    assert result.success, result.message
    assert result.epsilon is not None
    # Numerical optimization needs a tolerance, not exact floating-point equality.
    assert_allclose(result.sigma, model.sigma, rtol=1e-5, atol=1e-6)
    assert_allclose(np.array(result.epsilon)[[0, 2]],
                    np.array(model.epsilon)[[0, 2]], rtol=1e-5, atol=1e-6)
    assert result.epsilon[1] == model.epsilon[1]  # Supplied, not recovered.


@pytest.mark.parametrize("method", (1, 2))
def test_gmm_reports_ratios_without_a_normalization(baseline, method):
    model, observed, _ = baseline
    result = estimate(observed, method=method)

    assert result.success, result.message
    assert result.epsilon is None
    assert_allclose(result.epsilon_ratios,
                    np.array(model.epsilon)/model.epsilon[1], rtol=1e-5, atol=1e-6)


def test_known_consumption_slopes_match_the_notes(baseline):
    model, observed, truth = baseline
    result = estimate(observed, method=7, truth=truth)
    differences = np.array(model.epsilon)-model.epsilon[1]

    assert result.success, result.message
    assert result.epsilon is None
    assert_allclose(result.epsilon_differences, differences, atol=1e-9)
    # The raw regression slopes include the factor (1-sigma).
    assert_allclose(result.coefficients,
                    (1-model.sigma)*differences[[0, 2]], atol=1e-9)


def test_changing_consumption_scale_changes_epsilon_scale(baseline):
    model, observed, truth = baseline
    rescaled = truth.copy()
    rescaled["consumption"] = truth.consumption**2
    result = estimate(observed, method=7, truth=rescaled,
                      epsilon_m=model.epsilon[1]/2)

    # C_new=C_old**2 represents the same bundles with epsilon_new=epsilon_old/2.
    assert result.success, result.message
    assert_allclose(result.sigma, model.sigma, atol=1e-9)
    assert_allclose(result.epsilon, np.array(model.epsilon)/2, atol=1e-9)


def test_gmm_is_invariant_to_currency_units(baseline):
    model, observed, _ = baseline
    rescaled = observed.copy()
    columns = ["expenditure", "price_a", "price_m", "price_s"]
    rescaled[columns] *= 7
    result = estimate(rescaled, method=1, epsilon_m=model.epsilon[1])

    assert result.success, result.message
    assert_allclose(result.sigma, model.sigma, rtol=1e-5, atol=1e-6)
    assert_allclose(result.epsilon, model.epsilon, rtol=1e-5, atol=1e-6)


def test_benchmark_matches_truth_by_keys_not_row_order(baseline):
    model, observed, truth = baseline
    result = estimate(
        observed.sample(frac=1, random_state=3), method=7,
        truth=truth.sample(frac=1, random_state=4), epsilon_m=model.epsilon[1],
    )

    assert result.success, result.message
    assert_allclose(result.sigma, model.sigma, atol=1e-9)
    assert_allclose(result.epsilon, model.epsilon, atol=1e-9)


@pytest.mark.parametrize("problem", ("missing", "wrong_keys"))
def test_benchmark_rejects_unavailable_truth(baseline, problem):
    _, observed, truth = baseline
    if problem == "missing":
        supplied = None
    else:
        supplied = truth.copy()
        supplied.loc[0, "country"] = 999
    result = estimate(observed, method=7, truth=supplied)

    assert not result.success
    assert result.message


@pytest.mark.parametrize("method, options", [
    (1, {"max_nfev": 1}),
    (2, {"max_nfev": 1}),
    (3, {"max_iter": 1}),
])
def test_unfinished_estimation_is_not_reported_as_success(baseline, method, options):
    _, observed, _ = baseline
    result = estimate(observed, method=method, **options)

    assert not result.success
    assert result.message


@pytest.mark.parametrize("method", (1, 7))
def test_unidentified_data_do_not_pass_recovery(baseline, method):
    _, observed, truth = baseline
    constant = observed.copy()
    columns = ["expenditure", "price_a", "price_m", "price_s",
               "share_a", "share_m", "share_s"]
    constant[columns] = np.tile(observed[columns].iloc[0].to_numpy(), (len(observed), 1))
    result = estimate(constant, method=method, truth=truth if method == 7 else None)

    assert not result.success
    assert result.message


@pytest.mark.parametrize("method", (4, 5))
def test_proxy_coefficients_are_not_labeled_as_epsilon(baseline, method):
    model, observed, _ = baseline
    result = estimate(observed, method=method, epsilon_m=model.epsilon[1])

    assert result.success, result.message
    assert result.coefficients is not None
    assert np.isfinite(result.coefficients).all()
    assert result.epsilon is None
    assert result.epsilon_ratios is None
    if method == 4:
        assert result.sigma is None
