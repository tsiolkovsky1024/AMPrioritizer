"""Sanity checks for the censoring-aware Gaussian likelihood."""
import math

import torch
from scipy.stats import norm

from amp_prioritizer.losses import censored_gaussian_nll


def _nll(mu, log_sigma, target, right=False, left=False):
    t = lambda x: torch.tensor([[x]], dtype=torch.float32)
    mask = torch.ones(1, 1, dtype=torch.bool)
    return float(censored_gaussian_nll(
        t(mu), t(log_sigma), t(target), mask,
        torch.tensor([[right]]), torch.tensor([[left]]),
        reduction="cell_mean",
    ))


def test_exact_matches_gaussian_density_up_to_constant():
    # The exact-row term omits the constant 0.5*log(2*pi), which has no gradient.
    mu, sigma, y = 1.0, math.exp(0.3), 1.8
    expected = -norm.logpdf(y, loc=mu, scale=sigma) - 0.5 * math.log(2 * math.pi)
    assert abs(_nll(mu, 0.3, y) - expected) < 1e-4


def test_right_censored_uses_survival_term():
    mu, sigma, tau = 1.0, math.exp(0.3), 2.0
    expected = -norm.logsf(tau, loc=mu, scale=sigma)
    assert abs(_nll(mu, 0.3, tau, right=True) - expected) < 1e-4


def test_left_censored_uses_cdf_term():
    mu, sigma, tau = 1.0, math.exp(0.3), 0.5
    expected = -norm.logcdf(tau, loc=mu, scale=sigma)
    assert abs(_nll(mu, 0.3, tau, left=True) - expected) < 1e-4


def test_right_censored_does_not_penalise_predictions_above_threshold():
    below = _nll(0.0, 0.0, 2.0, right=True)
    above = _nll(4.0, 0.0, 2.0, right=True)
    assert above < below
