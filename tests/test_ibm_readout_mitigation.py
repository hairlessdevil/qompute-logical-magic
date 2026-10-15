"""Physics-level controls for asymmetric assignment inversion and conditioning."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "hardware_readout_analysis", Path(__file__).resolve().parents[1]/"scripts/mitigate_ibm_b2ed_readout.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_asymmetric_channels_recover_distinct_selected_ensembles():
    # Data is correlated with the true syndrome: assignment errors change which
    # states recorded-zero selection admits. Bit 0 has a different channel from
    # bit 1, so endian mistakes cannot accidentally pass this control.
    data = [np.array([[.96, .04], [.23, .77]]), np.array([[.79, .21], [.06, .94]])]
    syndrome = [np.array([[.91, .09], [.22, .78]]), np.array([[.95, .05], [.17, .83]])]
    true = np.array([[.32, .01, .01, .02], [.02, .20, .05, .01],
                     [.03, .02, .15, .01], [.04, .03, .02, .06]])
    true /= true.sum()
    # Enumerate every source/destination bit pattern directly. This intentionally
    # does not use the implementation's tensor-product constructor.
    observed = np.zeros_like(true)
    for d in range(4):
        for s in range(4):
            for rd in range(4):
                for rs in range(4):
                    channel = np.prod([data[q][(d >> q)&1, (rd >> q)&1] for q in range(2)])
                    channel *= np.prod([syndrome[q][(s >> q)&1, (rs >> q)&1] for q in range(2)])
                    observed[rd, rs] += true[d, s]*channel
    parity = np.array([1., -1., 1., -1.])  # IZ observes data bit 0 only.
    syn_record_zero = np.array([np.prod([syndrome[q][(s >> q)&1, 0] for q in range(2)]) for s in range(4)])
    retained = true @ syn_record_zero
    expected_recorded = retained @ parity/retained.sum()
    expected_true = true[:, 0] @ parity/true[:, 0].sum()
    u, v = analysis.score_weights("IZ", data, syndrome)
    means, variances, denom = analysis.ratio_stats(observed.ravel()*1e6, u, v)
    np.testing.assert_allclose(means[1:], [expected_recorded, expected_true], atol=1e-12)
    np.testing.assert_allclose(denom[2], true[:, 0].sum(), atol=1e-12)
    assert abs(expected_recorded-expected_true) > .03
    assert abs(means[0]-expected_recorded) > .01
    assert np.all(variances > 0)


def test_identity_calibration_recovers_raw_without_changing_acceptance():
    rng = np.random.default_rng(41)
    counts = rng.multinomial(4000, rng.dirichlet(np.ones(32)))
    u, v = analysis.score_weights("YX", [np.eye(2)]*2, [np.eye(2)]*3)
    means, variances, denominator = analysis.ratio_stats(counts, u, v)
    np.testing.assert_allclose(means, np.repeat(means[0], 3))
    np.testing.assert_allclose(variances, np.repeat(variances[0], 3))
    np.testing.assert_allclose(denominator, np.repeat(counts[::8].sum()/counts.sum(), 3))


def test_joint_ratio_uncertainty_matches_multinomial_sampling():
    rng = np.random.default_rng(72)
    data = [np.array([[.97, .03], [.12, .88]])]*2
    syndrome = [np.array([[.94, .06], [.18, .82]])]
    p = np.array([.28, .04, .10, .08, .16, .09, .15, .10])
    u, v = analysis.score_weights("ZZ", data, syndrome)
    # The joint ratio's numerator and denominator are correlated; treating them
    # as independent would fail this uncertainty check.
    _, variance, _ = analysis.ratio_stats(p*4000, u, v)
    counts = rng.multinomial(4000, p, size=12000)
    ratios = (counts @ u)/(counts @ v)
    np.testing.assert_allclose(np.var(ratios, axis=0, ddof=1), variance, rtol=.06)
