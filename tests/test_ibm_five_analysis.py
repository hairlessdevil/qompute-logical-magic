"""Controls for the five-data/four-syndrome offline analysis."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("five_analysis", SCRIPTS/"analyze_ibm_five_b2ed.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_fourth_syndrome_bit_rejects_and_highest_data_bit_sets_parity():
    u, v = analysis.group_weights("ZIIII", 4, "accepted")
    assert v[0b000001000] == 0  # s3 alone is not an accepted syndrome.
    assert u[0b100000000] == -1 and v[0b100000000] == 1
    assert u[0b000010000] == 1  # data q0 is an identity factor.


def test_five_data_four_syndrome_asymmetric_inversion():
    rng = np.random.default_rng(520)
    matrices = [np.array([[1-a, a], [b, 1-b]]) for a, b in
                [(0.02, .13), (.03, .17), (.04, .09), (.06, .12), (.07, .18),
                 (.08, .11), (.04, .21), (.09, .14), (.05, .19)]]
    sa, da = matrices[:4], matrices[4:]
    parity = np.array([(-1.)**(d & 0b10101).bit_count() for d in range(32)])
    true = rng.dirichlet(np.ones(512)).reshape(32, 16)
    true[:, 0] *= np.where(parity > 0, 16, 2)
    true /= true.sum()
    # Build the physical assignment channel by bit enumeration, without kron.
    channel = np.ones((512, 512))
    indices = np.arange(512)
    for q, matrix in enumerate(matrices):
        channel *= matrix[((indices >> q) & 1)[:, None], ((indices >> q) & 1)[None, :]]
    observed = true.ravel() @ channel
    recorded_zero = np.array([np.prod([sa[q][(s >> q)&1, 0] for q in range(4)]) for s in range(16)])
    retained = true @ recorded_zero
    expected = [retained @ parity/retained.sum(), true[:, 0] @ parity/true[:, 0].sum()]
    u, v = analysis.score_weights("YIXIZ", da, sa)
    actual, _, _ = analysis.ratio_stats(observed*1e6, u, v)
    np.testing.assert_allclose(actual[1:], expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(analysis.tensor_assignment(matrices), channel, atol=1e-14, rtol=0)
    assert abs(expected[0]-expected[1]) > .005
