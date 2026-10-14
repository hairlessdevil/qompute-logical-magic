import numpy as np
import pytest

from logical_magic import CODES
from logical_magic.fault_examples import run_representative_examples


@pytest.mark.parametrize("name", ["four", "five"])
def test_representative_locations_and_dynamic_recovery(name):
    result = run_representative_examples(CODES[name], shots=32, seed=19)
    rows = {r["name"]: r for r in result["rows"]}
    assert len(rows) == 8
    for key in ["control", "storage_X0"]:
        assert np.isclose(rows[key]["fidelity_after_recovery"], 1)
    input_fault = rows["input_Y0"]
    assert np.isclose(input_fault["zero_syndrome_probability"], 1)
    assert np.isclose(input_fault["codespace_after_recovery"], 1)
    assert np.isclose(input_fault["fidelity_after_recovery"], 0 if name == "four" else 1/3)
    assert np.isclose(input_fault["fidelity_after_hypothetical_ideal_round"], input_fault["fidelity_after_recovery"])
    for key in ["late_X0", "readout_flip", "faulty_recovery"]:
        assert np.isclose(rows[key]["fidelity_after_recovery"], 0)
        assert np.isclose(rows[key]["fidelity_after_hypothetical_ideal_round"], 1)
    assert rows["faulty_recovery"]["events"] == 2
    assert all(r["events"] == 1 for k, r in rows.items() if k not in ["control", "faulty_recovery"])
