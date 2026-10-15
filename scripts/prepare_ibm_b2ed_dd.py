#!/usr/bin/env python3
"""Prepare explicit XY4 for an existing B2_ED bundle; never submit a job."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from logical_magic.ibm_b2ed import INSTANCE, BACKEND, QiskitRuntimeService, file_hash
from logical_magic.ibm_dd import prepare_xy4_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    service = QiskitRuntimeService(instance=INSTANCE)
    backend = service.backend(BACKEND)
    if not backend.status().operational:
        parser.error("ibm_aachen is not operational")
    m = prepare_xy4_bundle(args.source, args.output, backend)
    print(f"Prepared only: {m['number_of_circuits']} circuits, {m['total_shots']} shots")
    print(f"XY4: {m['resources']['dd']['added_native_x_pulses']} physical pi pulses per preparation")
    print(f"Protected windows: {m['resources']['dd']['protected_idle_count']}")
    print(f"Duration unchanged: {m['resources']['duration_seconds']*1e6:.3f} us")
    print(f"Manifest: {args.output.resolve()/'manifest.json'}")
    print(f"Approval SHA256: {file_hash(args.output/'manifest.json')}")
    print("No job submitted.")


if __name__ == "__main__":
    main()
