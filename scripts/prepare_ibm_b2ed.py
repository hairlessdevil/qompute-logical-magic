#!/usr/bin/env python3
"""Read live calibration and prepare a review bundle; never submit a job."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logical_magic.ibm_b2ed import BACKEND, INSTANCE, QiskitRuntimeService, file_hash, prepare_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--code", choices=("four", "five"), default="four")
    args = parser.parse_args()
    service = QiskitRuntimeService(instance=INSTANCE)
    backend = service.backend(BACKEND)
    if not backend.status().operational:
        parser.error("ibm_aachen is not operational")
    manifest = prepare_bundle(backend, args.output, args.shots, code_name=args.code)
    print(f"Prepared only: {manifest['number_of_circuits']} circuits, {manifest['total_shots']} shots")
    print(f"Estimated QPU usage: {manifest['estimated_usage_seconds']:.1f} seconds (approximate)")
    print(f"Manifest: {args.output.resolve() / 'manifest.json'}")
    print(f"Approval SHA256: {file_hash(args.output / 'manifest.json')}")
    print("No job submitted.")


if __name__ == "__main__":
    main()
