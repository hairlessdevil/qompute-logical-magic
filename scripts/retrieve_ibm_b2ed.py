#!/usr/bin/env python3
"""Run later: check once, retrieve if DONE, and analyze recorded-zero shots."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logical_magic.ibm_b2ed import retrieve_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    result = retrieve_bundle(args.bundle)
    print(json.dumps({k: result[k] for k in
                     ("job_id", "status", "raw_fidelity_estimate", "standard_error") if k in result}, indent=2))


if __name__ == "__main__":
    main()
