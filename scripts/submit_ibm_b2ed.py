#!/usr/bin/env python3
"""Submit an explicitly approved bundle once, save the job ID, and exit."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logical_magic.ibm_b2ed import submit_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--approve-manifest-sha256", required=True,
                        help="Exact manifest digest; provide only after human approval")
    args = parser.parse_args()
    submit_bundle(args.bundle, args.approve_manifest_sha256)


if __name__ == "__main__":
    main()
