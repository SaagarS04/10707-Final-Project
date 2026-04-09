"""
MCMC promotion gate.

Runs the toy finite-state verification suite in mcmc/verify.py.
MCMC must pass this gate before it is used in any evaluation.

Usage:
    python scripts/verify_mcmc.py
    python scripts/verify_mcmc.py --seed 123
    python scripts/verify_mcmc.py --quiet
"""

import argparse
import sys

from mcmc.verify import run_verification


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MCMC toy verification (KL test at λ=0 and λ>0)."
    )
    parser.add_argument("--seed",  type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--quiet", action="store_true",  help="Suppress per-test output")
    args = parser.parse_args()

    try:
        results = run_verification(seed=args.seed, verbose=not args.quiet)
    except AssertionError as e:
        print(f"\n[verify_mcmc] FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    if args.quiet:
        for key, val in results.items():
            status = "PASS" if val["passed"] else "FAIL"
            print(f"{key}: KL={val['kl']:.5f} [{status}]")

    print("\n[verify_mcmc] MCMC promotion gate: PASSED")


if __name__ == "__main__":
    main()
