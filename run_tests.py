#!/usr/bin/env python
"""
Script to run the unit tests for the MIMIC demo project.
"""

import argparse
import os
import subprocess
import sys


def run_tests(args):
    """
    Run the unit tests.

    Args:
        args: Command-line arguments

    Returns:
        int: Exit code
    """
    # Construct the pytest command
    cmd = ["pytest"]

    # Add coverage options if requested
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term", "--cov-report=html"])

    # Add verbose option if requested
    if args.verbose:
        cmd.append("-v")

    # Add specific test path if provided
    if args.path:
        cmd.append(args.path)

    # Run the tests
    result = subprocess.run(cmd)

    return result.returncode


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(
        description="Run unit tests for the MIMIC demo project"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "path", nargs="?", help="Path to specific test file or directory"
    )

    args = parser.parse_args()

    # Run the tests
    exit_code = run_tests(args)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
