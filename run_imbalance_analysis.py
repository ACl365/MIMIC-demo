#!/usr/bin/env python
"""
Script to run the imbalance analysis for the readmission model.

This script provides a command-line interface to run the imbalance analysis,
which compares different techniques for handling class imbalance in the
readmission prediction model.

Example usage:
    python run_imbalance_analysis.py --output-dir results/imbalance_analysis --cv-folds 5
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.imbalance_analysis import analyze_imbalance_techniques
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """
    Main function to run the imbalance analysis.
    """
    parser = argparse.ArgumentParser(
        description="Run imbalance analysis for the readmission model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: results/imbalance_analysis)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    logger.info("Starting imbalance analysis for readmission model")
    logger.info(f"Cross-validation folds: {args.cv_folds}")
    logger.info(f"Random state: {args.random_state}")

    # Run analysis
    results = analyze_imbalance_techniques(
        output_dir=args.output_dir,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )

    logger.info("Imbalance analysis completed successfully")
    logger.info(f"Results saved to {args.output_dir or 'results/imbalance_analysis'}")


if __name__ == "__main__":
    main()
