"""
Script to run the data processing pipeline.

This script orchestrates the initial processing of raw MIMIC data by calling
the relevant processors defined in `src.data.processors`. It processes patient,
admission, and ICU stay data sequentially, saving the output of each step to
the configured processed data directory.
"""

import argparse
import os
from typing import Any, Dict, List, Optional  # Added Any

import pandas as pd

from utils import get_data_path, get_logger, load_config  # Corrected direct import

from .processors import AdmissionProcessor, ICUStayProcessor, PatientProcessor

logger = get_logger(__name__)


def process_data(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Run the main data processing pipeline.

    Instantiates and runs the PatientProcessor, AdmissionProcessor, and
    ICUStayProcessor in sequence. Each processor loads raw data, performs
    cleaning and transformation, and saves the result to the processed data directory
    as specified in the configuration.

    Args:
        config (Optional[Dict[str, Any]], optional): Configuration dictionary.
            If None, loads the default configuration using `load_config()`.
            Defaults to None.
    """
    if config is None:
        config = load_config()

    logger.info("Starting data processing pipeline")

    # Process patient data
    try:
        logger.info("Processing patient data...")
        patient_processor = PatientProcessor(config)
        patient_data = patient_processor.process()
        patient_output_path = get_data_path("processed", "patient_data", config)
        patient_processor.save(patient_data, patient_output_path)
        logger.info("Patient data processing complete.")
    except Exception as e:
        logger.error(f"Error processing patient data: {e}", exc_info=True)

    # Process admission data
    try:
        logger.info("Processing admission data...")
        admission_processor = AdmissionProcessor(config)
        admission_data = admission_processor.process()
        admission_output_path = get_data_path("processed", "admission_data", config)
        admission_processor.save(admission_data, admission_output_path)
        logger.info("Admission data processing complete.")
    except Exception as e:
        logger.error(f"Error processing admission data: {e}", exc_info=True)

    # Process ICU stay data
    try:
        logger.info("Processing ICU stay data...")
        icustay_processor = ICUStayProcessor(config)
        icustay_data = icustay_processor.process()
        icustay_output_path = get_data_path("processed", "icu_data", config)
        icustay_processor.save(icustay_data, icustay_output_path)
        logger.info("ICU stay data processing complete.")
    except Exception as e:
        logger.error(f"Error processing ICU stay data: {e}", exc_info=True)

    logger.info("Data processing pipeline completed")


def main() -> None:
    """
    Main execution function for the data processing script.

    Parses command-line arguments for an optional configuration file path,
    loads the configuration, and runs the `process_data` function.
    """
    parser = argparse.ArgumentParser(description="Process MIMIC data")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = load_config()

    # Process data
    process_data(config)


if __name__ == "__main__":
    main()
