"""
Script to run the data processing pipeline.
"""

import argparse
import os
from typing import Dict, List, Optional

import pandas as pd

from ..utils import get_logger, load_config, get_data_path
from .processors import (
    PatientProcessor,
    AdmissionProcessor,
    ICUStayProcessor,
)


logger = get_logger(__name__)


def process_data(config: Optional[Dict] = None) -> None:
    """
    Process the MIMIC data.
    
    Args:
        config (Optional[Dict], optional): Configuration dictionary.
            If None, loads the default configuration. Defaults to None.
    """
    if config is None:
        config = load_config()
    
    logger.info("Starting data processing pipeline")
    
    # Process patient data
    patient_processor = PatientProcessor(config)
    patient_data = patient_processor.process()
    patient_output_path = get_data_path("processed", "patient_data", config)
    patient_processor.save(patient_data, patient_output_path)
    
    # Process admission data
    admission_processor = AdmissionProcessor(config)
    admission_data = admission_processor.process()
    admission_output_path = get_data_path("processed", "admission_data", config)
    admission_processor.save(admission_data, admission_output_path)
    
    # Process ICU stay data
    icustay_processor = ICUStayProcessor(config)
    icustay_data = icustay_processor.process()
    icustay_output_path = get_data_path("processed", "icu_data", config)
    icustay_processor.save(icustay_data, icustay_output_path)
    
    logger.info("Data processing pipeline completed")


def main() -> None:
    """
    Main function to run the data processing pipeline.
    """
    parser = argparse.ArgumentParser(description="Process MIMIC data")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="Path to configuration file"
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