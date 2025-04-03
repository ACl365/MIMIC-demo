"""
Script to build features from processed data.

This script orchestrates the feature extraction process by calling different
feature extractors (Demographic, Clinical, Diagnosis) based on the configuration.
It then combines the extracted features into a single DataFrame and saves it.
Includes logic for handling missing feature files and performing imputation
on the combined dataset.
"""

import argparse
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # Added TYPE_CHECKING

import numpy as np  # Added import for NumPy
import pandas as pd
from numpy import number as np_number  # Explicit import for type checking

from utils import get_data_path, get_logger, load_config  # Corrected direct imports
from utils.logger import is_debug_enabled  # Import specifically from logger module

from .feature_extractors import (
    ClinicalFeatureExtractor,
    DemographicFeatureExtractor,
    DiagnosisFeatureExtractor,
)

logger = get_logger(__name__)


def build_features(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Run the feature building pipeline based on the provided configuration.

    Instantiates and runs feature extractors specified in the config
    (demographic, clinical, diagnosis). Saves the output of each extractor.
    Finally, combines all generated feature sets into a single 'combined_features.csv'
    file, performing imputation as needed.

    Args:
        config (Optional[Dict[str, Any]], optional): Configuration dictionary.
            If None, loads the default configuration using load_config().
            Defaults to None.
    """
    if config is None:
        config = load_config()

    logger.info("Starting feature building pipeline")

    # Extract demographic features
    if config["features"]["demographic"]["include"]:
        logger.info("Extracting demographic features")
        try:
            demographic_extractor = DemographicFeatureExtractor(config)
            demographic_features = demographic_extractor.extract()
            demographic_output_path = os.path.join(
                get_data_path("processed", "base_path", config),
                "demographic_features.csv",
            )
            demographic_extractor.save(demographic_features, demographic_output_path)
        except Exception as e:
            logger.error(f"Error extracting demographic features: {e}", exc_info=True)

    # Extract clinical features (Vitals and/or Lab Values)
    if (
        config["features"]["vitals"]["include"]
        or config["features"]["lab_values"]["include"]
    ):
        logger.info("Extracting clinical features")
        try:
            clinical_extractor = ClinicalFeatureExtractor(config)
            clinical_features = clinical_extractor.extract()
            clinical_output_path = os.path.join(
                get_data_path("processed", "base_path", config), "clinical_features.csv"
            )
            clinical_extractor.save(clinical_features, clinical_output_path)
        except Exception as e:
            logger.error(f"Error extracting clinical features: {e}", exc_info=True)

    # Extract diagnosis features
    if config["features"]["diagnoses"]["include"]:
        logger.info("Extracting diagnosis features")
        try:
            diagnosis_extractor = DiagnosisFeatureExtractor(config)
            diagnosis_features = diagnosis_extractor.extract()
            diagnosis_output_path = os.path.join(
                get_data_path("processed", "base_path", config),
                "diagnosis_features.csv",
            )
            diagnosis_extractor.save(diagnosis_features, diagnosis_output_path)
        except Exception as e:
            logger.error(f"Error extracting diagnosis features: {e}", exc_info=True)

    # Combine all features
    logger.info("Combining all features")
    try:
        combined_features = _combine_features(config)
        combined_output_path = get_data_path("processed", "combined_features", config)

        # Save combined features
        os.makedirs(
            os.path.dirname(combined_output_path), exist_ok=True
        )  # Ensure dir exists
        combined_features.to_csv(combined_output_path, index=False)
        logger.info(f"Saved combined features to {combined_output_path}")
    except Exception as e:
        logger.error(f"Error combining or saving features: {e}", exc_info=True)

    logger.info("Feature building pipeline completed")


def _combine_features(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and combine individual feature sets into a single DataFrame.

    Loads processed admission data as the base, then iteratively merges
    demographic, clinical, and diagnosis features based on configuration flags
    and file existence. Performs imputation on the final combined DataFrame.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        pd.DataFrame: Combined DataFrame with all selected features and imputed missing values.

    Raises:
        FileNotFoundError: If the base admission data file is not found.
        Exception: For errors during file loading or merging.
    """
    processed_path = get_data_path("processed", "base_path", config)

    # Load admission data for identifiers and targets
    admission_path = get_data_path("processed", "admission_data", config)
    if not os.path.exists(admission_path):
        logger.error(
            f"Base admission data not found at {admission_path}. Cannot combine features."
        )
        raise FileNotFoundError(f"Admission data not found: {admission_path}")
    admissions = pd.read_csv(admission_path)

    # Convert all column names to lowercase for consistency
    admissions.columns = admissions.columns.str.lower()

    # Ensure required columns exist for base DataFrame
    required_columns = [
        "subject_id",
        "hadm_id",
        "los_days",
        "hospital_death",
        "readmission_30day",
        "readmission_90day",
        "days_to_readmission",
    ]

    # Check which required columns exist in the dataframe
    existing_columns = [col for col in required_columns if col in admissions.columns]
    missing_base_cols = set(required_columns) - set(existing_columns)
    if missing_base_cols:
        logger.warning(
            f"Base admission data is missing expected columns: {missing_base_cols}. These will not be included."
        )

    # Select key columns from admissions
    combined_features = admissions[existing_columns].copy()

    # Load and merge demographic features if available and included
    demographic_path = os.path.join(processed_path, "demographic_features.csv")
    if config["features"]["demographic"]["include"] and os.path.exists(
        demographic_path
    ):
        logger.info(f"Loading demographic features from {demographic_path}")
        demographic_features = pd.read_csv(demographic_path)
        demographic_features.columns = demographic_features.columns.str.lower()

        if (
            "subject_id" in demographic_features.columns
            and "hadm_id" in demographic_features.columns
        ):
            # Drop potential duplicate ID columns before merge if they exist beyond the merge keys
            cols_to_drop = list(
                set(combined_features.columns)
                & set(demographic_features.columns) - set(["subject_id", "hadm_id"])
            )
            if cols_to_drop:
                logger.debug(
                    f"Dropping duplicate columns from demographic features before merge: {cols_to_drop}"
                )
                demographic_features = demographic_features.drop(columns=cols_to_drop)

            combined_features = pd.merge(
                combined_features,
                demographic_features,
                on=["subject_id", "hadm_id"],
                how="left",
            )
            logger.info("Merged demographic features.")
        else:
            logger.warning(
                f"Demographic features file at {demographic_path} missing ID columns. Skipping merge."
            )
    elif config["features"]["demographic"]["include"]:
        logger.warning(
            f"Demographic features included in config but file not found at {demographic_path}."
        )

    # Load and merge clinical features if available and included
    clinical_path = os.path.join(processed_path, "clinical_features.csv")
    if (
        config["features"]["vitals"]["include"]
        or config["features"]["lab_values"]["include"]
    ) and os.path.exists(clinical_path):
        logger.info(f"Loading clinical features from {clinical_path}")
        clinical_features = pd.read_csv(clinical_path, low_memory=False)
        clinical_features.columns = clinical_features.columns.str.lower()

        if (
            "subject_id" in clinical_features.columns
            and "hadm_id" in clinical_features.columns
        ):
            # Drop potential duplicate ID/stay columns before merge
            cols_to_drop = list(
                set(combined_features.columns)
                & set(clinical_features.columns) - set(["subject_id", "hadm_id"])
            )
            # Also drop stay_id if merging only on subject/hadm
            if "stay_id" in clinical_features.columns and "stay_id" not in [
                "subject_id",
                "hadm_id",
            ]:
                cols_to_drop.append("stay_id")
            cols_to_drop = list(set(cols_to_drop))  # Unique list

            if cols_to_drop:
                logger.debug(
                    f"Dropping duplicate/unneeded columns from clinical features before merge: {cols_to_drop}"
                )
                clinical_features = clinical_features.drop(columns=cols_to_drop)

            combined_features = pd.merge(
                combined_features,
                clinical_features,
                on=["subject_id", "hadm_id"],  # Merge on admission level
                how="left",
            )
            logger.info("Merged clinical features.")
        else:
            logger.warning(
                f"Clinical features file at {clinical_path} missing ID columns. Skipping merge."
            )
    elif (
        config["features"]["vitals"]["include"]
        or config["features"]["lab_values"]["include"]
    ):
        logger.warning(
            f"Clinical features included in config but file not found at {clinical_path}."
        )

    # Load and merge diagnosis features if available and included
    diagnosis_path = os.path.join(processed_path, "diagnosis_features.csv")
    if config["features"]["diagnoses"]["include"] and os.path.exists(diagnosis_path):
        logger.info(f"Loading diagnosis features from {diagnosis_path}")
        diagnosis_features = pd.read_csv(diagnosis_path)
        diagnosis_features.columns = diagnosis_features.columns.str.lower()

        if (
            "subject_id" in diagnosis_features.columns
            and "hadm_id" in diagnosis_features.columns
        ):
            # Drop potential duplicate ID columns before merge
            cols_to_drop = list(
                set(combined_features.columns)
                & set(diagnosis_features.columns) - set(["subject_id", "hadm_id"])
            )
            if cols_to_drop:
                logger.debug(
                    f"Dropping duplicate columns from diagnosis features before merge: {cols_to_drop}"
                )
                diagnosis_features = diagnosis_features.drop(columns=cols_to_drop)

            combined_features = pd.merge(
                combined_features,
                diagnosis_features,
                on=["subject_id", "hadm_id"],
                how="left",
            )
            logger.info("Merged diagnosis features.")
        else:
            logger.warning(
                f"Diagnosis features file at {diagnosis_path} missing ID columns. Skipping merge."
            )
    elif config["features"]["diagnoses"]["include"]:
        logger.warning(
            f"Diagnosis features included in config but file not found at {diagnosis_path}."
        )

    # Implement a more sophisticated imputation strategy
    logger.info("Implementing feature-specific imputation strategy")

    # Identify different types of features for appropriate imputation
    # 1. Categorical features (one-hot encoded) - fill with 0 (absence)
    # 2. Continuous clinical measurements - fill with median/mean
    # 3. Count-based features - fill with 0
    # 4. Sequence data - should already be handled (filled with empty lists/tuples), but check for NaNs

    # Get column types to determine appropriate imputation
    # Exclude sequence columns from numeric imputation targets
    numeric_cols = [
        col
        for col in combined_features.select_dtypes(include=["number"]).columns
        if not col.endswith("_sequence_data")
    ]
    logger.debug(
        f"Identified {len(numeric_cols)} numeric columns for potential imputation (excluding sequences)."
    )

    # Exclude ID columns and target variables from imputation
    id_cols = ["subject_id", "hadm_id", "stay_id"]
    target_cols = [
        "los_days",
        "hospital_death",
        "readmission_30day",
        "readmission_90day",
        "days_to_readmission",
    ]
    # Also exclude sequence columns explicitly from any imputation logic below if necessary
    sequence_cols = [
        col for col in combined_features.columns if col.endswith("_sequence_data")
    ]

    # Columns that should be imputed with 0 (binary/categorical flags, counts)
    # Includes one-hot encoded columns (often int/float but represent presence/absence)
    # and specific count columns.
    zero_impute_cols = [
        col
        for col in numeric_cols
        if col not in id_cols + target_cols
        and (
            combined_features[col].dropna().isin([0, 1]).all()  # Binary flags
            or col.endswith("_count")  # Count features
            or col.startswith("diag_category_")  # Diagnosis flags
        )
    ]

    # Clinical measurement columns (lab values, vitals aggregates if any) - impute with median
    # This logic might need refinement based on actual column names generated
    clinical_cols = [
        col
        for col in numeric_cols
        if col not in id_cols + target_cols + zero_impute_cols
        and (
            "_mean" in col
            or "_min" in col
            or "_max" in col
            or "_std" in col
            or any(
                lab in col.lower()
                for lab in [
                    "glucose",
                    "potassium",
                    "sodium",
                    "creatinine",
                    "bun",
                    "hemoglobin",
                    "wbc",
                    "platelet",
                    "heart_rate",
                    "bp",
                    "temp",
                    "resp",
                    "spo2",
                    "lactate",
                    "ph",
                    "pco2",
                    "po2",
                ]
            )
        )
    ]

    # Other numeric columns (e.g., age, potentially ED LOS) - impute with median
    other_numeric_cols = [
        col
        for col in numeric_cols
        if col not in id_cols + target_cols + zero_impute_cols + clinical_cols
    ]

    # Log the imputation strategy
    logger.info(
        f"Imputing {len(zero_impute_cols)} binary/categorical/count features with 0"
    )
    logger.info(
        f"Imputing {len(clinical_cols)} clinical measurement features with median"
    )
    logger.info(
        f"Imputing {len(other_numeric_cols)} other numeric features with median"
    )

    # Apply imputation
    # Binary, count, and diagnosis flag features - fill with 0
    for col in zero_impute_cols:
        if col in combined_features.columns:
            combined_features[col] = combined_features[col].fillna(0)

    # Clinical measurements - fill with median
    for col in clinical_cols:
        if col in combined_features.columns:
            median_value = combined_features[col].median()
            combined_features[col] = combined_features[col].fillna(median_value)
            logger.debug(f"Imputed {col} with median: {median_value}")

    # Other numeric features - fill with median
    for col in other_numeric_cols:
        if col in combined_features.columns:
            median_value = combined_features[col].median()
            combined_features[col] = combined_features[col].fillna(median_value)
            logger.debug(f"Imputed {col} with median: {median_value}")

    # Sequence columns - fill potential NaNs with empty list representation
    for col in sequence_cols:
        if col in combined_features.columns and combined_features[col].isnull().any():
            logger.warning(
                f"Found unexpected NaNs in sequence column {col}. Filling with empty list."
            )
            # Assuming sequence columns store lists, fillna with an empty list object
            # This might require adjustment based on how NaNs appear (e.g., actual np.nan vs. None)
            # A robust way is to apply a function that returns [] if the value is NaN/None
            combined_features[col] = combined_features[col].apply(
                lambda x: x if isinstance(x, list) else []
            )

    # Any remaining NaN values in non-numeric, non-sequence columns - fill with 'Unknown'
    remaining_na_cols = (
        combined_features.select_dtypes(exclude=np_number)
        .columns[combined_features.select_dtypes(exclude=np_number).isna().any()]
        .tolist()
    )
    # Exclude sequence columns again just in case they weren't caught as numeric/object
    remaining_na_cols = [col for col in remaining_na_cols if col not in sequence_cols]

    if remaining_na_cols:
        logger.info(
            f"Filling {len(remaining_na_cols)} remaining non-numeric columns with 'Unknown'"
        )
        for col in remaining_na_cols:
            combined_features[col] = combined_features[col].fillna("Unknown")

    # Final check for NaNs
    if combined_features.isnull().any().any():
        na_cols_final = combined_features.columns[
            combined_features.isnull().any()
        ].tolist()
        logger.warning(
            f"NaN values still remain after imputation in columns: {na_cols_final}"
        )
        # Consider more robust imputation or error handling here if needed

    return combined_features


def main() -> None:
    """
    Main execution function for the script.

    Parses command-line arguments (optional config path) and runs the
    feature building pipeline.
    """
    parser = argparse.ArgumentParser(description="Build features from processed data")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = load_config()

    # Build features
    build_features(config)


if __name__ == "__main__":
    main()
