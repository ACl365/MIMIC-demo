"""
Script to build features from processed data.
"""

import argparse
import os
from typing import Dict, List, Optional

import pandas as pd

from ..utils import get_logger, load_config, get_data_path
from .feature_extractors import (
    DemographicFeatureExtractor,
    ClinicalFeatureExtractor,
    DiagnosisFeatureExtractor,
)


logger = get_logger(__name__)


def build_features(config: Optional[Dict] = None) -> None:
    """
    Build features from processed data.
    
    Args:
        config (Optional[Dict], optional): Configuration dictionary.
            If None, loads the default configuration. Defaults to None.
    """
    if config is None:
        config = load_config()
    
    logger.info("Starting feature building pipeline")
    
    # Extract demographic features
    if config["features"]["demographic"]["include"]:
        logger.info("Extracting demographic features")
        demographic_extractor = DemographicFeatureExtractor(config)
        demographic_features = demographic_extractor.extract()
        demographic_output_path = os.path.join(
            get_data_path("processed", "base_path", config),
            "demographic_features.csv"
        )
        demographic_extractor.save(demographic_features, demographic_output_path)
    
    # Extract clinical features
    if config["features"]["vitals"]["include"] or config["features"]["lab_values"]["include"]:
        logger.info("Extracting clinical features")
        clinical_extractor = ClinicalFeatureExtractor(config)
        clinical_features = clinical_extractor.extract()
        clinical_output_path = os.path.join(
            get_data_path("processed", "base_path", config),
            "clinical_features.csv"
        )
        clinical_extractor.save(clinical_features, clinical_output_path)
    
    # Extract diagnosis features
    if config["features"]["diagnoses"]["include"]:
        logger.info("Extracting diagnosis features")
        diagnosis_extractor = DiagnosisFeatureExtractor(config)
        diagnosis_features = diagnosis_extractor.extract()
        diagnosis_output_path = os.path.join(
            get_data_path("processed", "base_path", config),
            "diagnosis_features.csv"
        )
        diagnosis_extractor.save(diagnosis_features, diagnosis_output_path)
    
    # Combine all features
    logger.info("Combining all features")
    combined_features = _combine_features(config)
    combined_output_path = get_data_path("processed", "combined_features", config)
    
    # Save combined features
    combined_features.to_csv(combined_output_path, index=False)
    logger.info(f"Saved combined features to {combined_output_path}")
    
    logger.info("Feature building pipeline completed")


def _combine_features(config: Dict) -> pd.DataFrame:
    """
    Combine all extracted features.
    
    Args:
        config (Dict): Configuration dictionary
    
    Returns:
        pd.DataFrame: Combined features
    """
    processed_path = get_data_path("processed", "base_path", config)
    
    # Load admission data for identifiers and targets
    admission_path = get_data_path("processed", "admission_data", config)
    admissions = pd.read_csv(admission_path)
    
    # Convert all column names to lowercase for consistency
    admissions.columns = admissions.columns.str.lower()
    
    # Ensure required columns exist
    required_columns = ["subject_id", "hadm_id", "los_days", "hospital_death",
                        "readmission_30day", "readmission_90day"]
    
    # Check which required columns exist in the dataframe
    existing_columns = [col for col in required_columns if col in admissions.columns]
    
    # Select key columns from admissions
    combined_features = admissions[existing_columns].copy()
    
    # Load and merge demographic features if available
    demographic_path = os.path.join(processed_path, "demographic_features.csv")
    if os.path.exists(demographic_path) and config["features"]["demographic"]["include"]:
        demographic_features = pd.read_csv(demographic_path)
        
        # Convert all column names to lowercase for consistency
        demographic_features.columns = demographic_features.columns.str.lower()
        
        from src.utils import is_debug_enabled
        
        # Only log detailed debug information if debug logging is enabled
        if is_debug_enabled():
            logger.debug(f"Demographic features columns: {demographic_features.columns.tolist()}")
            logger.debug(f"Demographic features shape: {demographic_features.shape}")
            logger.debug(f"Demographic features head: \n{demographic_features.head()}")
        else:
            logger.info("Processing demographic features")
        
        # Check if the dataframe has the required columns
        if len(demographic_features) > 0 and "subject_id" in demographic_features.columns and "hadm_id" in demographic_features.columns:
            # Make a copy of the dataframe to avoid modifying the original
            df_to_merge = demographic_features.copy()
            
            # Only log detailed debug information if debug logging is enabled
            if is_debug_enabled():
                logger.debug(f"df_to_merge columns before drop: {df_to_merge.columns.tolist()}")
                logger.debug(f"df_to_merge columns (keeping IDs): {df_to_merge.columns.tolist()}")
                logger.debug(f"Combined features columns before merge: {combined_features.columns.tolist()}")
                logger.debug(f"Combined features shape before merge: {combined_features.shape}")
            
            combined_features = pd.merge(
                combined_features,
                df_to_merge,
                on=["subject_id", "hadm_id"],
                how="left"
            )
            
            if is_debug_enabled():
                logger.debug(f"Combined features columns after merge: {combined_features.columns.tolist()}")
                logger.debug(f"Combined features shape after merge: {combined_features.shape}")
            else:
                logger.info("Merged demographic features successfully")
        else:
            logger.warning("Missing required columns in demographic_features")
            if len(demographic_features) > 0:
                logger.warning(f"Available columns: {demographic_features.columns.tolist()}")
    
    # Load and merge clinical features if available
    clinical_path = os.path.join(processed_path, "clinical_features.csv")
    if os.path.exists(clinical_path) and (
        config["features"]["vitals"]["include"] or config["features"]["lab_values"]["include"]
    ):
        # Log appropriate information based on debug setting
        if is_debug_enabled():
            logger.debug(f"Loading clinical features from {clinical_path}")
        else:
            logger.info("Processing clinical features")
        
        # Load the clinical features with explicit low_memory=False to avoid mixed type inference
        clinical_features = pd.read_csv(clinical_path, low_memory=False)
        
        # Only log detailed debug information if debug logging is enabled
        if is_debug_enabled():
            logger.debug(f"Clinical features columns: {clinical_features.columns.tolist()}")
        
        # Convert all column names to lowercase for consistency
        clinical_features.columns = clinical_features.columns.str.lower()
        
        # Only log detailed debug information if debug logging is enabled
        if is_debug_enabled():
            logger.debug(f"Clinical features columns after lowercase conversion: {clinical_features.columns.tolist()}")
        
        # Check if subject_id exists in the dataframe
        if "subject_id" in clinical_features.columns:
            if is_debug_enabled():
                logger.debug("subject_id column found in clinical_features")
                logger.debug(f"Clinical features columns before merge: {clinical_features.columns.tolist()}")
                logger.debug(f"Clinical features shape: {clinical_features.shape}")
                logger.debug(f"Combined features columns before clinical merge: {combined_features.columns.tolist()}")
                logger.debug(f"Combined features shape before clinical merge: {combined_features.shape}")
            
            # Merge with the combined features
            combined_features = pd.merge(
                combined_features,
                clinical_features,
                on=["subject_id", "hadm_id"],
                how="left"
            )
            
            if is_debug_enabled():
                logger.debug(f"Combined features columns after clinical merge: {combined_features.columns.tolist()}")
                logger.debug(f"Combined features shape after clinical merge: {combined_features.shape}")
            else:
                logger.info("Merged clinical features successfully")
        else:
            logger.warning("subject_id column not found in clinical_features")
            logger.warning(f"Available columns: {clinical_features.columns.tolist()}")
    
    # Load and merge diagnosis features if available
    diagnosis_path = os.path.join(processed_path, "diagnosis_features.csv")
    if os.path.exists(diagnosis_path) and config["features"]["diagnoses"]["include"]:
        if is_debug_enabled():
            logger.debug(f"Loading diagnosis features from {diagnosis_path}")
        else:
            logger.info("Processing diagnosis features")
            
        diagnosis_features = pd.read_csv(diagnosis_path)
        
        # Convert all column names to lowercase for consistency
        diagnosis_features.columns = diagnosis_features.columns.str.lower()
        
        # Check if the dataframe has the required columns
        if len(diagnosis_features) > 0 and "subject_id" in diagnosis_features.columns and "hadm_id" in diagnosis_features.columns:
            # Make a copy of the dataframe to avoid modifying the original
            df_to_merge = diagnosis_features.copy()
            
            # Only log detailed debug information if debug logging is enabled
            if is_debug_enabled():
                logger.debug(f"Diagnosis df_to_merge columns (keeping IDs): {df_to_merge.columns.tolist()}")
                logger.debug(f"Combined features columns before diagnosis merge: {combined_features.columns.tolist()}")
                logger.debug(f"Combined features shape before diagnosis merge: {combined_features.shape}")
            
            # Merge with the combined features
            combined_features = pd.merge(
                combined_features,
                df_to_merge,
                on=["subject_id", "hadm_id"],
                how="left"
            )
            
            if is_debug_enabled():
                logger.debug(f"Combined features columns after diagnosis merge: {combined_features.columns.tolist()}")
                logger.debug(f"Combined features shape after diagnosis merge: {combined_features.shape}")
            else:
                logger.info("Merged diagnosis features successfully")
    
    # Implement a more sophisticated imputation strategy
    logger.info("Implementing feature-specific imputation strategy")
    
    # Identify different types of features for appropriate imputation
    # 1. Categorical features (one-hot encoded) - fill with 0 (absence)
    # 2. Continuous clinical measurements - fill with median/mean
    # 3. Count-based features - fill with 0
    
    # Get column types to determine appropriate imputation
    numeric_cols = combined_features.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude ID columns and target variables from imputation
    id_cols = ['subject_id', 'hadm_id', 'stay_id']
    target_cols = ['los_days', 'hospital_death', 'readmission_30day', 'readmission_90day', 'days_to_readmission']
    
    # Columns that should be imputed with 0 (binary/categorical features)
    binary_cols = [col for col in numeric_cols if combined_features[col].dropna().isin([0, 1]).all()]
    
    # Columns that are likely counts (should be imputed with 0)
    count_cols = [col for col in numeric_cols if
                 (col.endswith('_count') or
                  col.startswith('count_') or
                  'count' in col.lower()) and
                 col not in id_cols + target_cols]
    
    # Clinical measurement columns (lab values, vitals) - impute with median
    clinical_cols = [col for col in numeric_cols if
                    (col not in id_cols + target_cols + binary_cols + count_cols) and
                    ('_mean' in col or '_min' in col or '_max' in col or '_std' in col or
                     any(lab in col.lower() for lab in ['glucose', 'potassium', 'sodium', 'creatinine',
                                                       'bun', 'hemoglobin', 'wbc', 'platelet',
                                                       'heart_rate', 'bp', 'temp', 'resp']))]
    
    # Other numeric columns - impute with median
    other_numeric_cols = [col for col in numeric_cols if
                         col not in id_cols + target_cols + binary_cols + count_cols + clinical_cols]
    
    # Log the imputation strategy
    logger.info(f"Imputing {len(binary_cols)} binary/categorical features with 0")
    logger.info(f"Imputing {len(count_cols)} count-based features with 0")
    logger.info(f"Imputing {len(clinical_cols)} clinical measurement features with median")
    logger.info(f"Imputing {len(other_numeric_cols)} other numeric features with median")
    
    # Apply imputation
    # Binary and count features - fill with 0
    for col in binary_cols + count_cols:
        combined_features[col] = combined_features[col].fillna(0)
    
    # Clinical measurements - fill with median
    for col in clinical_cols:
        median_value = combined_features[col].median()
        combined_features[col] = combined_features[col].fillna(median_value)
        logger.debug(f"Imputed {col} with median: {median_value}")
    
    # Other numeric features - fill with median
    for col in other_numeric_cols:
        median_value = combined_features[col].median()
        combined_features[col] = combined_features[col].fillna(median_value)
    
    # Any remaining NaN values (non-numeric columns) - fill with 'Unknown'
    remaining_na_cols = combined_features.columns[combined_features.isna().any()].tolist()
    if remaining_na_cols:
        logger.info(f"Filling {len(remaining_na_cols)} remaining columns with appropriate values")
        for col in remaining_na_cols:
            if col not in numeric_cols:
                combined_features[col] = combined_features[col].fillna('Unknown')
    
    return combined_features


def main() -> None:
    """
    Main function to run the feature building pipeline.
    """
    parser = argparse.ArgumentParser(description="Build features from processed data")
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
    
    # Build features
    build_features(config)


if __name__ == "__main__":
    main()