"""
Feature extractors for the MIMIC datasets.

This module defines classes responsible for extracting different types of features
(demographic, clinical, diagnosis) from processed MIMIC data. Each extractor
inherits from a base class and implements an `extract` method. The clinical
extractor specifically generates time-series sequence data for temporal modeling.
"""

import logging  # Import logging for type hint
import os
import re  # Import re globally
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union  # Added Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import get_data_path, get_logger, load_config  # Corrected direct import
from utils.config import load_mappings  # Import load_mappings specifically

logger = get_logger(__name__)


class BaseFeatureExtractor(ABC):
    """
    Abstract Base Class for all feature extractors.

    Provides common initialization (loading config, logger) and a `save` method.
    Subclasses must implement the `extract` method.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the feature extractor.

        Args:
            config (Optional[Dict[str, Any]], optional): Configuration dictionary.
                If None, loads the default configuration using `load_config()`.
                Defaults to None.
        """
        self.config = config if config is not None else load_config()
        self.logger = logger

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """
        Abstract method to extract features.

        Must be implemented by subclasses to perform the specific feature
        extraction logic.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted features, typically
                          indexed or containing identifier columns like 'subject_id',
                          'hadm_id', and potentially 'stay_id'.
        """
        pass

    def save(self, data: pd.DataFrame, output_path: str) -> None:
        """
        Save the extracted features DataFrame to a CSV file.

        Creates the output directory if it doesn't exist.

        Args:
            data (pd.DataFrame): The DataFrame containing extracted features to save.
            output_path (str): The full path (including filename) to save the data to.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save data
        data.to_csv(output_path, index=False)
        self.logger.info(f"Saved extracted features to {output_path}")


class DemographicFeatureExtractor(BaseFeatureExtractor):
    """
    Extractor for demographic and admission-related categorical features.

    Loads processed patient and admission data, merges them, and extracts features like
    age, age group bins, and one-hot encoded representations of gender, admission type,
    insurance, marital status, and ethnicity/race.
    """

    def extract(self) -> pd.DataFrame:
        """
        Loads processed patient and admission data, merges them, and extracts demographic features.

        Returns:
            pd.DataFrame: A DataFrame containing demographic features, indexed by admission,
                          including 'subject_id', 'hadm_id', 'age', and one-hot encoded columns.
        """
        self.logger.info("Extracting demographic features")

        # Load patient data
        patient_path = get_data_path("processed", "patient_data", self.config)
        patients = pd.read_csv(patient_path, parse_dates=["dod"])

        # Load admission data
        admission_path = get_data_path("processed", "admission_data", self.config)
        # The column names are already lowercase in the processed data
        admissions = pd.read_csv(admission_path)

        # Convert date columns to datetime if they exist
        date_columns = ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]
        for col in date_columns:
            if col in admissions.columns:
                admissions[col] = pd.to_datetime(admissions[col], errors="coerce")

        # Merge patient and admission data
        data = pd.merge(
            admissions,
            patients,
            on=[
                "subject_id",
                "source",
            ],  # Assuming source helps differentiate if IDs overlap
            how="left",
            suffixes=("", "_patient"),  # Avoids duplicate columns like 'dod'
        )

        # Extract features
        features = self._extract_demographic_features(data)

        return features

    def _extract_demographic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to extract specific demographic features from merged data.

        Creates age group bins and performs one-hot encoding for specified categorical features.

        Args:
            data (pd.DataFrame): Merged patient and admission data.

        Returns:
            pd.DataFrame: DataFrame containing extracted demographic features.
        """
        # Initialize features dataframe with identifiers
        features = data[["subject_id", "hadm_id"]].copy()

        # Add basic numeric demographics directly to features
        if "age" in data.columns:
            features["age"] = data["age"]
        else:
            logger.warning("Age column not found in data.")

        # --- Pre-calculate categorical columns needed for dummy creation ON THE 'data' DF ---
        # Age bins
        if "age" in data.columns:
            try:
                age_bins = self.config["features"]["demographic"]["age_bins"]
                age_labels = [
                    f"{age_bins[i]}-{age_bins[i+1]}" for i in range(len(age_bins) - 1)
                ]
                # Add 'age_group' column to the original 'data' DataFrame
                data["age_group"] = pd.cut(
                    data["age"], bins=age_bins, labels=age_labels, right=False
                )
            except KeyError:
                logger.error("age_bins not found in config['features']['demographic']")
            except Exception as e:
                logger.error(f"Error creating age_group: {e}")
        else:
            logger.warning("Age column not found in data, cannot create age_group.")

        # --- One-hot encode categorical variables ---
        # Define the list of categorical features to potentially encode
        categorical_features_to_encode = [
            "gender",
            "age_group",
            "admission_type",
            "insurance",
            "marital_status",
        ]

        for feature_name in categorical_features_to_encode:
            if feature_name in data.columns:
                # Create dummy variables from the 'data' DataFrame
                logger.debug(f"Creating dummies for: {feature_name}")
                dummies = pd.get_dummies(
                    data[feature_name], prefix=feature_name, dummy_na=True, dtype=int
                )  # Ensure dtype is int
                # Add to the final 'features' DataFrame
                features = pd.concat([features, dummies], axis=1)
            else:
                logger.warning(
                    f"Categorical feature '{feature_name}' not found in data, skipping dummy creation."
                )

        # Add ethnicity/race features if available
        if "ethnicity" in data.columns:
            logger.debug("Creating dummies for: ethnicity")
            ethnicity_dummies = pd.get_dummies(
                data["ethnicity"], prefix="ethnicity", dummy_na=True, dtype=int
            )
            features = pd.concat([features, ethnicity_dummies], axis=1)
        elif "race" in data.columns:
            logger.debug("Creating dummies for: race")
            race_dummies = pd.get_dummies(
                data["race"], prefix="race", dummy_na=True, dtype=int
            )
            features = pd.concat([features, race_dummies], axis=1)
        else:
            logger.warning("Neither 'ethnicity' nor 'race' column found in data.")

        logger.info(
            f"Finished extracting demographic features. Shape: {features.shape}"
        )
        return features


class ClinicalFeatureExtractor(BaseFeatureExtractor):
    """
    Extractor for clinical time-series features from lab values and vital signs.

    Loads relevant MIMIC-III tables (LABEVENTS, D_LABITEMS, CHARTEVENTS, D_ITEMS),
    filters data based on mappings and time windows defined in the configuration,
    and generates sequences of (time_since_admission, value) tuples for each
    lab test and vital sign per ICU stay.
    """

    def extract(self) -> pd.DataFrame:
        """
        Extracts and combines clinical features (lab and vital sequences).

        Returns:
            pd.DataFrame: A DataFrame where each row represents an ICU stay,
                          containing 'subject_id', 'hadm_id', 'stay_id', and columns
                          for each extracted lab/vital sequence (e.g.,
                          'lab_Glucose_sequence_data', 'vital_heart_rate_sequence_data').
                          Sequence columns contain lists of (time, value) tuples.
        """
        self.logger.info("Extracting clinical features")

        # Load admission data (needed for linking potentially)
        admission_path = get_data_path("processed", "admission_data", self.config)
        admissions = pd.read_csv(
            admission_path,
            parse_dates=[
                "admittime",
                "dischtime",
                "deathtime",
                "edregtime",
                "edouttime",
            ],
        )

        # Load ICU stay data (base for clinical features)
        icu_path = get_data_path("processed", "icu_data", self.config)
        icustays = pd.read_csv(icu_path, parse_dates=["intime", "outtime"])

        # Extract lab features
        lab_features = self._extract_lab_features(admissions, icustays)

        # Extract vital sign features
        vital_features = self._extract_vital_features(admissions, icustays)

        # Combine features using an outer merge to keep all stays
        # Start with a base of all stays to ensure none are dropped if one feature type is missing
        base_ids = icustays[["subject_id", "hadm_id", "icustay_id"]].drop_duplicates()

        features = base_ids
        if not lab_features.empty:
            features = pd.merge(
                features,
                lab_features,
                on=["subject_id", "hadm_id", "icustay_id"],
                how="left",
            )
        else:
            logger.warning(
                "Lab features DataFrame is empty. No lab features will be added."
            )

        if not vital_features.empty:
            features = pd.merge(
                features,
                vital_features,
                on=["subject_id", "hadm_id", "icustay_id"],
                how="left",
            )
        else:
            logger.warning(
                "Vital features DataFrame is empty. No vital features will be added."
            )

        # Fill missing sequence columns (for stays with no labs/vitals) with empty lists representation if needed
        # Note: The sequence processing methods should ideally return empty lists, but fillna ensures it.
        for col in features.columns:
            if col.endswith("_sequence_data"):
                # Check if column dtype is object (likely contains lists or NaN)
                if features[col].dtype == "object":
                    # Fill NaN with empty list representation (adjust if needed based on actual NaN type)
                    features[col] = features[col].apply(
                        lambda x: x if isinstance(x, list) else []
                    )

        return features

    def _extract_lab_features(
        self, admissions: pd.DataFrame, icustays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Loads and processes MIMIC-III lab events (LABEVENTS.csv, D_LABITEMS.csv).

        Args:
            admissions (pd.DataFrame): Processed admission data (potentially unused here but passed for consistency).
            icustays (pd.DataFrame): Processed ICU stay data with 'intime', 'outtime'.

        Returns:
            pd.DataFrame: DataFrame containing lab sequences per ICU stay, or an empty
                          DataFrame with ID columns if loading/processing fails.
        """
        self.logger.info("Extracting laboratory features")
        # Start with ICU stay IDs as the base
        lab_features_final = (
            icustays[["subject_id", "hadm_id", "icustay_id"]].copy().drop_duplicates()
        )

        # Load MIMIC-III lab data
        try:
            mimic3_path = get_data_path("raw", "mimic_iii", self.config)
            lab_path = os.path.join(mimic3_path, "LABEVENTS.csv")
            lab_items_path = os.path.join(mimic3_path, "D_LABITEMS.csv")

            if os.path.exists(lab_path) and os.path.exists(lab_items_path):
                # Load lab data - use lowercase column names for parse_dates
                labs = pd.read_csv(
                    lab_path, parse_dates=["charttime"], low_memory=False
                )
                labs.columns = labs.columns.str.lower()

                # Load lab items dictionary
                lab_items = pd.read_csv(lab_items_path)
                lab_items.columns = lab_items.columns.str.lower()

                # Process lab data
                processed_labs = self._process_lab_data(
                    labs, lab_items, admissions, icustays
                )

                # Merge processed labs back onto the base df with all stays
                if not processed_labs.empty:
                    # Use left merge to keep all stays, fill missing sequences later if needed
                    lab_features_final = pd.merge(
                        lab_features_final,
                        processed_labs,
                        on=["subject_id", "hadm_id", "icustay_id"],
                        how="left",
                    )
                else:
                    self.logger.warning(
                        "Processed labs DataFrame is empty after _process_lab_data."
                    )

            else:
                self.logger.warning(
                    f"MIMIC-III lab data or item dictionary not found (checked {lab_path}, {lab_items_path})"
                )

        except Exception as e:
            self.logger.error(
                f"Error loading or processing MIMIC-III lab data: {e}", exc_info=True
            )
            # Return the base df with IDs only if loading/processing fails

        return lab_features_final

    def _process_lab_data(
        self,
        labs: pd.DataFrame,
        lab_items: pd.DataFrame,
        admissions: pd.DataFrame,  # Currently unused, kept for potential future use
        icustays: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Process laboratory data: filter, standardize, calculate relative time, and create sequences.

        Filters lab events based on mappings and time window, converts values to numeric,
        calculates time relative to ICU admission, groups by stay and lab type,
        and pivots to create columns containing lists of (time, value) tuples.

        Args:
            labs (pd.DataFrame): Raw LABEVENTS data.
            lab_items (pd.DataFrame): D_LABITEMS data.
            admissions (pd.DataFrame): Admission data (unused).
            icustays (pd.DataFrame): ICU stay data with 'intime', 'outtime', 'stay_id'.

        Returns:
            pd.DataFrame: DataFrame with columns 'subject_id', 'hadm_id', 'stay_id',
                          and 'lab_FEATURE_sequence_data' for each processed lab test.
                          Returns an empty DataFrame with ID columns on error.
        """
        try:
            # Import load_mappings specifically from config module
            from utils.config import load_mappings

            self.logger.info("Processing lab data with optimised vectorized operations")

            # Merge lab data with lab items to get labels
            labs = pd.merge(
                labs, lab_items[["itemid", "label"]], on="itemid", how="left"
            )

            # Load lab test mappings from configuration
            try:
                # Cache the mappings to avoid loading them repeatedly
                if not hasattr(self, "_lab_mappings"):
                    mappings = load_mappings()
                    self._lab_mappings = mappings.get("lab_tests", {})

                # Get common labs from mappings
                common_labs = self._lab_mappings.get("common_labs", [])

                # Get lab name variations from mappings
                lab_name_mapping = self._lab_mappings.get("lab_name_variations", {})

                if not common_labs or not lab_name_mapping:
                    self.logger.warning(
                        "Lab mappings ('common_labs' or 'lab_name_variations') are empty or missing in mappings file."
                    )
                    # Decide fallback: either return empty or use hardcoded defaults
                    # Using hardcoded defaults for now as per original logic
                    raise ValueError("Missing lab mappings")  # Trigger fallback

                self.logger.info(
                    f"Loaded {len(common_labs)} common lab tests and {len(lab_name_mapping)} lab name mappings from configuration"
                )
            except Exception as e:
                self.logger.warning(
                    f"Error loading lab mappings from configuration: {e}"
                )
                self.logger.warning("Falling back to hardcoded lab test lists")

                # Fallback to hardcoded lists if mappings fail
                common_labs = [
                    "Glucose",
                    "Potassium",
                    "Sodium",
                    "Chloride",
                    "Creatinine",
                    "BUN",
                    "Bicarbonate",
                    "Anion Gap",
                    "Hemoglobin",
                    "Hematocrit",
                    "WBC",
                    "Platelet Count",
                    "Magnesium",
                    "Calcium",
                    "Phosphate",
                    "Lactate",
                    "pH",
                    "pO2",
                    "pCO2",
                    "Base Excess",
                    "Albumin",
                    "ALT",
                    "AST",
                    "Alkaline Phosphatase",
                    "Bilirubin",
                    "Troponin",
                ]
                lab_name_mapping = {  # Example mapping
                    "Glucose": ["Glucose", "Glucose, CSF", "Glucose, Whole Blood"],
                    "Potassium": ["Potassium", "Potassium, Whole Blood"],
                    # ... (add other mappings as needed for fallback)
                }

            # Create a flat list of all lab name variations
            all_lab_variations = [
                variation
                for variations in lab_name_mapping.values()
                for variation in variations
            ]

            # Filter labs to only include common labs
            labs = labs[labs["label"].isin(all_lab_variations)]

            # Map lab variations to standardised names
            lab_name_reverse_mapping = {}
            for std_name, variations in lab_name_mapping.items():
                for variation in variations:
                    lab_name_reverse_mapping[variation] = std_name

            labs["standardized_label"] = labs["label"].map(lab_name_reverse_mapping)

            # Merge with ICU stays to get stay_id
            labs = pd.merge(
                labs,
                icustays[["subject_id", "hadm_id", "icustay_id", "intime", "outtime"]],
                on=["subject_id", "hadm_id"],
                how="inner",  # Inner merge ensures only labs associated with an ICU stay are kept
            )

            # Filter to labs within ICU stay window
            labs = labs[
                (labs["charttime"] >= labs["intime"])
                & (labs["charttime"] <= labs["outtime"])
            ]

            # Get lab window hours from config
            lab_window_hours = self.config["features"]["lab_values"]["window_hours"]

            # Calculate time from ICU admission - do this once for all labs
            labs["hours_from_admission"] = (
                labs["charttime"] - labs["intime"]
            ).dt.total_seconds() / 3600

            # Filter to labs within window - do this once for all labs
            window_labs = labs[
                labs["hours_from_admission"] <= lab_window_hours
            ].copy()  # Use copy to avoid SettingWithCopyWarning

            # Convert valuenum to numeric, coercing errors
            window_labs["valuenum"] = pd.to_numeric(
                window_labs["valuenum"], errors="coerce"
            )
            # Drop rows where valuenum could not be converted
            window_labs = window_labs.dropna(
                subset=["valuenum", "standardized_label", "icustay_id"]
            )  # Ensure key columns are not NaN

            # If there are no labs after filtering, return an empty dataframe with ID columns
            if len(window_labs) == 0:
                self.logger.warning(
                    "No valid lab data found within the specified window after cleaning."
                )
                return pd.DataFrame(columns=["subject_id", "hadm_id", "icustay_id"])

            # Sort by time for sequence creation
            window_labs = window_labs.sort_values(
                ["subject_id", "hadm_id", "icustay_id", "charttime"]
            )

            # Create (time, value) tuples
            window_labs["time_value_tuple"] = list(
                zip(window_labs["hours_from_admission"], window_labs["valuenum"])
            )

            # Group by stay and lab type, then aggregate tuples into a list
            self.logger.info("Aggregating lab data into sequences (time, value)")
            lab_sequences = (
                window_labs.groupby(
                    ["subject_id", "hadm_id", "icustay_id", "standardized_label"]
                )["time_value_tuple"]
                .apply(list)  # Use list aggregation
                .reset_index()
            )

            # Pivot the table to have one row per stay_id and columns for each lab sequence
            lab_features = lab_sequences.pivot_table(
                index=["subject_id", "hadm_id", "icustay_id"],
                columns="standardized_label",
                values="time_value_tuple",
                aggfunc=list,  # Use list aggregation
            )

            # Rename columns to indicate they are lab sequences
            lab_features.columns = [
                f"lab_{col}_sequence_data" for col in lab_features.columns
            ]
            lab_features = lab_features.reset_index()

            self.logger.info(
                f"Finished processing lab data into sequences. Shape: {lab_features.shape}"
            )
            return lab_features  # type: ignore [no-any-return]

        except Exception as e:
            self.logger.error(f"Error processing lab data: {e}", exc_info=True)
            # Return empty dataframe with expected ID columns in case of error
            return pd.DataFrame(columns=["subject_id", "hadm_id", "icustay_id"])

    def _extract_vital_features(
        self, admissions: pd.DataFrame, icustays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Loads and processes MIMIC-III vital signs (CHARTEVENTS.csv, D_ITEMS.csv).

        Args:
            admissions (pd.DataFrame): Processed admission data (potentially unused).
            icustays (pd.DataFrame): Processed ICU stay data with 'intime', 'outtime'.

        Returns:
            pd.DataFrame: DataFrame containing vital sign sequences per ICU stay, or an empty
                          DataFrame with ID columns if loading/processing fails.
        """
        self.logger.info("Extracting vital sign features")
        vital_features_final = (
            icustays[["subject_id", "hadm_id", "icustay_id"]].copy().drop_duplicates()
        )  # Start with unique ICU stays

        # Load MIMIC-III vital sign data
        try:
            mimic3_path = get_data_path("raw", "mimic_iii", self.config)
            chart_path = os.path.join(mimic3_path, "CHARTEVENTS.csv")
            items_path = os.path.join(mimic3_path, "D_ITEMS.csv")

            if os.path.exists(chart_path) and os.path.exists(items_path):
                # Load chart events data - use lowercase column names for parse_dates
                # Consider loading in chunks if the file is very large
                self.logger.info(f"Loading chart events from {chart_path}...")
                chart_events = pd.read_csv(
                    chart_path, parse_dates=["charttime"], low_memory=False
                )
                chart_events.columns = chart_events.columns.str.lower()
                self.logger.info("Chart events loaded.")

                # Load items dictionary
                items = pd.read_csv(items_path)
                items.columns = items.columns.str.lower()

                # Process vital sign data
                processed_vitals = self._process_vital_data(
                    chart_events, items, admissions, icustays
                )

                # Merge processed vitals back onto the base df with all stays
                if not processed_vitals.empty:
                    # Use left merge to keep all stays
                    vital_features_final = pd.merge(
                        vital_features_final,
                        processed_vitals,
                        on=["subject_id", "hadm_id", "stay_id"],
                        how="left",
                    )
                else:
                    self.logger.warning(
                        "Processed vitals DataFrame is empty after _process_vital_data."
                    )

            else:
                self.logger.warning(
                    f"MIMIC-III chart events or item dictionary not found (checked {chart_path}, {items_path})"
                )

        except Exception as e:
            self.logger.error(
                f"Error loading or processing MIMIC-III vital sign data: {e}",
                exc_info=True,
            )
            # Return the base df with IDs only if loading/processing fails

        return vital_features_final

    def _process_vital_data(
        self,
        chart_events: pd.DataFrame,
        items: pd.DataFrame,
        admissions: pd.DataFrame,  # Unused
        icustays: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Process vital sign data: filter, standardize, calculate relative time, create sequences.

        Filters chart events based on mappings and time window, converts values to numeric,
        handles temperature unit conversion, calculates time relative to ICU admission,
        groups by stay and vital sign type, and pivots to create columns containing
        lists of (time, value) tuples.

        Args:
            chart_events (pd.DataFrame): Raw CHARTEVENTS data.
            items (pd.DataFrame): D_ITEMS data.
            admissions (pd.DataFrame): Admission data (unused).
            icustays (pd.DataFrame): ICU stay data with 'intime', 'outtime', 'stay_id'.

        Returns:
            pd.DataFrame: DataFrame with columns 'subject_id', 'hadm_id', 'stay_id',
                          and 'vital_FEATURE_sequence_data' for each processed vital sign.
                          Returns an empty DataFrame with ID columns on error.
        """
        try:
            # Import load_mappings specifically from config module
            from utils.config import load_mappings

            self.logger.info(
                "Processing vital sign data with optimised vectorized operations"
            )

            # Load vital sign mappings from configuration
            try:
                # Cache the mappings
                if not hasattr(self, "_vital_mappings"):
                    mappings = load_mappings()
                    self._vital_mappings = mappings.get("vital_signs", {})

                # Get vital sign categories and item IDs from mappings
                vital_categories = self._vital_mappings.get("categories", {})
                vital_itemids = self._vital_mappings.get("itemids", [])

                if not vital_categories or not vital_itemids:
                    self.logger.warning(
                        "Vital sign mappings ('categories' or 'itemids') are empty or missing."
                    )
                    raise ValueError("Missing vital sign mappings")  # Trigger fallback

                self.logger.info(
                    f"Loaded {len(vital_categories)} vital sign categories from configuration"
                )
            except Exception as e:
                self.logger.warning(
                    f"Error loading vital sign mappings from configuration: {e}"
                )
                self.logger.warning("Falling back to hardcoded vital sign item IDs")

                # Fallback to hardcoded item IDs if mappings fail
                vital_itemids = [
                    211,
                    220045,  # Heart Rate
                    51,
                    442,
                    455,
                    6701,
                    220179,
                    220050,  # Systolic BP
                    8368,
                    8440,
                    8441,
                    8555,
                    220180,
                    220051,  # Diastolic BP
                    52,
                    6702,
                    456,
                    220181,
                    220052,  # Mean BP
                    618,
                    615,
                    220210,
                    224690,  # Respiratory Rate
                    646,
                    220277,  # SpO2
                    678,
                    223761,  # Temperature C
                    223762,  # Temperature F (convert later)
                ]
                # Define categories based on hardcoded IDs (example)
                vital_categories = {
                    "heart_rate": [211, 220045],
                    "systolic_bp": [51, 442, 455, 6701, 220179, 220050],
                    "diastolic_bp": [8368, 8440, 8441, 8555, 220180, 220051],
                    "mean_bp": [52, 6702, 456, 220181, 220052],
                    "resp_rate": [618, 615, 220210, 224690],
                    "spo2": [646, 220277],
                    "temp_c": [678, 223761],
                    "temp_f": [223762],
                }

            # Filter chart events to only include relevant vital sign item IDs
            chart_events = chart_events[
                chart_events["itemid"].isin(vital_itemids)
            ].copy()  # Use copy

            # Merge with items to get labels (optional, but useful for mapping)
            # chart_events = pd.merge(
            #     chart_events, items[["itemid", "label"]], on="itemid", how="left"
            # )

            # Map item IDs to standardized vital sign names
            itemid_to_category = {}
            for category, itemids_list in vital_categories.items():
                for itemid in itemids_list:
                    itemid_to_category[itemid] = category

            chart_events["standardized_label"] = chart_events["itemid"].map(
                itemid_to_category
            )

            # Merge with ICU stays to get stay_id and time boundaries
            # Ensure stay_id is present after merge
            chart_events = pd.merge(
                chart_events,
                icustays[["subject_id", "hadm_id", "intime", "outtime"]],
                on=["subject_id", "hadm_id"],
                how="inner",  # Inner merge keeps only events associated with these stays
            )

            # Filter to events within ICU stay window
            chart_events = chart_events[
                (chart_events["charttime"] >= chart_events["intime"])
                & (chart_events["charttime"] <= chart_events["outtime"])
            ]

            # Get vital sign window hours from config
            vital_window_hours = self.config["features"]["vitals"]["window_hours"]

            # Calculate time from ICU admission
            chart_events["hours_from_admission"] = (
                chart_events["charttime"] - chart_events["intime"]
            ).dt.total_seconds() / 3600

            # Filter to events within the specified window
            window_vitals = chart_events[
                chart_events["hours_from_admission"] <= vital_window_hours
            ].copy()  # Use copy

            # Convert valuenum to numeric, coercing errors
            window_vitals["valuenum"] = pd.to_numeric(
                window_vitals["valuenum"], errors="coerce"
            )
            # Drop rows where valuenum could not be converted or essential IDs/labels are missing
            window_vitals = window_vitals.dropna(
                subset=["valuenum", "standardized_label", "icustay_id"]
            )

            # Convert Temperature F to C if present
            if "temp_f" in window_vitals["standardized_label"].unique():
                temp_f_mask = window_vitals["standardized_label"] == "temp_f"
                window_vitals.loc[temp_f_mask, "valuenum"] = (
                    (window_vitals.loc[temp_f_mask, "valuenum"] - 32) * 5 / 9
                )
                # Rename category to temp_c after conversion
                window_vitals.loc[temp_f_mask, "standardized_label"] = "temp_c"

            # If there are no vitals after filtering, return an empty dataframe with ID columns
            if len(window_vitals) == 0:
                self.logger.warning(
                    "No vital sign data found within the specified window after cleaning."
                )
                return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])

            # Ensure stay_id is present before sorting (should be guaranteed by merge and dropna)
            if "stay_id" not in window_vitals.columns:
                self.logger.error(
                    "'stay_id' column unexpectedly missing before sorting vital signs."
                )
                return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])

            # Sort by time for sequence creation
            window_vitals = window_vitals.sort_values(
                ["subject_id", "hadm_id", "stay_id", "charttime"]
            )

            # Create (time, value) tuples
            window_vitals["time_value_tuple"] = list(
                zip(window_vitals["hours_from_admission"], window_vitals["valuenum"])
            )

            # Group by stay and vital type, then aggregate tuples into a list
            self.logger.info("Aggregating vital data into sequences (time, value)")
            # Ensure stay_id is included in groupby keys
            vital_sequences = (
                window_vitals.groupby(
                    ["subject_id", "hadm_id", "stay_id", "standardized_label"]
                )["time_value_tuple"]
                .apply(list)  # Use list aggregation
                .reset_index()
            )

            # Pivot the table to have one row per stay_id and columns for each vital sequence
            vital_features = vital_sequences.pivot_table(
                index=["subject_id", "hadm_id", "stay_id"],
                columns="standardized_label",
                values="time_value_tuple",
                aggfunc=list,  # Use list aggregation
            )

            # Rename columns to indicate they are vital sequences
            # Sanitize column names (replace spaces, etc.)
            def sanitize_vital_col(col: str) -> str:
                return f"vital_{col.lower().replace(' ', '_')}_sequence_data"

            vital_features.columns = [
                sanitize_vital_col(col) for col in vital_features.columns
            ]
            vital_features = vital_features.reset_index()

            self.logger.info(
                f"Finished processing vital data into sequences. Shape: {vital_features.shape}"
            )
            return vital_features  # type: ignore [no-any-return]

        except Exception as e:
            self.logger.error(f"Error processing vital sign data: {e}", exc_info=True)
            # Return empty dataframe with expected ID columns in case of error
            return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])


class DiagnosisFeatureExtractor(BaseFeatureExtractor):
    """
    Extractor for diagnosis features based on ICD-9 codes.

    Loads MIMIC-III diagnosis data, filters for ICD-9 codes, maps codes to
    categories based on configuration (ranges and specific codes), and creates
    one-hot encoded category flags and a diagnosis count per admission.
    """

    def extract(self) -> pd.DataFrame:
        """
        Extracts diagnosis category features and counts.

        Loads DIAGNOSES_ICD.csv, processes it using _process_diagnosis_data,
        and merges the results with the base admission list to ensure all
        admissions are represented (filling missing diagnosis features with 0).

        Returns:
            pd.DataFrame: DataFrame with 'subject_id', 'hadm_id', 'diag_count',
                          and 'diag_category_...' columns for each category.
        """
        self.logger.info("Extracting diagnosis features")

        # Load admission data to get the base list of admissions
        admission_path = get_data_path("processed", "admission_data", self.config)
        try:
            admissions = pd.read_csv(admission_path)[["subject_id", "hadm_id"]]
        except FileNotFoundError:
            self.logger.error(
                f"Base admission data not found at {admission_path}. Cannot extract diagnosis features."
            )
            return pd.DataFrame(
                columns=["subject_id", "hadm_id"]
            )  # Return empty with IDs

        # Load MIMIC-III diagnosis data
        try:
            mimic3_path = get_data_path("raw", "mimic_iii", self.config)
            diag_path = os.path.join(mimic3_path, "DIAGNOSES_ICD.csv")

            if os.path.exists(diag_path):
                diagnoses = pd.read_csv(diag_path)
                diagnoses.columns = diagnoses.columns.str.lower()

                # Process diagnosis data
                processed_diagnoses = self._process_diagnosis_data(diagnoses)

                # Merge with base admissions to ensure all admissions are included
                features = pd.merge(
                    admissions,
                    processed_diagnoses,
                    on=["subject_id", "hadm_id"],
                    how="left",
                )
                # Fill NaNs created by the left merge (admissions with no diagnoses) with 0
                # Identify columns to fill (all except IDs)
                cols_to_fill = features.columns.difference(["subject_id", "hadm_id"])
                features[cols_to_fill] = features[cols_to_fill].fillna(0)
                # Ensure integer types for count/flag columns
                for col in cols_to_fill:
                    # Check if column exists before trying to access dtype
                    if col in features.columns and pd.api.types.is_numeric_dtype(
                        features[col]
                    ):
                        # Check if float before converting to int to avoid errors on existing ints
                        if features[col].dtype == "float":
                            features[col] = features[col].astype(int)

            else:
                self.logger.warning(
                    f"MIMIC-III diagnosis data not found (checked {diag_path})"
                )
                # Return base admissions with zeroed diagnosis columns if file not found
                # Need to determine expected columns from mappings
                try:
                    mappings = load_mappings()
                    icd9_categories = mappings.get("icd9_categories", {})
                    category_names = list(
                        icd9_categories.get("ranges", {}).keys()
                    ) + list(icd9_categories.get("specific_codes", {}).keys())

                    def sanitize_col_name(col: str) -> str:
                        # import re # Already imported globally
                        # Sanitize the category name itself first
                        sanitized_base = re.sub(r"[^a-zA-Z0-9_]+", "_", str(col))
                        sanitized_base = (
                            re.sub(r"_+", "_", sanitized_base).strip("_").lower()
                        )  # Also convert to lowercase
                        # Construct the final column name
                        final_name = f"diag_category_{sanitized_base}"
                        return final_name if sanitized_base else f"col_{hash(col)}"

                    diag_cols = [sanitize_col_name(name) for name in category_names] + [
                        "diag_count"
                    ]
                except Exception as e:
                    self.logger.warning(
                        f"Could not load mappings to determine diagnosis columns for missing file: {e}. Returning only IDs."
                    )
                    diag_cols = ["diag_count"]  # Fallback

                for col in diag_cols:
                    admissions[col] = 0  # Add zeroed columns
                features = admissions

        except Exception as e:
            self.logger.error(
                f"Error loading or processing MIMIC-III diagnosis data: {e}",
                exc_info=True,
            )
            # Return base admissions if processing fails
            features = admissions
            # Optionally add a zeroed count column
            if "diag_count" not in features.columns:
                features["diag_count"] = 0

        # TODO: Add MIMIC-IV diagnosis processing if needed

        return features

    def _process_diagnosis_data(self, diagnoses: pd.DataFrame) -> pd.DataFrame:
        """
        Processes raw diagnosis data to create category flags and counts per admission.

        Filters for ICD-9 codes, maps codes to categories using configured ranges and
        specific codes, handles V/E codes, one-hot encodes categories, calculates
        diagnosis counts, and aggregates results to the admission level.

        Args:
            diagnoses (pd.DataFrame): Raw diagnosis data (e.g., from DIAGNOSES_ICD.csv).
                                      Expected columns: 'subject_id', 'hadm_id',
                                      'icd9_code' (or 'icd_code'), optionally 'icd_version'.

        Returns:
            pd.DataFrame: DataFrame aggregated by admission ('subject_id', 'hadm_id')
                          with one-hot encoded 'diag_category_*' columns and a 'diag_count'.
                          Returns an empty DataFrame if essential columns are missing or
                          no valid ICD-9 diagnoses are found.
        """
        self.logger.info("Processing diagnosis data")

        # Ensure correct column name for ICD codes (handle potential variations)
        if "icd_code" in diagnoses.columns and "icd9_code" not in diagnoses.columns:
            diagnoses = diagnoses.rename(columns={"icd_code": "icd9_code"})
        elif "icd9_code" not in diagnoses.columns:
            self.logger.error(
                "Missing 'icd9_code' or 'icd_code' column in diagnosis data."
            )
            return pd.DataFrame(
                columns=["subject_id", "hadm_id"]
            )  # Return empty with IDs

        # Filter for ICD-9 codes if version column exists
        if "icd_version" in diagnoses.columns:
            diagnoses = diagnoses[diagnoses["icd_version"] == 9].copy()
        else:
            self.logger.warning(
                "'icd_version' column not found. Assuming all codes are ICD-9."
            )

        # Get unique admission identifiers from the input diagnoses df *before* filtering further
        # If diagnoses is empty initially, this will be empty too.
        if diagnoses.empty:
            admission_ids = pd.DataFrame(columns=["subject_id", "hadm_id"])
        else:
            admission_ids = (
                diagnoses[["subject_id", "hadm_id"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

        # START FIX 1: Empty DataFrame Handling
        if diagnoses.empty:
            self.logger.warning("No ICD-9 diagnoses found after filtering.")
            # Return a DataFrame with IDs and zeroed-out expected columns
            # Need to know the expected category columns from mappings
            try:
                mappings = load_mappings()
                icd9_categories = mappings.get("icd9_categories", {})
                category_names = list(icd9_categories.get("ranges", {}).keys()) + list(
                    icd9_categories.get("specific_codes", {}).keys()
                )

                # Apply sanitization to expected column names
                # import re # Already imported globally
                def sanitize_col_name(col: str) -> str:
                    # Sanitize the category name itself first
                    sanitized_base = re.sub(r"[^a-zA-Z0-9_]+", "_", str(col))
                    sanitized_base = (
                        re.sub(r"_+", "_", sanitized_base).strip("_").lower()
                    )  # Also convert to lowercase
                    # Construct the final column name
                    final_name = f"diag_category_{sanitized_base}"
                    return final_name if sanitized_base else f"col_{hash(col)}"

                diag_cols = [sanitize_col_name(name) for name in category_names] + [
                    "diag_count"
                ]
            except Exception as e:
                self.logger.warning(
                    f"Could not load mappings to determine diagnosis columns for empty result: {e}. Returning only IDs."
                )
                diag_cols = ["diag_count"]  # Fallback

            if not admission_ids.empty:
                # If we had admissions but no diagnoses, create zeroed rows for them
                empty_features = pd.DataFrame(
                    0, index=admission_ids.index, columns=diag_cols
                )
                result_df = pd.concat([admission_ids, empty_features], axis=1)
                # Ensure correct types after creating zeroed df
                for col in diag_cols:
                    if col in result_df.columns:
                        result_df[col] = result_df[col].astype(int)
                return result_df
            else:
                # If the input 'diagnoses' was truly empty (no admissions), return empty with expected columns
                return pd.DataFrame(columns=["subject_id", "hadm_id"] + diag_cols)
        # END FIX 1

        # Calculate diagnosis count per admission BEFORE filtering by category
        diag_counts = (
            diagnoses.groupby(["subject_id", "hadm_id"])
            .size()
            .reset_index(name="diag_count")
        )

        # Get ICD-9 category for each diagnosis
        diagnoses["icd9_category"] = diagnoses["icd9_code"].apply(
            self._get_icd9_category
        )

        # Filter out diagnoses without a category (optional, depends on desired behavior)
        processed_diagnoses = diagnoses.dropna(
            subset=["icd9_category"]
        ).copy()  # Use copy

        # If no diagnoses remain after category mapping, return IDs + counts + zeroed categories
        if processed_diagnoses.empty:
            self.logger.warning("No diagnoses remained after mapping to categories.")
            # Determine expected category columns again
            try:
                mappings = load_mappings()
                icd9_categories = mappings.get("icd9_categories", {})
                category_names = list(icd9_categories.get("ranges", {}).keys()) + list(
                    icd9_categories.get("specific_codes", {}).keys()
                )

                def sanitize_col_name(col: str) -> str:
                    sanitized_base = re.sub(r"[^a-zA-Z0-9_]+", "_", str(col))
                    sanitized_base = (
                        re.sub(r"_+", "_", sanitized_base).strip("_").lower()
                    )
                    final_name = f"diag_category_{sanitized_base}"
                    return final_name if sanitized_base else f"col_{hash(col)}"

                diag_category_cols = [
                    sanitize_col_name(name) for name in category_names
                ]
            except Exception as e:
                self.logger.warning(
                    f"Could not load mappings to determine diagnosis columns for empty result: {e}."
                )
                diag_category_cols = []

            # Merge counts with all original admission IDs
            final_df = pd.merge(
                admission_ids, diag_counts, on=["subject_id", "hadm_id"], how="left"
            ).fillna({"diag_count": 0})
            # Add zeroed category columns
            for col in diag_category_cols:
                final_df[col] = 0
            # Ensure types
            for col in diag_category_cols + ["diag_count"]:
                if col in final_df.columns:
                    final_df[col] = final_df[col].astype(int)
            return final_df

        # One-hot encode the categories
        diagnosis_dummies = pd.get_dummies(
            processed_diagnoses["icd9_category"], prefix="diag_category", dtype=int
        )

        # Combine IDs with dummies
        processed_diagnoses = pd.concat(
            [processed_diagnoses[["subject_id", "hadm_id"]], diagnosis_dummies], axis=1
        )

        # START FIX 2: Sanitization and Aggregation
        # Sanitize only the newly created category column names before aggregation
        category_cols_to_sanitize = [
            col
            for col in processed_diagnoses.columns
            if col.startswith("diag_category_")
        ]
        sanitized_col_names = {}

        # import re # Already imported globally
        # Rename to avoid redefinition from line 1069
        def sanitize_diag_col_name(col: str) -> str:
            # Remove the initial 'diag_category_' prefix for sanitization
            base_name = col.replace("diag_category_", "", 1)
            sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(base_name))
            sanitized = (
                re.sub(r"_+", "_", sanitized).strip("_").lower()
            )  # Also convert to lowercase
            # Add the prefix back
            final_name = f"diag_category_{sanitized}"
            return final_name if sanitized else f"col_{hash(col)}"

        for col in category_cols_to_sanitize:
            sanitized_col_names[col] = sanitize_diag_col_name(col)

        processed_diagnoses = processed_diagnoses.rename(columns=sanitized_col_names)
        self.logger.debug(f"Sanitized diagnosis column names: {sanitized_col_names}")

        # Aggregate features per admission (summing the one-hot flags)
        agg_dict = {
            col: "max" for col in sanitized_col_names.values()
        }  # Use max (or sum) for flags

        # Group by admission and aggregate the dummy variables
        final_diagnoses_agg = (
            processed_diagnoses.groupby(["subject_id", "hadm_id"])
            .agg(agg_dict)
            .reset_index()
        )

        # Merge the aggregated category flags with the counts
        final_diagnoses = pd.merge(
            admission_ids,  # Start with all unique admissions from the input
            diag_counts,
            on=["subject_id", "hadm_id"],
            how="left",
        )
        final_diagnoses = pd.merge(
            final_diagnoses,
            final_diagnoses_agg,
            on=["subject_id", "hadm_id"],
            how="left",
        )

        # Fill NaN values that might result from aggregation/merges
        # Fill counts with 0, fill category flags with 0
        final_diagnoses["diag_count"] = final_diagnoses["diag_count"].fillna(0)
        for col in sanitized_col_names.values():
            if col in final_diagnoses.columns:
                final_diagnoses[col] = final_diagnoses[col].fillna(0)
            else:
                # If a category column wasn't created at all (e.g., no codes mapped to it)
                # add it with zeros
                final_diagnoses[col] = 0

        # Convert flag columns and count to integer
        for col in list(sanitized_col_names.values()) + ["diag_count"]:
            if col in final_diagnoses.columns:
                final_diagnoses[col] = final_diagnoses[col].astype(int)

        self.logger.info(
            f"Created {len(sanitized_col_names)} diagnosis category features and count for {len(final_diagnoses)} admissions"
        )
        return final_diagnoses
        # END FIX 2

    def _get_icd9_category(self, icd9_code: str) -> Optional[str]:
        """
        Get the category for an ICD-9 code based on mappings defined in config.

        Checks specific codes first, then ranges. Handles V and E codes separately.
        Converts codes to numeric for range comparison where possible.

        Args:
            icd9_code (str): ICD-9 code (as string).

        Returns:
            Optional[str]: Category name if found, otherwise None.
        """
        # Lazy load and cache mappings
        if not hasattr(self, "_icd9_mappings"):
            try:
                mappings = load_mappings()
                self._icd9_mappings = mappings.get("icd9_categories", {})
                # Pre-process ranges for faster lookup
                self._icd9_ranges = []
                for category, code_ranges_list in self._icd9_mappings.get(
                    "ranges", {}
                ).items():
                    # Handle list of lists for ranges, e.g., "Category": [[1, 10], [20, 30]]
                    if isinstance(code_ranges_list, list):
                        for (
                            code_range
                        ) in code_ranges_list:  # Iterate through the inner list(s)
                            if isinstance(code_range, list) and len(code_range) == 2:
                                try:
                                    # Attempt to convert range boundaries to integers
                                    start = int(code_range[0])
                                    end = int(code_range[1])
                                    self._icd9_ranges.append((category, start, end))
                                except (ValueError, TypeError):
                                    self.logger.warning(
                                        f"Invalid non-integer range format within list for ICD-9 category '{category}': {code_range}. Skipping."
                                    )
                            else:
                                self.logger.warning(
                                    f"Invalid range format within list for ICD-9 category '{category}': {code_range}. Expected [start, end]. Skipping."
                                )
                    else:
                        self.logger.warning(
                            f"Invalid range format for ICD-9 category '{category}': {code_ranges_list}. Expected list of lists. Skipping."
                        )
                # Sort ranges for efficient checking (optional but good practice)
                self._icd9_ranges.sort(key=lambda x: x[1])

                self._icd9_specific = self._icd9_mappings.get("specific_codes", {})

            except Exception as e:
                self.logger.error(f"Error loading ICD-9 mappings: {e}")
                self._icd9_mappings = {}
                self._icd9_ranges = []
                self._icd9_specific = {}

        if not isinstance(icd9_code, str):
            return None  # Handle non-string input

        # Handle V codes and E codes (often supplementary)
        if icd9_code.startswith("V"):
            return "supplementary_classification_v_codes"
        if icd9_code.startswith("E"):
            return "external_causes_of_injury_e_codes"

        # Check specific codes first
        for category, codes in self._icd9_specific.items():
            if icd9_code in codes:
                return category  # type: ignore [no-any-return]

        # Check ranges
        try:
            # Convert code to numeric for range comparison (handle potential errors)
            # Take only the part before any decimal point for range check
            code_prefix = icd9_code.split(".")[0]
            code_num_str = "".join(filter(str.isdigit, code_prefix))
            if not code_num_str:
                return None  # Cannot convert to number
            code_num = int(code_num_str)

            for category, start, end in self._icd9_ranges:
                if start <= code_num <= end:
                    return category  # type: ignore [no-any-return]
        except ValueError:
            # Handle cases where the code cannot be converted to an integer after removing non-digits
            return None
        except Exception as e:
            self.logger.warning(f"Error processing ICD-9 code '{icd9_code}': {e}")
            return None

        return None  # No category found
