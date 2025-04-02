"""
Feature extractors for the MIMIC datasets.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import get_data_path, get_logger, load_config  # Corrected direct import
from utils.config import load_mappings  # Import load_mappings specifically

logger = get_logger(__name__)


class BaseFeatureExtractor(ABC):
    """
    Base class for feature extractors.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature extractor.

        Args:
            config (Optional[Dict], optional): Configuration dictionary.
                If None, loads the default configuration. Defaults to None.
        """
        self.config = config if config is not None else load_config()
        self.logger = logger

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """
        Extract features.

        Returns:
            pd.DataFrame: Extracted features
        """
        pass

    def save(self, data: pd.DataFrame, output_path: str) -> None:
        """
        Save the extracted features.

        Args:
            data (pd.DataFrame): Data to save
            output_path (str): Path to save the data to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save data
        data.to_csv(output_path, index=False)
        self.logger.info(f"Saved extracted features to {output_path}")


class DemographicFeatureExtractor(BaseFeatureExtractor):
    """
    Extractor for demographic features.
    """

    def extract(self) -> pd.DataFrame:
        """
        Extract demographic features from patient and admission data.

        Returns:
            pd.DataFrame: Demographic features
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
            on=["subject_id", "source"],
            how="left",
            suffixes=("", "_patient"),
        )

        # Extract features
        features = self._extract_demographic_features(data)

        return features

    def _extract_demographic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract demographic features from merged patient and admission data.

        Args:
            data (pd.DataFrame): Merged patient and admission data

        Returns:
            pd.DataFrame: Demographic features
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

        # Drop original categorical columns if they exist in the 'features' df (shouldn't happen with current logic)
        # cols_to_drop = [col for col in ["gender", "age_group"] if col in features.columns]
        # if cols_to_drop:
        #      logger.debug(f"Dropping original categorical columns from features: {cols_to_drop}")
        #      features = features.drop(columns=cols_to_drop)

        logger.info(
            f"Finished extracting demographic features. Shape: {features.shape}"
        )
        return features


class ClinicalFeatureExtractor(BaseFeatureExtractor):
    """
    Extractor for clinical features from lab values and vital signs.
    """

    def extract(self) -> pd.DataFrame:
        """
        Extract clinical features from lab values and vital signs.

        Returns:
            pd.DataFrame: Clinical features
        """
        self.logger.info("Extracting clinical features")

        # Load admission data
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

        # Load ICU stay data
        icu_path = get_data_path("processed", "icu_data", self.config)
        icustays = pd.read_csv(icu_path, parse_dates=["intime", "outtime"])

        # Extract lab features
        lab_features = self._extract_lab_features(admissions, icustays)

        # Extract vital sign features
        vital_features = self._extract_vital_features(admissions, icustays)

        # Combine features
        # Ensure ID columns are present before merge
        if lab_features.empty:
            logger.warning(
                "Lab features DataFrame is empty. Clinical features will only contain vital signs."
            )
            features = vital_features
        elif vital_features.empty:
            logger.warning(
                "Vital features DataFrame is empty. Clinical features will only contain lab values."
            )
            features = lab_features
        elif not all(
            col in lab_features.columns for col in ["subject_id", "hadm_id", "stay_id"]
        ):
            logger.error(
                "Lab features DataFrame missing required ID columns for merge."
            )
            features = vital_features  # Or handle error differently
        elif not all(
            col in vital_features.columns
            for col in ["subject_id", "hadm_id", "stay_id"]
        ):
            logger.error(
                "Vital features DataFrame missing required ID columns for merge."
            )
            features = lab_features  # Or handle error differently
        else:
            features = pd.merge(
                lab_features,
                vital_features,
                on=["subject_id", "hadm_id", "stay_id"],
                how="outer",
            )

        return features

    def _extract_lab_features(
        self, admissions: pd.DataFrame, icustays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract features from laboratory values.

        Args:
            admissions (pd.DataFrame): Admission data
            icustays (pd.DataFrame): ICU stay data

        Returns:
            pd.DataFrame: Laboratory features
        """
        self.logger.info("Extracting laboratory features")
        lab_features_final = icustays[
            ["subject_id", "hadm_id", "stay_id"]
        ].copy()  # Start with all stays

        # Load MIMIC-III lab data
        try:
            mimic3_path = get_data_path("raw", "mimic_iii", self.config)
            lab_path = os.path.join(mimic3_path, "LABEVENTS.csv")
            lab_items_path = os.path.join(mimic3_path, "D_LABITEMS.csv")

            if os.path.exists(lab_path) and os.path.exists(lab_items_path):
                # Load lab data - use lowercase column names for parse_dates
                labs = pd.read_csv(lab_path, parse_dates=["charttime"])
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
                    lab_features_final = pd.merge(
                        lab_features_final,
                        processed_labs,
                        on=["subject_id", "hadm_id", "stay_id"],
                        how="left",
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
        admissions: pd.DataFrame,
        icustays: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Process laboratory data and extract features.
        MODIFIED: Extracts sequences (time, value) instead of aggregates.

        Args:
            labs (pd.DataFrame): Laboratory data
            lab_items (pd.DataFrame): Laboratory items dictionary
            admissions (pd.DataFrame): Admission data
            icustays (pd.DataFrame): ICU stay data

        Returns:
            pd.DataFrame: Laboratory features as sequences per stay
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

                # Create a mapping of common lab names to their variations in the dataset
                lab_name_mapping = {
                    "Glucose": ["Glucose", "Glucose, CSF", "Glucose, Whole Blood"],
                    "Potassium": ["Potassium", "Potassium, Whole Blood"],
                    "Sodium": ["Sodium", "Sodium, Whole Blood"],
                    "Chloride": ["Chloride", "Chloride, Whole Blood"],
                    "Creatinine": ["Creatinine"],
                    "BUN": ["BUN", "Urea Nitrogen"],
                    "Bicarbonate": [
                        "Bicarbonate",
                        "HCO3",
                        "Calculated Bicarbonate, Whole Blood",
                    ],
                    "Anion Gap": ["Anion Gap"],
                    "Hemoglobin": ["Hemoglobin", "Hemoglobin, Whole Blood"],
                    "Hematocrit": ["Hematocrit", "Hematocrit, Calculated"],
                    "WBC": ["WBC", "White Blood Cells"],
                    "Platelet Count": ["Platelet Count", "Platelets"],
                    "Magnesium": ["Magnesium"],
                    "Calcium": ["Calcium", "Calcium, Total"],
                    "Phosphate": ["Phosphate", "Phosphorus"],
                    "Lactate": ["Lactate", "Lactate, Whole Blood"],
                    "pH": ["pH", "pH, Whole Blood"],
                    "pO2": ["pO2", "PO2, Whole Blood"],
                    "pCO2": ["pCO2", "PCO2, Whole Blood"],
                    "Base Excess": ["Base Excess", "Base Excess, Whole Blood"],
                    "Albumin": ["Albumin"],
                    "ALT": ["ALT", "Alanine Aminotransferase"],
                    "AST": ["AST", "Aspartate Aminotransferase"],
                    "Alkaline Phosphatase": ["Alkaline Phosphatase"],
                    "Bilirubin": ["Bilirubin, Total", "Total Bilirubin"],
                    "Troponin": ["Troponin I", "Troponin T", "Troponin"],
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
                icustays[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]],
                on=["subject_id", "hadm_id"],
                how="inner",
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
            window_labs = labs[labs["hours_from_admission"] <= lab_window_hours]

            # Check if aggregation_methods is a list or a string
            if isinstance(
                self.config["features"]["lab_values"]["aggregation_methods"], list
            ):
                aggregation_methods = self.config["features"]["lab_values"][
                    "aggregation_methods"
                ]
            else:
                # Default to a list with a single method if it's not a list
                aggregation_methods = ["mean"]

            # If there are no labs after filtering, return an empty dataframe with ID columns
            if len(window_labs) == 0:
                self.logger.warning("No lab data found within the specified window")
                # Return empty df, merge later will handle it
                return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])

            # --- MODIFIED FOR SEQUENCE EXTRACTION (replacing pivot_table logic) ---

            # Convert 'valuenum' to numeric, coercing errors (ensure this is done before sequence creation)
            window_labs["valuenum"] = pd.to_numeric(
                window_labs["valuenum"], errors="coerce"
            )
            # Drop rows where valuenum is NaN after conversion
            window_labs = window_labs.dropna(subset=["valuenum"])

            # If all values became NaN, return empty
            if len(window_labs) == 0:
                self.logger.warning(
                    "No valid numeric lab values found after coercion within the window"
                )
                return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])

            # Sort by time within each group for consistent sequence order
            window_labs = window_labs.sort_values(
                by=["stay_id", "standardized_label", "hours_from_admission"]
            )

            # Define a function to aggregate time and value into lists
            # Using NamedAgg for clarity and potential future flexibility
            self.logger.info("Aggregating lab data into sequences (time, value)")
            lab_sequences = (
                window_labs.groupby(
                    ["subject_id", "hadm_id", "stay_id", "standardized_label"]
                )
                .agg(
                    timestamps=pd.NamedAgg(column="hours_from_admission", aggfunc=list),
                    values=pd.NamedAgg(column="valuenum", aggfunc=list),
                )
                .reset_index()
            )  # Keep IDs and label as columns

            # Combine the two lists into a tuple for each lab/stay in a new column
            lab_sequences["sequence_data"] = lab_sequences.apply(
                lambda row: (row["timestamps"], row["values"]), axis=1
            )

            # Pivot the table to get labs as columns, using the combined IDs as index
            lab_sequences_pivot = lab_sequences.pivot_table(
                index=["subject_id", "hadm_id", "stay_id"],
                columns="standardized_label",
                values="sequence_data",
                aggfunc="first",  # Should only be one sequence per group
            )

            # Rename columns to include 'lab_' prefix and '_sequence_data' suffix
            lab_sequences_pivot.columns = [
                f"lab_{col}_sequence_data" for col in lab_sequences_pivot.columns
            ]

            # Reset index to make IDs columns again
            lab_sequences_pivot = lab_sequences_pivot.reset_index()

            # Merge with icustays to ensure all stays are included (might not be necessary if pivot_table index included all stays)
            # Let's rely on the pivot table index for now, assuming it covers all relevant stays from window_labs
            # lab_features_final = pd.merge(
            #     icustays[["subject_id", "hadm_id", "stay_id"]],
            #     lab_sequences_pivot,
            #     on=["subject_id", "hadm_id", "stay_id"],
            #     how="left",
            # )

            # Fill NaN values (for stays/labs with no data) with empty tuples
            seq_cols = [
                col
                for col in lab_sequences_pivot.columns
                if col.endswith("_sequence_data")
            ]
            for col in seq_cols:
                if col in lab_sequences_pivot.columns:
                    lab_sequences_pivot[col] = lab_sequences_pivot[col].apply(
                        lambda x: x if pd.notna(x) else ([], [])
                    )

            self.logger.info(
                f"Finished processing lab data into sequences. Shape: {lab_sequences_pivot.shape}"
            )
            # Return the pivoted dataframe with sequences
            return lab_sequences_pivot

        except Exception as e:
            self.logger.error(f"Error in _process_lab_data: {e}", exc_info=True)
            # Return an empty dataframe with ID columns in case of error
            return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])

    def _extract_vital_features(
        self, admissions: pd.DataFrame, icustays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract features from vital signs.

        Args:
            admissions (pd.DataFrame): Admission data
            icustays (pd.DataFrame): ICU stay data

        Returns:
            pd.DataFrame: Vital sign features
        """
        self.logger.info("Extracting vital sign features")
        vital_features_final = icustays[
            ["subject_id", "hadm_id", "stay_id"]
        ].copy()  # Start with all stays

        # Load MIMIC-III chart data
        try:
            mimic3_path = get_data_path("raw", "mimic_iii", self.config)
            chart_path = os.path.join(mimic3_path, "CHARTEVENTS.csv")
            items_path = os.path.join(mimic3_path, "D_ITEMS.csv")

            if os.path.exists(chart_path) and os.path.exists(items_path):
                # Load chart data (this could be very large, so we'll be selective)
                # Define vital sign itemids (example, should ideally come from mappings)
                vital_itemids = [
                    211,
                    220045,
                    51,
                    442,
                    455,
                    6701,
                    220179,
                    220050,
                    8368,
                    8440,
                    8441,
                    8555,
                    220180,
                    220051,
                    456,
                    52,
                    6702,
                    443,
                    220052,
                    220181,
                    615,
                    618,
                    220210,
                    224690,
                    223761,
                    678,
                    223762,
                    646,
                    220277,
                    807,
                    811,
                    1529,
                    3745,
                    3744,
                    225664,
                    220621,
                    226537,
                    198,
                    220739,
                    184,
                    220734,
                    723,
                    223900,
                    454,
                    223901,
                ]

                # Read only the rows with these itemids (consider chunking for large files)
                chart_data = pd.read_csv(
                    chart_path,
                    parse_dates=["charttime"],
                    usecols=[
                        "subject_id",
                        "hadm_id",
                        "stay_id",
                        "itemid",
                        "charttime",
                        "value",
                        "valuenum",
                    ],
                    dtype={
                        "subject_id": int,
                        "hadm_id": int,
                        "stay_id": float,
                        "itemid": int,
                    },
                )
                chart_data = chart_data[
                    chart_data["itemid"].isin(vital_itemids)
                ].copy()  # Filter early
                chart_data.columns = chart_data.columns.str.lower()

                # Load items dictionary
                items = pd.read_csv(items_path)
                items.columns = items.columns.str.lower()

                # Process vital sign data
                processed_vitals = self._process_vital_data(
                    chart_data, items, admissions, icustays
                )

                # Merge processed vitals back onto the base df with all stays
                if not processed_vitals.empty:
                    vital_features_final = pd.merge(
                        vital_features_final,
                        processed_vitals,
                        on=["subject_id", "hadm_id", "stay_id"],
                        how="left",
                    )
            else:
                self.logger.warning(
                    f"MIMIC-III chart data or item dictionary not found (checked {chart_path}, {items_path})"
                )

        except Exception as e:
            self.logger.error(
                f"Error loading or processing MIMIC-III chart data: {e}", exc_info=True
            )
            # Return the base df with IDs only

        return vital_features_final

    def _process_vital_data(
        self,
        chart_data: pd.DataFrame,
        items: pd.DataFrame,
        admissions: pd.DataFrame,
        icustays: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Process vital sign data and extract features using vectorized operations.

        Args:
            chart_data (pd.DataFrame): Chart events data
            items (pd.DataFrame): Items dictionary
            admissions (pd.DataFrame): Admission data
            icustays (pd.DataFrame): ICU stay data

        Returns:
            pd.DataFrame: Vital sign features aggregated per stay
        """
        try:
            # Import load_mappings specifically from config module
            from utils.config import load_mappings

            self.logger.info(
                "Processing vital sign data with optimised vectorized operations"
            )

            # Merge chart data with items to get labels
            chart_data = pd.merge(
                chart_data, items[["itemid", "label"]], on="itemid", how="left"
            )

            # Load vital sign mappings from configuration
            try:
                # Cache the mappings to avoid loading them repeatedly
                if not hasattr(self, "_vital_mappings"):
                    mappings = load_mappings()
                    self._vital_mappings = mappings.get("vital_signs", {})

                # Get vital sign categories from mappings
                vital_categories = self._vital_mappings.get("categories", {})

                self.logger.info(
                    f"Loaded {len(vital_categories)} vital sign categories from configuration"
                )
            except Exception as e:
                self.logger.warning(
                    f"Error loading vital sign mappings from configuration: {e}"
                )
                self.logger.warning("Falling back to hardcoded vital sign categories")

                # Fallback to hardcoded categories if mappings fail
                vital_categories = {
                    "Heart Rate": [211, 220045],
                    "Systolic BP": [51, 442, 455, 6701, 220179, 220050],
                    "Diastolic BP": [8368, 8440, 8441, 8555, 220180, 220051],
                    "Mean BP": [456, 52, 6702, 443, 220052, 220181],
                    "Respiratory Rate": [615, 618, 220210, 224690],
                    "Temperature": [223761, 678, 223762],
                    "SpO2": [646, 220277],
                    "Glucose": [807, 811, 1529, 3745, 3744, 225664, 220621, 226537],
                    "GCS_Motor": [454, 223901],
                    "GCS_Verbal": [723, 223900],
                    "GCS_Eyes": [184, 220734],
                    "GCS_Total": [198, 220739],
                }

            # Create a mapping from itemid to standardized vital sign name
            itemid_to_vital_name = {}
            for vital_name, itemids in vital_categories.items():
                for itemid in itemids:
                    itemid_to_vital_name[itemid] = vital_name.replace(
                        " ", "_"
                    ).lower()  # Standardize name

            # Map itemids to standardized names
            chart_data["standardized_label"] = chart_data["itemid"].map(
                itemid_to_vital_name
            )

            # Filter out rows where itemid didn't map to a vital sign
            chart_data = chart_data.dropna(subset=["standardized_label"])

            # Merge with ICU stays to get stay_id and time window
            # Ensure stay_id is integer for merge if possible
            if "stay_id" in chart_data.columns:
                chart_data["stay_id"] = (
                    chart_data["stay_id"].fillna(-1).astype(int)
                )  # Handle potential NaNs before cast
            if "stay_id" in icustays.columns:
                icustays["stay_id"] = icustays["stay_id"].fillna(-1).astype(int)

            chart_data = pd.merge(
                chart_data,
                icustays[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]],
                on=["subject_id", "hadm_id", "stay_id"],  # Use stay_id if available
                how="inner",
            )

            # Filter to vitals within ICU stay window
            chart_data = chart_data[
                (chart_data["charttime"] >= chart_data["intime"])
                & (chart_data["charttime"] <= chart_data["outtime"])
            ]

            # Get vital window hours from config
            vital_window_hours = self.config["features"]["vitals"]["window_hours"]

            # Calculate time from ICU admission
            chart_data["hours_from_admission"] = (
                chart_data["charttime"] - chart_data["intime"]
            ).dt.total_seconds() / 3600

            # Filter to vitals within window
            window_vitals = chart_data[
                chart_data["hours_from_admission"] <= vital_window_hours
            ]

            # --- MODIFIED FOR SEQUENCE EXTRACTION (replacing pivot_table logic) ---

            # If no vital signs found in window, return empty df with IDs
            if len(window_vitals) == 0:
                self.logger.warning(
                    "No vital sign data found within the specified window"
                )
                return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])

            # Convert 'valuenum' to numeric, coercing errors (ensure this is done before sequence creation)
            # This might have already been done earlier, but ensure it happens before aggregation
            window_vitals["valuenum"] = pd.to_numeric(
                window_vitals["valuenum"], errors="coerce"
            )
            # Drop rows where valuenum is NaN after conversion
            window_vitals = window_vitals.dropna(subset=["valuenum"])

            # If all values became NaN, return empty
            if len(window_vitals) == 0:
                self.logger.warning(
                    "No valid numeric vital values found after coercion within the window"
                )
                return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])

            # Sort by time within each group for consistent sequence order
            window_vitals = window_vitals.sort_values(
                by=["stay_id", "standardized_label", "hours_from_admission"]
            )

            # Aggregate time and value into lists using NamedAgg
            self.logger.info("Aggregating vital data into sequences (time, value)")
            vital_sequences = (
                window_vitals.groupby(
                    ["subject_id", "hadm_id", "stay_id", "standardized_label"]
                )
                .agg(
                    timestamps=pd.NamedAgg(column="hours_from_admission", aggfunc=list),
                    values=pd.NamedAgg(column="valuenum", aggfunc=list),
                )
                .reset_index()
            )

            # Combine timestamps and values into a tuple
            vital_sequences["sequence_data"] = vital_sequences.apply(
                lambda row: (row["timestamps"], row["values"]), axis=1
            )

            # Pivot the table to get vitals as columns
            vital_sequences_pivot = vital_sequences.pivot_table(
                index=["subject_id", "hadm_id", "stay_id"],
                columns="standardized_label",
                values="sequence_data",
                aggfunc="first",  # Should only be one sequence per group
            )

            # Rename columns
            vital_sequences_pivot.columns = [
                f"vital_{col}_sequence_data" for col in vital_sequences_pivot.columns
            ]

            # Reset index
            vital_sequences_pivot = vital_sequences_pivot.reset_index()

            # Fill NaN values (for stays/vitals with no data) with empty tuples
            seq_cols = [
                col
                for col in vital_sequences_pivot.columns
                if col.endswith("_sequence_data")
            ]
            for col in seq_cols:
                if col in vital_sequences_pivot.columns:
                    vital_sequences_pivot[col] = vital_sequences_pivot[col].apply(
                        lambda x: x if pd.notna(x) else ([], [])
                    )

            self.logger.info(
                f"Finished processing vital data into sequences. Shape: {vital_sequences_pivot.shape}"
            )
            return vital_sequences_pivot

        except Exception as e:
            self.logger.error(
                f"Error processing vital sign data into sequences: {e}", exc_info=True
            )
            # Return empty dataframe with IDs in case of error
            return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])


class DiagnosisFeatureExtractor(BaseFeatureExtractor):
    """
    Extractor for diagnosis features.
    """

    def extract(self) -> pd.DataFrame:
        """
        Extract diagnosis features from MIMIC-III and MIMIC-IV.

        Returns:
            pd.DataFrame: Diagnosis features
        """
        self.logger.info("Extracting diagnosis features")
        diagnosis_features_final = pd.DataFrame(
            columns=["subject_id", "hadm_id"]
        )  # Initialize

        # Load MIMIC-III diagnosis data
        try:
            mimic3_path = get_data_path("raw", "mimic_iii", self.config)
            diag_path = os.path.join(mimic3_path, "DIAGNOSES_ICD.csv")

            if os.path.exists(diag_path):
                diagnoses = pd.read_csv(diag_path)
                diagnoses.columns = diagnoses.columns.str.lower()

                # Process diagnosis data
                processed_diagnoses = self._process_diagnosis_data(diagnoses)
                if not processed_diagnoses.empty:
                    # Ensure we only keep necessary columns if merging later
                    diagnosis_features_final = processed_diagnoses
            else:
                self.logger.warning("MIMIC-III diagnosis data not found")

        except Exception as e:
            self.logger.error(
                f"Error loading or processing MIMIC-III diagnosis data: {e}",
                exc_info=True,
            )

        # TODO: Add support for MIMIC-IV diagnosis data (ICD-10)
        # If MIMIC-IV data is loaded, merge it with diagnosis_features_final

        return diagnosis_features_final

    def _process_diagnosis_data(self, diagnoses: pd.DataFrame) -> pd.DataFrame:
        """
        Process diagnosis data and extract features.

        Args:
            diagnoses (pd.DataFrame): Diagnosis data

        Returns:
            pd.DataFrame: Diagnosis features
        """
        self.logger.info("Processing diagnosis data")

        # Filter to ICD-9 codes (MIMIC-III)
        # Ensure icd_version column exists and handle potential non-integer values
        if "icd_version" not in diagnoses.columns:
            logger.warning(
                "Column 'icd_version' not found in diagnoses data. Assuming ICD-9."
            )
        else:
            # Attempt conversion to numeric, coercing errors, then fillna and convert to int
            diagnoses["icd_version"] = (
                pd.to_numeric(diagnoses["icd_version"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            diagnoses = diagnoses[diagnoses["icd_version"] == 9]

        if diagnoses.empty:
            logger.warning("No ICD-9 diagnoses found after filtering.")
            return pd.DataFrame(columns=["subject_id", "hadm_id"])

        # Get ICD-9 categories
        diagnoses["icd9_category"] = diagnoses["icd9_code"].apply(
            self._get_icd9_category
        )

        # Filter out rows where category could not be determined
        diagnoses = diagnoses.dropna(subset=["icd9_category"])

        if diagnoses.empty:
            logger.warning("No valid ICD-9 categories found after mapping.")
            return pd.DataFrame(columns=["subject_id", "hadm_id"])

        # One-hot encode the categories
        # Use pivot_table for efficient one-hot encoding
        try:
            diagnosis_features = pd.pivot_table(
                diagnoses,
                index=["subject_id", "hadm_id"],
                columns="icd9_category",
                aggfunc=lambda x: 1,  # Mark presence of category
                fill_value=0,
            )
        except Exception as e:
            logger.error(
                f"Error creating pivot table for diagnoses: {e}", exc_info=True
            )
            return pd.DataFrame(columns=["subject_id", "hadm_id"])

        # Reset index
        diagnosis_features = diagnosis_features.reset_index()

        # Rename columns for clarity (handle potential invalid characters)
        def sanitize_col_name(col):
            if col in ["subject_id", "hadm_id"]:
                return col
            # Replace invalid characters with underscore
            new_col = "".join(c if c.isalnum() else "_" for c in str(col))
            # Ensure it starts with a letter or underscore
            if not new_col[0].isalpha() and new_col[0] != "_":
                new_col = "_" + new_col
            return new_col.lower()

        diagnosis_features.columns = [
            sanitize_col_name(col) for col in diagnosis_features.columns
        ]

        self.logger.info(
            f"Created {diagnosis_features.shape[1] - 2} diagnosis category features for {diagnosis_features.shape[0]} admissions"
        )

        return diagnosis_features

    def _get_icd9_category(self, icd9_code: str) -> Optional[str]:
        """
        Map an ICD-9 code to its major category based on standard ranges.

        Args:
            icd9_code (str): ICD-9 code

        Returns:
            Optional[str]: ICD-9 category name or None if not found
        """
        if not isinstance(icd9_code, str):
            return None

        icd9_code = icd9_code.strip().upper()  # Clean the code

        # Handle V codes and E codes
        if icd9_code.startswith("V"):
            try:
                # Extract numeric part after 'V'
                code_num_str = icd9_code[1:].split(".")[0]
                if code_num_str:  # Check if there's anything after 'V'
                    code_num = int(code_num_str)
                    # Define V code ranges more precisely if needed
                    return "Supplementary Classification V codes"
            except ValueError:
                pass  # Ignore if conversion fails
            return "Supplementary Classification V codes"  # Default V code category
        elif icd9_code.startswith("E"):
            try:
                # Extract numeric part after 'E'
                code_num_str = icd9_code[1:].split(".")[0]
                if code_num_str:
                    code_num = int(code_num_str)
                    # Define E code ranges more precisely if needed
                    return "External Causes of Injury E codes"
            except ValueError:
                pass
            return "External Causes of Injury E codes"  # Default E code category

        # Handle numeric codes
        try:
            # Remove potential decimal points for comparison
            code_prefix = icd9_code.split(".")[0]
            if not code_prefix:
                return None  # Handle empty string after split
            code_num = int(code_prefix)

            if 1 <= code_num <= 139:
                return "Infectious and Parasitic Diseases"
            elif 140 <= code_num <= 239:
                return "Neoplasms"
            elif 240 <= code_num <= 279:
                return "Endocrine Nutritional Metabolic Immunity"  # Shorter name
            elif 280 <= code_num <= 289:
                return "Blood and Blood forming Organs"
            elif 290 <= code_num <= 319:
                return "Mental Disorders"
            elif 320 <= code_num <= 389:
                return "Nervous System and Sense Organs"
            elif 390 <= code_num <= 459:
                return "Circulatory System"
            elif 460 <= code_num <= 519:
                return "Respiratory System"
            elif 520 <= code_num <= 579:
                return "Digestive System"
            elif 580 <= code_num <= 629:
                return "Genitourinary System"
            elif 630 <= code_num <= 679:
                return "Complications of Pregnancy Childbirth"
            elif 680 <= code_num <= 709:
                return "Diseases of the Skin and Subcutaneous Tissue"
            elif 710 <= code_num <= 739:
                return "Musculoskeletal System"
            elif 740 <= code_num <= 759:
                return "Congenital Anomalies"
            elif 760 <= code_num <= 779:
                return "Perinatal Period Conditions"
            elif 780 <= code_num <= 799:
                return "Symptoms Signs Ill defined Conditions"
            elif 800 <= code_num <= 999:
                return "Injury and Poisoning"
            else:
                return "Unknown Numeric Range"  # Handle unexpected numbers
        except ValueError:
            # Handle cases where the code is not purely numeric after removing prefix
            logger.debug(
                f"Could not convert ICD-9 code prefix to int: '{code_prefix}' from '{icd9_code}'"
            )
            return "Invalid ICD9 Format"
        except Exception as e:
            self.logger.warning(f"Error categorizing ICD-9 code '{icd9_code}': {e}")
            return "Categorization Error"
