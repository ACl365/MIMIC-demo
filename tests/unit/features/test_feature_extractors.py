"""
Unit tests for the feature_extractors module.
"""

import os  # Import the os module
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd

from features.feature_extractors import (
    BaseFeatureExtractor,  # Import Base class if needed for type hinting or direct testing
)
from features.feature_extractors import (
    ClinicalFeatureExtractor,
    DemographicFeatureExtractor,
    DiagnosisFeatureExtractor,
)

# Assuming utils and feature_extractors are importable after 'pip install -e .'
from utils.config import load_config  # Need this for BaseFeatureExtractor init

# Mock config for tests
MOCK_CONFIG = {
    "data": {
        "raw": {"mimic_iii": "mock/raw/mimic_iii", "mimic_iv": "mock/raw/mimic_iv"},
        "processed": {
            "base_path": "mock/processed",
            "patient_data": "mock/processed/patient_data.csv",
            "admission_data": "mock/processed/admission_data.csv",
            "icu_data": "mock/processed/icu_data.csv",  # Added for clinical extractor
            "combined_features": "mock/processed/combined_features.csv",
        },
        "external": {},
    },
    "features": {
        "demographic": {"include": True, "age_bins": [0, 18, 65, 100]},
        "vitals": {
            "include": True,
            "window_hours": 24,
            "aggregation_methods": ["mean", "std"],
        },
        "lab_values": {
            "include": True,
            "window_hours": 24,
            "aggregation_methods": ["mean"],
        },
        "medications": {"include": False},
        "procedures": {"include": False},
        "diagnoses": {"include": True},
        "temporal": {"include": False},
    },
    # Add other sections if BaseFeatureExtractor or others need them during init
    "logging": {"level": "INFO"},
    "models": {},
    "evaluation": {},
    "api": {},
    "dashboard": {},
}

# Mock mappings
MOCK_MAPPINGS = {
    "lab_tests": {
        "common_labs": ["Glucose", "Potassium"],
        "mappings": {"50809": "Glucose", "50971": "Potassium"},
        "lab_name_variations": {"Glucose": ["Glucose"], "Potassium": ["Potassium"]},
    },
    "vital_signs": {
        "categories": {"Heart Rate": [211], "Systolic BP": [51]},
        "itemids": [211, 51],
    },
    "icd9_categories": {
        "ranges": {"Infectious Diseases": [[1, 139]]},
        "specific_codes": {},
    },  # Reverted: Ranges should be list of lists
}


class TestDemographicFeatureExtractor(unittest.TestCase):

    @patch("utils.config.load_config", return_value=MOCK_CONFIG)
    @patch("features.feature_extractors.get_data_path")
    @patch("features.feature_extractors.pd.read_csv")
    def test_extract_demographics(self, mock_read_csv, mock_get_path, mock_load_cfg):
        # --- Setup Mocks ---
        # Mock get_data_path to return specific paths
        def get_path_side_effect(data_type, dataset, config):
            if dataset == "patient_data":
                return "mock/processed/patient_data.csv"
            if dataset == "admission_data":
                return "mock/processed/admission_data.csv"
            return f"mock/{data_type}/{dataset}"

        mock_get_path.side_effect = get_path_side_effect

        # Mock dataframes returned by pd.read_csv
        mock_patients = pd.DataFrame(
            {
                "subject_id": [1, 2],
                "source": ["mimic_iii"] * 2,
                "age": [70, 45],
                "gender": ["F", "M"],
                "dod": [pd.NaT] * 2,
            }
        )
        mock_admissions = pd.DataFrame(
            {
                "subject_id": [1, 2],
                "hadm_id": [101, 102],
                "source": ["mimic_iii"] * 2,
                "admittime": pd.to_datetime(["2023-01-01", "2023-02-10"]),
                "dischtime": pd.to_datetime(["2023-01-05", "2023-02-15"]),
                "deathtime": [pd.NaT] * 2,
                "edregtime": [pd.NaT] * 2,
                "edouttime": [pd.NaT] * 2,
                "admission_type": ["EMERGENCY", "ELECTIVE"],
                "insurance": ["Medicare", "Private"],
                "marital_status": ["WIDOWED", "MARRIED"],
                "ethnicity": ["WHITE", "ASIAN"],  # Use ethnicity or race
            }
        )

        def read_csv_side_effect(path, **kwargs):
            if path == "mock/processed/patient_data.csv":
                return mock_patients
            if path == "mock/processed/admission_data.csv":
                return mock_admissions
            return pd.DataFrame()  # Return empty for other unexpected calls

        mock_read_csv.side_effect = read_csv_side_effect

        # --- Instantiate and Run ---
        extractor = DemographicFeatureExtractor(config=MOCK_CONFIG)
        features = extractor.extract()

        # --- Assertions ---
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 2)
        self.assertIn("subject_id", features.columns)
        self.assertIn("hadm_id", features.columns)
        self.assertIn("age", features.columns)
        # Check for actual dummy variable names (case-sensitive based on original data)
        self.assertIn("gender_F", features.columns)
        self.assertIn("gender_M", features.columns)
        self.assertIn("age_group_65-100", features.columns)
        self.assertIn("admission_type_EMERGENCY", features.columns)
        self.assertIn("insurance_Medicare", features.columns)
        self.assertIn("marital_status_WIDOWED", features.columns)
        self.assertIn("ethnicity_WHITE", features.columns)

        # Check values
        self.assertEqual(features.loc[features["subject_id"] == 1, "age"].iloc[0], 70)
        self.assertEqual(
            features.loc[features["subject_id"] == 1, "gender_F"].iloc[0], 1
        )
        self.assertEqual(
            features.loc[features["subject_id"] == 2, "gender_M"].iloc[0], 1
        )
        self.assertEqual(
            features.loc[features["subject_id"] == 1, "age_group_65-100"].iloc[0], 1
        )
        self.assertEqual(
            features.loc[features["subject_id"] == 2, "age_group_18-65"].iloc[0], 1
        )  # Check other age group


class TestClinicalFeatureExtractor(unittest.TestCase):

    @patch("utils.config.load_config", return_value=MOCK_CONFIG)
    @patch(
        "utils.config.load_mappings", return_value=MOCK_MAPPINGS
    )  # Mock mappings load
    @patch("features.feature_extractors.get_data_path")
    @patch("features.feature_extractors.pd.read_csv")
    @patch("features.feature_extractors.os.path.exists")  # Mock os.path.exists
    def test_extract_clinical_features(
        self, mock_exists, mock_read_csv, mock_get_path, mock_load_map, mock_load_cfg
    ):
        # --- Setup Mocks ---
        mock_exists.return_value = True  # Assume all raw files exist

        # Mock get_data_path
        def get_path_side_effect(data_type, dataset, config):
            if data_type == "processed":
                if dataset == "admission_data":
                    return "mock/processed/admission_data.csv"
                if dataset == "icu_data":
                    return "mock/processed/icu_data.csv"
            elif data_type == "raw":
                if dataset == "mimic_iii":
                    return "mock/raw/mimic_iii"
            return f"mock/{data_type}/{dataset}"  # Fallback

        mock_get_path.side_effect = get_path_side_effect

        # Mock dataframes
        mock_admissions = pd.DataFrame({"subject_id": [1], "hadm_id": [101]})
        mock_icustays = pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [101],
                "stay_id": [1001],
                "intime": [datetime(2023, 1, 1, 12, 0, 0)],
                "outtime": [datetime(2023, 1, 3, 12, 0, 0)],
            }
        )
        mock_labevents = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [101, 101],
                "itemid": [50809, 50971],  # Glucose, Potassium
                "charttime": [
                    datetime(2023, 1, 1, 14, 0, 0),
                    datetime(2023, 1, 1, 16, 0, 0),
                ],
                "valuenum": [120.0, 4.5],
            }
        )
        mock_d_labitems = pd.DataFrame(
            {"itemid": [50809, 50971], "label": ["Glucose", "Potassium"]}
        )
        mock_chartevents = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [101, 101],
                "stay_id": [1001, 1001],
                "itemid": [211, 51],  # HR, SBP
                "charttime": [
                    datetime(2023, 1, 1, 13, 0, 0),
                    datetime(2023, 1, 1, 13, 5, 0),
                ],
                "valuenum": [80.0, 120.0],
            }
        )
        mock_d_items = pd.DataFrame(
            {"itemid": [211, 51], "label": ["Heart Rate", "Arterial BP Systolic"]}
        )

        # This side effect function needs os imported in this test file
        def read_csv_side_effect(path, **kwargs):
            if path == "mock/processed/admission_data.csv":
                return mock_admissions
            if path == "mock/processed/icu_data.csv":
                return mock_icustays
            # Use os.path.join here
            if path == os.path.join("mock/raw/mimic_iii", "LABEVENTS.csv"):
                return mock_labevents
            if path == os.path.join("mock/raw/mimic_iii", "D_LABITEMS.csv"):
                return mock_d_labitems
            if path == os.path.join("mock/raw/mimic_iii", "CHARTEVENTS.csv"):
                return mock_chartevents
            if path == os.path.join("mock/raw/mimic_iii", "D_ITEMS.csv"):
                return mock_d_items
            print(f"Warning: Unmocked read_csv call for path: {path}")  # Debugging line
            return pd.DataFrame()

        mock_read_csv.side_effect = read_csv_side_effect

        # --- Instantiate and Run ---
        extractor = ClinicalFeatureExtractor(config=MOCK_CONFIG)
        features = extractor.extract()

        # --- Assertions ---
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 1)  # Should have one row per ICU stay
        self.assertIn("subject_id", features.columns)
        self.assertIn("hadm_id", features.columns)
        self.assertIn("stay_id", features.columns)
        # Check for sequence lab features
        self.assertIn("lab_Glucose_sequence_data", features.columns)
        self.assertIn("lab_Potassium_sequence_data", features.columns)
        # Check for sequence vital features (standardized names might differ slightly, adjust if needed)
        self.assertIn(
            "vital_heart_rate_sequence_data", features.columns
        )  # Corrected name
        self.assertIn(
            "vital_systolic_bp_sequence_data", features.columns
        )  # Corrected name

        # Check content of sequence data (optional, more complex)
        # Example: Check if the sequence contains the expected number of tuples (time, value)
        glucose_seq = features.loc[0, "lab_Glucose_sequence_data"]
        self.assertIsInstance(glucose_seq, list)
        if glucose_seq:  # Check if not empty list or NaN
            # Check the inner structure: list containing tuples
            self.assertIsInstance(glucose_seq[0], list)
            if glucose_seq[0]:  # Check if the inner list is not empty
                self.assertIsInstance(glucose_seq[0][0], tuple)
                self.assertEqual(len(glucose_seq[0][0]), 2)  # Should be (time, value)

        hr_seq = features.loc[0, "vital_heart_rate_sequence_data"]  # Corrected name
        self.assertIsInstance(hr_seq, list)
        if hr_seq:
            # Check the inner structure: list containing tuples
            self.assertIsInstance(hr_seq[0], list)
            if hr_seq[0]:  # Check if the inner list is not empty
                self.assertIsInstance(hr_seq[0][0], tuple)
                self.assertEqual(len(hr_seq[0][0]), 2)

    @patch("utils.config.load_config", return_value=MOCK_CONFIG)
    @patch("utils.config.load_mappings", return_value={})  # Return empty mappings
    @patch("features.feature_extractors.get_data_path")
    @patch("features.feature_extractors.pd.read_csv")
    @patch("features.feature_extractors.os.path.exists")
    def test_extract_clinical_features_no_mappings(
        self, mock_exists, mock_read_csv, mock_get_path, mock_load_map, mock_load_cfg
    ):
        # --- Setup Mocks (similar to test_extract_clinical_features) ---
        mock_exists.return_value = True

        # Mock get_data_path (copy from previous test or simplify if needed)
        def get_path_side_effect(data_type, dataset, config):
            if data_type == "processed":
                if dataset == "admission_data":
                    return "mock/processed/admission_data.csv"
                if dataset == "icu_data":
                    return "mock/processed/icu_data.csv"
            elif data_type == "raw":
                if dataset == "mimic_iii":
                    return "mock/raw/mimic_iii"
            return f"mock/{data_type}/{dataset}"

        mock_get_path.side_effect = get_path_side_effect

        # Mock dataframes (copy from previous test)
        mock_admissions = pd.DataFrame({"subject_id": [1], "hadm_id": [101]})
        mock_icustays = pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [101],
                "stay_id": [1001],
                "intime": [datetime(2023, 1, 1, 12, 0, 0)],
                "outtime": [datetime(2023, 1, 3, 12, 0, 0)],
            }
        )
        mock_labevents = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [101, 101],
                "itemid": [50809, 50971],
                "charttime": [
                    datetime(2023, 1, 1, 14, 0, 0),
                    datetime(2023, 1, 1, 16, 0, 0),
                ],
                "valuenum": [120.0, 4.5],
            }
        )
        mock_d_labitems = pd.DataFrame(
            {"itemid": [50809, 50971], "label": ["Glucose", "Potassium"]}
        )
        mock_chartevents = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [101, 101],
                "stay_id": [1001, 1001],
                "itemid": [211, 51],
                "charttime": [
                    datetime(2023, 1, 1, 13, 0, 0),
                    datetime(2023, 1, 1, 13, 5, 0),
                ],
                "valuenum": [80.0, 120.0],
            }
        )
        mock_d_items = pd.DataFrame(
            {"itemid": [211, 51], "label": ["Heart Rate", "Arterial BP Systolic"]}
        )

        def read_csv_side_effect(path, **kwargs):
            if path == "mock/processed/admission_data.csv":
                return mock_admissions
            if path == "mock/processed/icu_data.csv":
                return mock_icustays
            if path == os.path.join("mock/raw/mimic_iii", "LABEVENTS.csv"):
                return mock_labevents
            if path == os.path.join("mock/raw/mimic_iii", "D_LABITEMS.csv"):
                return mock_d_labitems
            if path == os.path.join("mock/raw/mimic_iii", "CHARTEVENTS.csv"):
                return mock_chartevents
            if path == os.path.join("mock/raw/mimic_iii", "D_ITEMS.csv"):
                return mock_d_items
            return pd.DataFrame()

        mock_read_csv.side_effect = read_csv_side_effect

        # --- Instantiate and Run ---
        extractor = ClinicalFeatureExtractor(config=MOCK_CONFIG)
        features = extractor.extract()

        # --- Assertions ---
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 1)
        self.assertIn("subject_id", features.columns)
        self.assertIn("hadm_id", features.columns)
        self.assertIn("stay_id", features.columns)
        # Assert that NO mapped features were created
        self.assertNotIn("glucose_mean", features.columns)
        self.assertNotIn("potassium_mean", features.columns)
        self.assertNotIn("heart_rate_mean", features.columns)
        self.assertNotIn("systolic_bp_mean", features.columns)
        self.assertNotIn("heart_rate_std", features.columns)
        self.assertNotIn("systolic_bp_std", features.columns)

    @patch("utils.config.load_config", return_value=MOCK_CONFIG)
    @patch(
        "utils.config.load_mappings",
        return_value={"vital_signs": MOCK_MAPPINGS["vital_signs"]},
    )  # Missing lab_tests
    @patch("features.feature_extractors.get_data_path")
    @patch("features.feature_extractors.pd.read_csv")
    @patch("features.feature_extractors.os.path.exists")
    def test_extract_clinical_features_missing_lab_mapping(
        self, mock_exists, mock_read_csv, mock_get_path, mock_load_map, mock_load_cfg
    ):
        # --- Setup Mocks (similar setup) ---
        mock_exists.return_value = True

        def get_path_side_effect(data_type, dataset, config):
            if data_type == "processed":
                if dataset == "admission_data":
                    return "mock/processed/admission_data.csv"
                if dataset == "icu_data":
                    return "mock/processed/icu_data.csv"
            elif data_type == "raw":
                if dataset == "mimic_iii":
                    return "mock/raw/mimic_iii"
            return f"mock/{data_type}/{dataset}"

        mock_get_path.side_effect = get_path_side_effect
        mock_admissions = pd.DataFrame({"subject_id": [1], "hadm_id": [101]})
        mock_icustays = pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [101],
                "stay_id": [1001],
                "intime": [datetime(2023, 1, 1, 12, 0, 0)],
                "outtime": [datetime(2023, 1, 3, 12, 0, 0)],
            }
        )
        mock_labevents = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [101, 101],
                "itemid": [50809, 50971],
                "charttime": [
                    datetime(2023, 1, 1, 14, 0, 0),
                    datetime(2023, 1, 1, 16, 0, 0),
                ],
                "valuenum": [120.0, 4.5],
            }
        )
        mock_d_labitems = pd.DataFrame(
            {"itemid": [50809, 50971], "label": ["Glucose", "Potassium"]}
        )
        mock_chartevents = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [101, 101],
                "stay_id": [1001, 1001],
                "itemid": [211, 51],
                "charttime": [
                    datetime(2023, 1, 1, 13, 0, 0),
                    datetime(2023, 1, 1, 13, 5, 0),
                ],
                "valuenum": [80.0, 120.0],
            }
        )
        mock_d_items = pd.DataFrame(
            {"itemid": [211, 51], "label": ["Heart Rate", "Arterial BP Systolic"]}
        )

        def read_csv_side_effect(path, **kwargs):
            if path == "mock/processed/admission_data.csv":
                return mock_admissions
            if path == "mock/processed/icu_data.csv":
                return mock_icustays
            if path == os.path.join("mock/raw/mimic_iii", "LABEVENTS.csv"):
                return mock_labevents
            if path == os.path.join("mock/raw/mimic_iii", "D_LABITEMS.csv"):
                return mock_d_labitems
            if path == os.path.join("mock/raw/mimic_iii", "CHARTEVENTS.csv"):
                return mock_chartevents
            if path == os.path.join("mock/raw/mimic_iii", "D_ITEMS.csv"):
                return mock_d_items
            return pd.DataFrame()

        mock_read_csv.side_effect = read_csv_side_effect

        # --- Instantiate and Run ---
        extractor = ClinicalFeatureExtractor(config=MOCK_CONFIG)
        features = extractor.extract()

        # --- Assertions ---
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 1)
        # Assert that lab sequence features are missing, but vital sequence features are present
        self.assertNotIn("lab_Glucose_sequence_data", features.columns)
        self.assertNotIn("lab_Potassium_sequence_data", features.columns)
        self.assertIn(
            "vital_heart_rate_sequence_data", features.columns
        )  # Corrected name
        self.assertIn(
            "vital_systolic_bp_sequence_data", features.columns
        )  # Corrected name

    @patch("utils.config.load_config", return_value=MOCK_CONFIG)
    @patch(
        "utils.config.load_mappings",
        return_value={"lab_tests": MOCK_MAPPINGS["lab_tests"]},
    )  # Missing vital_signs
    @patch("features.feature_extractors.get_data_path")
    @patch("features.feature_extractors.pd.read_csv")
    @patch("features.feature_extractors.os.path.exists")
    def test_extract_clinical_features_missing_vital_mapping(
        self, mock_exists, mock_read_csv, mock_get_path, mock_load_map, mock_load_cfg
    ):
        # --- Setup Mocks (similar setup) ---
        mock_exists.return_value = True

        def get_path_side_effect(data_type, dataset, config):
            if data_type == "processed":
                if dataset == "admission_data":
                    return "mock/processed/admission_data.csv"
                if dataset == "icu_data":
                    return "mock/processed/icu_data.csv"
            elif data_type == "raw":
                if dataset == "mimic_iii":
                    return "mock/raw/mimic_iii"
            return f"mock/{data_type}/{dataset}"

        mock_get_path.side_effect = get_path_side_effect
        mock_admissions = pd.DataFrame({"subject_id": [1], "hadm_id": [101]})
        mock_icustays = pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [101],
                "stay_id": [1001],
                "intime": [datetime(2023, 1, 1, 12, 0, 0)],
                "outtime": [datetime(2023, 1, 3, 12, 0, 0)],
            }
        )
        mock_labevents = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [101, 101],
                "itemid": [50809, 50971],
                "charttime": [
                    datetime(2023, 1, 1, 14, 0, 0),
                    datetime(2023, 1, 1, 16, 0, 0),
                ],
                "valuenum": [120.0, 4.5],
            }
        )
        mock_d_labitems = pd.DataFrame(
            {"itemid": [50809, 50971], "label": ["Glucose", "Potassium"]}
        )
        mock_chartevents = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [101, 101],
                "stay_id": [1001, 1001],
                "itemid": [211, 51],
                "charttime": [
                    datetime(2023, 1, 1, 13, 0, 0),
                    datetime(2023, 1, 1, 13, 5, 0),
                ],
                "valuenum": [80.0, 120.0],
            }
        )
        mock_d_items = pd.DataFrame(
            {"itemid": [211, 51], "label": ["Heart Rate", "Arterial BP Systolic"]}
        )

        def read_csv_side_effect(path, **kwargs):
            if path == "mock/processed/admission_data.csv":
                return mock_admissions
            if path == "mock/processed/icu_data.csv":
                return mock_icustays
            if path == os.path.join("mock/raw/mimic_iii", "LABEVENTS.csv"):
                return mock_labevents
            if path == os.path.join("mock/raw/mimic_iii", "D_LABITEMS.csv"):
                return mock_d_labitems
            if path == os.path.join("mock/raw/mimic_iii", "CHARTEVENTS.csv"):
                return mock_chartevents
            if path == os.path.join("mock/raw/mimic_iii", "D_ITEMS.csv"):
                return mock_d_items
            return pd.DataFrame()

        mock_read_csv.side_effect = read_csv_side_effect

        # --- Instantiate and Run ---
        extractor = ClinicalFeatureExtractor(config=MOCK_CONFIG)
        features = extractor.extract()

        # --- Assertions ---
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 1)
        # Assert that vital sequence features are missing, but lab sequence features are present
        self.assertIn("lab_Glucose_sequence_data", features.columns)
        self.assertIn("lab_Potassium_sequence_data", features.columns)
        self.assertNotIn(
            "vital_heart_rate_sequence_data", features.columns
        )  # Corrected name
        self.assertNotIn(
            "vital_systolic_bp_sequence_data", features.columns
        )  # Corrected name

    @patch("utils.config.load_config", return_value=MOCK_CONFIG)
    @patch(
        "utils.config.load_mappings", return_value=MOCK_MAPPINGS
    )  # Use full mappings
    @patch("features.feature_extractors.get_data_path")
    @patch("features.feature_extractors.pd.read_csv")
    @patch("features.feature_extractors.os.path.exists")
    def test_extract_clinical_features_unmapped_itemid(
        self, mock_exists, mock_read_csv, mock_get_path, mock_load_map, mock_load_cfg
    ):
        # --- Setup Mocks ---
        mock_exists.return_value = True

        def get_path_side_effect(data_type, dataset, config):
            if data_type == "processed":
                if dataset == "admission_data":
                    return "mock/processed/admission_data.csv"
                if dataset == "icu_data":
                    return "mock/processed/icu_data.csv"
            elif data_type == "raw":
                if dataset == "mimic_iii":
                    return "mock/raw/mimic_iii"
            return f"mock/{data_type}/{dataset}"

        mock_get_path.side_effect = get_path_side_effect

        # Mock dataframes - Add an unmapped lab itemid (99999)
        mock_admissions = pd.DataFrame({"subject_id": [1], "hadm_id": [101]})
        mock_icustays = pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [101],
                "stay_id": [1001],
                "intime": [datetime(2023, 1, 1, 12, 0, 0)],
                "outtime": [datetime(2023, 1, 3, 12, 0, 0)],
            }
        )
        mock_labevents = pd.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "hadm_id": [101, 101, 101],
                "itemid": [50809, 50971, 99999],  # Glucose, Potassium, Unmapped
                "charttime": [
                    datetime(2023, 1, 1, 14, 0, 0),
                    datetime(2023, 1, 1, 16, 0, 0),
                    datetime(2023, 1, 1, 17, 0, 0),
                ],
                "valuenum": [120.0, 4.5, 10.0],
            }
        )
        mock_d_labitems = pd.DataFrame(
            {
                "itemid": [50809, 50971, 99999],
                "label": ["Glucose", "Potassium", "Unknown"],
            }
        )
        mock_chartevents = pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [101],
                "stay_id": [1001],
                "itemid": [211],
                "charttime": [datetime(2023, 1, 1, 13, 0, 0)],
                "valuenum": [80.0],
            }
        )
        mock_d_items = pd.DataFrame({"itemid": [211], "label": ["Heart Rate"]})

        def read_csv_side_effect(path, **kwargs):
            if path == "mock/processed/admission_data.csv":
                return mock_admissions
            if path == "mock/processed/icu_data.csv":
                return mock_icustays
            if path == os.path.join("mock/raw/mimic_iii", "LABEVENTS.csv"):
                return mock_labevents
            if path == os.path.join("mock/raw/mimic_iii", "D_LABITEMS.csv"):
                return mock_d_labitems
            if path == os.path.join("mock/raw/mimic_iii", "CHARTEVENTS.csv"):
                return mock_chartevents
            if path == os.path.join("mock/raw/mimic_iii", "D_ITEMS.csv"):
                return mock_d_items
            return pd.DataFrame()

        mock_read_csv.side_effect = read_csv_side_effect

        # --- Instantiate and Run ---
        extractor = ClinicalFeatureExtractor(config=MOCK_CONFIG)
        features = extractor.extract()

        # --- Assertions ---
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 1)
        # Assert that mapped sequence features are present
        self.assertIn("lab_Glucose_sequence_data", features.columns)
        self.assertIn("lab_Potassium_sequence_data", features.columns)
        self.assertIn(
            "vital_heart_rate_sequence_data", features.columns
        )  # Corrected name
        # Assert that the unmapped itemid did not create a sequence feature column
        self.assertNotIn(
            "lab_Unknown_sequence_data", features.columns
        )  # Based on label in mock D_LABITEMS
        self.assertNotIn("lab_99999_sequence_data", features.columns)  # Based on itemid


# TODO: Add tests for DiagnosisFeatureExtractor
# - Mock diagnoses_icd.csv
# - Test _process_diagnosis_data, _get_icd9_category
# - Test cases with V codes, E codes, different numeric ranges, invalid codes, missing mappings


class TestDiagnosisFeatureExtractor(unittest.TestCase):

    @patch("utils.config.load_config", return_value=MOCK_CONFIG)
    @patch(
        "utils.config.load_mappings", return_value=MOCK_MAPPINGS
    )  # Use mock mappings
    @patch("features.feature_extractors.get_data_path")
    @patch("features.feature_extractors.pd.read_csv")
    @patch("features.feature_extractors.os.path.exists")
    def test_extract_diagnosis_features_basic(
        self, mock_exists, mock_read_csv, mock_get_path, mock_load_map, mock_load_cfg
    ):
        # --- Setup Mocks ---
        mock_exists.return_value = True

        def get_path_side_effect(data_type, dataset, config):
            if data_type == "processed" and dataset == "admission_data":
                return "mock/processed/admission_data.csv"
            elif data_type == "raw" and dataset == "mimic_iii":
                return "mock/raw/mimic_iii"
            return f"mock/{data_type}/{dataset}"

        mock_get_path.side_effect = get_path_side_effect

        mock_admissions = pd.DataFrame({"subject_id": [1, 2], "hadm_id": [101, 102]})
        mock_diagnoses = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 2],
                "hadm_id": [101, 101, 102, 102],
                "seq_num": [1, 2, 1, 2],
                "icd9_code": [
                    "042",
                    "100",
                    "V3000",
                    "E8800",
                ],  # HIV, TB, Single liveborn, Fall
                "icd_version": [9, 9, 9, 9],
            }
        )

        def read_csv_side_effect(path, **kwargs):
            if path == "mock/processed/admission_data.csv":
                return mock_admissions
            # Use os.path.join for OS compatibility
            if path == os.path.join("mock/raw/mimic_iii", "DIAGNOSES_ICD.csv"):
                # Filter by version 9 as the extractor does
                return mock_diagnoses[mock_diagnoses["icd_version"] == 9]
            print(f"Warning: Unmocked read_csv call for path: {path}")
            return pd.DataFrame()

        mock_read_csv.side_effect = read_csv_side_effect

        # --- Instantiate and Run ---
        extractor = DiagnosisFeatureExtractor(config=MOCK_CONFIG)
        features = extractor.extract()

        # --- Assertions ---
        self.assertIsInstance(features, pd.DataFrame)
        # Should aggregate to one row per hadm_id
        self.assertEqual(len(features), 2)
        self.assertIn("subject_id", features.columns)
        self.assertIn("hadm_id", features.columns)
        # Check for expected category columns based on MOCK_MAPPINGS
        self.assertIn(
            "diag_category_infectious_diseases", features.columns
        )  # Check sanitized name from "Infectious Diseases"
        # Check other potential columns based on processing logic (e.g., count)
        self.assertIn("diag_count", features.columns)

        # Check values for hadm_id 101 (Infectious=1, count=2)
        hadm101 = features[features["hadm_id"] == 101].iloc[0]
        self.assertEqual(
            hadm101["diag_category_infectious_diseases"], 1
        )  # Check sanitized name
        self.assertEqual(hadm101["diag_count"], 2)

        # Check values for hadm_id 102 (Infectious=0, count=2 - V/E codes might be counted)
        hadm102 = features[features["hadm_id"] == 102].iloc[0]
        self.assertEqual(
            hadm102["diag_category_infectious_diseases"], 0
        )  # Check sanitized name
        # Assuming V/E codes are counted by default implementation
        self.assertEqual(hadm102["diag_count"], 2)

    @patch("utils.config.load_config", return_value=MOCK_CONFIG)
    @patch(
        "utils.config.load_mappings", return_value={"lab_tests": {}, "vital_signs": {}}
    )  # Missing icd9_categories
    @patch("features.feature_extractors.get_data_path")
    @patch("features.feature_extractors.pd.read_csv")
    @patch("features.feature_extractors.os.path.exists")
    def test_extract_diagnosis_features_missing_mapping(
        self, mock_exists, mock_read_csv, mock_get_path, mock_load_map, mock_load_cfg
    ):
        # --- Setup Mocks (similar setup) ---
        mock_exists.return_value = True

        def get_path_side_effect(data_type, dataset, config):
            if data_type == "processed" and dataset == "admission_data":
                return "mock/processed/admission_data.csv"
            elif data_type == "raw" and dataset == "mimic_iii":
                return "mock/raw/mimic_iii"
            return f"mock/{data_type}/{dataset}"

        mock_get_path.side_effect = get_path_side_effect
        mock_admissions = pd.DataFrame({"subject_id": [1], "hadm_id": [101]})
        mock_diagnoses = pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [101],
                "seq_num": [1],
                "icd9_code": ["100"],
                "icd_version": [9],
            }
        )

        def read_csv_side_effect(path, **kwargs):
            if path == "mock/processed/admission_data.csv":
                return mock_admissions
            if path == os.path.join("mock/raw/mimic_iii", "DIAGNOSES_ICD.csv"):
                return mock_diagnoses[mock_diagnoses["icd_version"] == 9]
            return pd.DataFrame()

        mock_read_csv.side_effect = read_csv_side_effect

        # --- Instantiate and Run ---
        extractor = DiagnosisFeatureExtractor(config=MOCK_CONFIG)
        features = extractor.extract()

        # --- Assertions ---
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 1)
        # Check that NO category columns were created
        self.assertNotIn("diag_category_infectious", features.columns)
        # Check that basic columns and count still exist
        self.assertIn("subject_id", features.columns)
        self.assertIn("hadm_id", features.columns)
        self.assertIn("diag_count", features.columns)
        self.assertEqual(
            features.iloc[0]["diag_count"], 1
        )  # Should be 1 based on input data, even if mappings are missing

    @patch("utils.config.load_config", return_value=MOCK_CONFIG)
    @patch("utils.config.load_mappings", return_value=MOCK_MAPPINGS)
    @patch("features.feature_extractors.get_data_path")
    @patch("features.feature_extractors.pd.read_csv")
    @patch("features.feature_extractors.os.path.exists")
    def test_extract_diagnosis_features_empty_diagnoses(
        self, mock_exists, mock_read_csv, mock_get_path, mock_load_map, mock_load_cfg
    ):
        # --- Setup Mocks ---
        mock_exists.return_value = True

        def get_path_side_effect(data_type, dataset, config):
            if data_type == "processed" and dataset == "admission_data":
                return "mock/processed/admission_data.csv"
            elif data_type == "raw" and dataset == "mimic_iii":
                return "mock/raw/mimic_iii"
            return f"mock/{data_type}/{dataset}"

        mock_get_path.side_effect = get_path_side_effect

        mock_admissions = pd.DataFrame({"subject_id": [1], "hadm_id": [101]})
        # Return empty dataframe for diagnoses
        mock_empty_diagnoses = pd.DataFrame(
            columns=["subject_id", "hadm_id", "seq_num", "icd9_code", "icd_version"]
        )

        def read_csv_side_effect(path, **kwargs):
            if path == "mock/processed/admission_data.csv":
                return mock_admissions
            if path == os.path.join("mock/raw/mimic_iii", "DIAGNOSES_ICD.csv"):
                return mock_empty_diagnoses  # Return empty
            return pd.DataFrame()

        mock_read_csv.side_effect = read_csv_side_effect

        # --- Instantiate and Run ---
        extractor = DiagnosisFeatureExtractor(config=MOCK_CONFIG)
        features = extractor.extract()

        # --- Assertions ---
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 1)  # Should still have the admission row
        # Check that category columns exist but are 0
        self.assertIn(
            "diag_category_infectious_diseases", features.columns
        )  # Check sanitized name
        self.assertEqual(
            features.iloc[0]["diag_category_infectious_diseases"], 0
        )  # Check sanitized name
        # Check count is 0
        self.assertIn("diag_count", features.columns)
        self.assertEqual(features.iloc[0]["diag_count"], 0)


if __name__ == "__main__":
    unittest.main()
    unittest.main()
