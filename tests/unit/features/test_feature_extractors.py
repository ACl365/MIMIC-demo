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
    "icd9_categories": {"ranges": {"Infectious": [1, 139]}, "specific_codes": {}},
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
        # Check for aggregated lab features (mean is default in mock config)
        self.assertIn("glucose_mean", features.columns)
        self.assertIn("potassium_mean", features.columns)
        # Check for aggregated vital features (mean and std requested in mock config)
        self.assertIn("heart_rate_mean", features.columns)
        self.assertIn("systolic_bp_mean", features.columns)
        self.assertIn("heart_rate_std", features.columns)
        self.assertIn("systolic_bp_std", features.columns)

        # Check values (simple check, more rigorous checks needed for edge cases)
        self.assertAlmostEqual(features.loc[0, "glucose_mean"], 120.0)
        self.assertAlmostEqual(features.loc[0, "potassium_mean"], 4.5)
        self.assertAlmostEqual(features.loc[0, "heart_rate_mean"], 80.0)
        self.assertAlmostEqual(features.loc[0, "systolic_bp_mean"], 120.0)
        # Std requires more than one value, should be NaN or 0 depending on pandas version/ddof
        self.assertTrue(
            pd.isna(features.loc[0, "heart_rate_std"])
            or features.loc[0, "heart_rate_std"] == 0
        )


# TODO: Add tests for DiagnosisFeatureExtractor
# - Mock diagnoses_icd.csv
# - Test _process_diagnosis_data, _get_icd9_category
# - Test cases with V codes, E codes, different numeric ranges, invalid codes

if __name__ == "__main__":
    unittest.main()
