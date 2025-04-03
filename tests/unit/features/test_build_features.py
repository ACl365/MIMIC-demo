"""
Unit tests for the build_features module.
"""

import unittest
from unittest.mock import patch  # Removed MagicMock, os

import numpy as np
import pandas as pd

from features.build_features import _combine_features  # Corrected import


class TestCombineFeatures(unittest.TestCase):
    """
    Test cases for the _combine_features function.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a mock config
        self.config = {
            "data": {
                "processed": {
                    "base_path": "data/processed/",
                    "admission_data": "data/processed/admission_data.csv",
                }
            },
            "features": {
                "demographic": {"include": True},
                "vitals": {"include": True},
                "lab_values": {"include": True},
                "diagnoses": {"include": True},
            },
        }

        # Create sample dataframes
        self.admissions = pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "hadm_id": [100, 200, 300],
                "los_days": [5.2, 3.1, 7.5],
                "hospital_death": [False, True, False],
                "readmission_30day": [True, False, False],
                "readmission_90day": [True, False, True],
            }
        )

        self.demographic_features = pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "hadm_id": [100, 200, 300],
                "age": [65, 45, 72],
                "gender_M": [1, 1, 0],  # Keep original case for mock data loading
                "gender_F": [0, 0, 1],  # Keep original case for mock data loading
            }
        )

        self.clinical_features = pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "hadm_id": [100, 200, 300],
                "stay_id": [1000, 2000, 3000],
                "heart_rate_mean": [80.5, 92.3, 75.1],
                "sbp_mean": [120.5, 135.2, 110.8],
                "glucose_mean": [110.2, 180.5, 95.3],
            }
        )

        self.diagnosis_features = pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "hadm_id": [100, 200, 300],
                "infectious": [0, 1, 0],
                "circulatory": [1, 0, 1],
                "respiratory": [0, 1, 0],
            }
        )

    @patch("features.build_features.pd.read_csv")  # Corrected mock path
    @patch("features.build_features.os.path.exists")  # Corrected mock path
    @patch("features.build_features.get_data_path")  # Corrected mock path
    def test_combine_features_all_available(
        self, mock_get_data_path, mock_exists, mock_read_csv
    ):
        """
        Test _combine_features when all feature types are available.
        """
        # Configure mocks
        # Use only hashable types (strings) for the dictionary key
        mock_get_data_path.side_effect = lambda data_type, dataset, config: {
            ("processed", "admission_data"): "data/processed/admission_data.csv",
            ("processed", "base_path"): "data/processed/",
        }[
            (data_type, dataset)
        ]  # Lookup using only data_type and dataset

        mock_exists.return_value = True

        mock_read_csv.side_effect = lambda path, **kwargs: {
            "data/processed/admission_data.csv": self.admissions,
            "data/processed/demographic_features.csv": self.demographic_features,
            "data/processed/clinical_features.csv": self.clinical_features,
            "data/processed/diagnosis_features.csv": self.diagnosis_features,
        }[path]

        # Call the function
        result = _combine_features(self.config)

        # Assertions
        self.assertEqual(len(result), 3)  # Should have 3 rows
        self.assertIn("subject_id", result.columns)
        self.assertIn("hadm_id", result.columns)
        self.assertIn("age", result.columns)
        self.assertIn("heart_rate_mean", result.columns)
        self.assertIn("infectious", result.columns)

        # Check that the merge was done correctly
        self.assertEqual(result.loc[0, "age"], 65)
        self.assertEqual(result.loc[1, "heart_rate_mean"], 92.3)
        self.assertEqual(result.loc[2, "circulatory"], 1)

    @patch("features.build_features.pd.read_csv")  # Corrected mock path
    @patch("features.build_features.os.path.exists")  # Corrected mock path
    @patch("features.build_features.get_data_path")  # Corrected mock path
    def test_combine_features_missing_clinical(
        self, mock_get_data_path, mock_exists, mock_read_csv
    ):
        """
        Test _combine_features when clinical features are missing.
        """
        # Configure mocks
        # Use only hashable types (strings) for the dictionary key
        mock_get_data_path.side_effect = lambda data_type, dataset, config: {
            ("processed", "admission_data"): "data/processed/admission_data.csv",
            ("processed", "base_path"): "data/processed/",
        }[
            (data_type, dataset)
        ]  # Lookup using only data_type and dataset

        # Clinical features file doesn't exist
        mock_exists.side_effect = (
            lambda path: path != "data/processed/clinical_features.csv"
        )

        mock_read_csv.side_effect = lambda path, **kwargs: {
            "data/processed/admission_data.csv": self.admissions,
            "data/processed/demographic_features.csv": self.demographic_features,
            "data/processed/diagnosis_features.csv": self.diagnosis_features,
        }[path]

        # Call the function
        result = _combine_features(self.config)

        # Assertions
        self.assertEqual(len(result), 3)  # Should have 3 rows
        self.assertIn("subject_id", result.columns)
        self.assertIn("hadm_id", result.columns)
        self.assertIn("age", result.columns)
        self.assertNotIn(
            "heart_rate_mean", result.columns
        )  # Clinical features should be missing
        self.assertIn("infectious", result.columns)

    @patch("features.build_features.pd.read_csv")  # Corrected mock path
    @patch("features.build_features.os.path.exists")  # Corrected mock path
    @patch("features.build_features.get_data_path")  # Corrected mock path
    def test_imputation_strategy(self, mock_get_data_path, mock_exists, mock_read_csv):
        """
        Test the imputation strategy in _combine_features.
        """
        # Configure mocks
        # Use only hashable types (strings) for the dictionary key
        mock_get_data_path.side_effect = lambda data_type, dataset, config: {
            ("processed", "admission_data"): "data/processed/admission_data.csv",
            ("processed", "base_path"): "data/processed/",
        }[
            (data_type, dataset)
        ]  # Lookup using only data_type and dataset

        mock_exists.return_value = True

        # Create data with missing values
        demographic_with_na = self.demographic_features.copy()
        demographic_with_na.loc[1, "age"] = np.nan

        clinical_with_na = self.clinical_features.copy()
        clinical_with_na.loc[0, "heart_rate_mean"] = np.nan
        clinical_with_na.loc[2, "glucose_mean"] = np.nan

        mock_read_csv.side_effect = lambda path, **kwargs: {
            "data/processed/admission_data.csv": self.admissions,
            "data/processed/demographic_features.csv": demographic_with_na,
            "data/processed/clinical_features.csv": clinical_with_na,
            "data/processed/diagnosis_features.csv": self.diagnosis_features,
        }[path]

        # Call the function
        result = _combine_features(self.config)

        # Assertions
        self.assertEqual(len(result), 3)  # Should have 3 rows

        # Check that missing values were imputed correctly
        # Age (clinical measurement) should be imputed with median
        self.assertFalse(pd.isna(result.loc[1, "age"]))

        # heart_rate_mean (clinical measurement) should be imputed with median
        self.assertFalse(pd.isna(result.loc[0, "heart_rate_mean"]))

        # glucose_mean (clinical measurement) should be imputed with median
        self.assertFalse(pd.isna(result.loc[2, "glucose_mean"]))

        # Binary features (gender_m, gender_f) should be imputed with 0
        # Use lowercase column names as they are converted in _combine_features
        self.assertEqual(result["gender_m"].isna().sum(), 0)
        self.assertEqual(result["gender_f"].isna().sum(), 0)

    @patch("features.build_features.pd.read_csv")  # Corrected mock path
    @patch("features.build_features.os.path.exists")  # Corrected mock path
    @patch("features.build_features.get_data_path")  # Corrected mock path
    def test_combine_features_empty_clinical(
        self, mock_get_data_path, mock_exists, mock_read_csv
    ):
        """
        Test _combine_features when clinical features file exists but is empty.
        """
        # Configure mocks
        mock_get_data_path.side_effect = lambda data_type, dataset, config: {
            ("processed", "admission_data"): "data/processed/admission_data.csv",
            ("processed", "base_path"): "data/processed/",
        }[(data_type, dataset)]

        mock_exists.return_value = True  # All files exist

        # Create an empty clinical features DataFrame
        empty_clinical = pd.DataFrame(columns=self.clinical_features.columns)

        mock_read_csv.side_effect = lambda path, **kwargs: {
            "data/processed/admission_data.csv": self.admissions,
            "data/processed/demographic_features.csv": self.demographic_features,
            "data/processed/clinical_features.csv": empty_clinical,  # Return empty DF
            "data/processed/diagnosis_features.csv": self.diagnosis_features,
        }[path]

        # Call the function
        result = _combine_features(self.config)

        # Assertions
        self.assertEqual(len(result), 3)  # Should still have 3 rows from admissions
        self.assertIn("subject_id", result.columns)
        self.assertIn("hadm_id", result.columns)
        self.assertIn("age", result.columns)
        # Clinical features should NOT be present as the merge input was empty
        self.assertNotIn("heart_rate_mean", result.columns)
        self.assertNotIn("sbp_mean", result.columns)
        self.assertNotIn("glucose_mean", result.columns)
        self.assertIn(
            "infectious", result.columns
        )  # Diagnosis features should still be there


if __name__ == "__main__":
    unittest.main()
