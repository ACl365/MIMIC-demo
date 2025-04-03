"""
Unit tests for the processors module.
"""

import unittest
from datetime import datetime  # Removed timedelta

# Removed unused unittest.mock imports
import pandas as pd

from data.processors import AdmissionProcessor

# Removed unused numpy import


class TestIdentifyReadmissions(unittest.TestCase):
    """
    Test cases for the _identify_readmissions function.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a sample admissions dataframe
        self.admissions = pd.DataFrame(
            {
                "subject_id": [1, 1, 1, 2, 2, 3],
                "hadm_id": [101, 102, 103, 201, 202, 301],
                "admittime": [
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 15),
                    datetime(2023, 3, 1),
                    datetime(2023, 2, 1),
                    datetime(2023, 4, 15),
                    datetime(2023, 3, 10),
                ],
                "dischtime": [
                    datetime(2023, 1, 5),
                    datetime(2023, 1, 20),
                    datetime(2023, 3, 10),
                    datetime(2023, 2, 10),
                    datetime(2023, 4, 25),
                    datetime(2023, 3, 15),
                ],
                "hospital_death": [False, False, True, False, False, False],
                # Add hospital_expire_flag if needed by _calculate_derived_features
                "hospital_expire_flag": [0, 0, 1, 0, 0, 0],
            }
        )

        # Sort by subject_id and admittime
        self.admissions = self.admissions.sort_values(["subject_id", "admittime"])

        # Create a processor instance
        # Mock config if processor uses it during init or methods under test
        with unittest.mock.patch("utils.config.load_config", return_value={}):
            self.processor = AdmissionProcessor()

    def test_identify_readmissions(self):
        """
        Test the _identify_readmissions function with a typical dataset.
        """
        # Call the function
        # Ensure derived features are calculated if needed by _identify_readmissions
        processed_admissions = self.processor._calculate_derived_features(
            self.admissions.copy()
        )
        result = self.processor._identify_readmissions(processed_admissions)

        # Assertions
        self.assertEqual(len(result), 6)  # Should have 6 rows

        # Check readmission flags
        # Patient 1: First admission (hadm_id 101) should have 30-day readmission
        self.assertTrue(
            result.loc[result["hadm_id"] == 101, "readmission_30day"].iloc[0]
        )
        self.assertTrue(
            result.loc[result["hadm_id"] == 101, "readmission_90day"].iloc[0]
        )

        # Patient 1: Second admission (hadm_id 102) should have 90-day readmission but not 30-day
        self.assertFalse(
            result.loc[result["hadm_id"] == 102, "readmission_30day"].iloc[0]
        )
        self.assertTrue(
            result.loc[result["hadm_id"] == 102, "readmission_90day"].iloc[0]
        )

        # Patient 1: Third admission (hadm_id 103) resulted in death, so no readmission
        self.assertFalse(
            result.loc[result["hadm_id"] == 103, "readmission_30day"].iloc[0]
        )
        self.assertFalse(
            result.loc[result["hadm_id"] == 103, "readmission_90day"].iloc[0]
        )

        # Patient 2: First admission (hadm_id 201) should not have 30-day readmission
        self.assertFalse(
            result.loc[result["hadm_id"] == 201, "readmission_30day"].iloc[0]
        )
        self.assertTrue(
            result.loc[result["hadm_id"] == 201, "readmission_90day"].iloc[0]
        )

        # Patient 3: Only one admission, so no readmission
        self.assertFalse(
            result.loc[result["hadm_id"] == 301, "readmission_30day"].iloc[0]
        )
        self.assertFalse(
            result.loc[result["hadm_id"] == 301, "readmission_90day"].iloc[0]
        )

    def test_days_to_readmission(self):
        """
        Test the calculation of days_to_readmission.
        """
        # Call the function
        processed_admissions = self.processor._calculate_derived_features(
            self.admissions.copy()
        )
        result = self.processor._identify_readmissions(processed_admissions)

        # Assertions
        # Patient 1: First admission to second admission
        days_to_readmission_1 = result.loc[
            result["hadm_id"] == 101, "days_to_readmission"
        ].iloc[0]
        expected_days_1 = (
            datetime(2023, 1, 15) - datetime(2023, 1, 5)
        ).total_seconds() / (24 * 60 * 60)
        self.assertAlmostEqual(days_to_readmission_1, expected_days_1, places=1)

        # Patient 1: Second admission to third admission
        days_to_readmission_2 = result.loc[
            result["hadm_id"] == 102, "days_to_readmission"
        ].iloc[0]
        expected_days_2 = (
            datetime(2023, 3, 1) - datetime(2023, 1, 20)
        ).total_seconds() / (24 * 60 * 60)
        self.assertAlmostEqual(days_to_readmission_2, expected_days_2, places=1)

        # Patient 2: First admission to second admission
        days_to_readmission_3 = result.loc[
            result["hadm_id"] == 201, "days_to_readmission"
        ].iloc[0]
        expected_days_3 = (
            datetime(2023, 4, 15) - datetime(2023, 2, 10)
        ).total_seconds() / (24 * 60 * 60)
        self.assertAlmostEqual(days_to_readmission_3, expected_days_3, places=1)

    def test_no_readmissions(self):
        """
        Test the function with a dataset that has no readmissions.
        """
        # Create a dataset with no readmissions (all different patients)
        admissions_no_readmissions = pd.DataFrame(
            {
                "subject_id": [1, 2, 3, 4, 5],
                "hadm_id": [101, 201, 301, 401, 501],
                "admittime": [
                    datetime(2023, 1, 1),
                    datetime(2023, 2, 1),
                    datetime(2023, 3, 1),
                    datetime(2023, 4, 1),
                    datetime(2023, 5, 1),
                ],
                "dischtime": [
                    datetime(2023, 1, 5),
                    datetime(2023, 2, 5),
                    datetime(2023, 3, 5),
                    datetime(2023, 4, 5),
                    datetime(2023, 5, 5),
                ],
                "hospital_death": [False, False, False, False, False],
                "hospital_expire_flag": [0, 0, 0, 0, 0],  # Add flag
            }
        )

        # Call the function
        processed_admissions = self.processor._calculate_derived_features(
            admissions_no_readmissions.copy()
        )
        result = self.processor._identify_readmissions(processed_admissions)

        # Assertions
        self.assertEqual(len(result), 5)  # Should have 5 rows

        # Check that no readmissions were identified
        self.assertEqual(result["readmission_30day"].sum(), 0)
        self.assertEqual(result["readmission_90day"].sum(), 0)

        # Check that days_to_readmission is NaN for all rows
        self.assertTrue(result["days_to_readmission"].isna().all())

    def test_all_deaths(self):
        """
        Test the function with a dataset where all admissions result in death.
        """
        # Create a dataset where all admissions result in death
        admissions_all_deaths = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 3],
                "hadm_id": [101, 102, 201, 301],
                "admittime": [
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 15),
                    datetime(2023, 2, 1),
                    datetime(2023, 3, 1),
                ],
                "dischtime": [
                    datetime(2023, 1, 5),
                    datetime(2023, 1, 20),
                    datetime(2023, 2, 5),
                    datetime(2023, 3, 5),
                ],
                "hospital_death": [True, True, True, True],
                "hospital_expire_flag": [1, 1, 1, 1],  # Add flag
            }
        )

        # Sort by subject_id and admittime
        admissions_all_deaths = admissions_all_deaths.sort_values(
            ["subject_id", "admittime"]
        )

        # Call the function
        processed_admissions = self.processor._calculate_derived_features(
            admissions_all_deaths.copy()
        )
        result = self.processor._identify_readmissions(processed_admissions)

        # Assertions
        self.assertEqual(len(result), 4)  # Should have 4 rows

        # Check that no readmissions were identified (all resulted in death)
        self.assertEqual(result["readmission_30day"].sum(), 0)
        self.assertEqual(result["readmission_90day"].sum(), 0)

    def test_empty_dataframe(self):
        """
        Test the function with an empty DataFrame.
        """
        empty_df = pd.DataFrame(
            columns=[
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "hospital_death",
                "hospital_expire_flag",
            ]
        )
        # Ensure correct dtypes for empty df processing
        empty_df = empty_df.astype(
            {
                "subject_id": int,
                "hadm_id": int,
                "admittime": "datetime64[ns]",
                "dischtime": "datetime64[ns]",
                "hospital_death": bool,
                "hospital_expire_flag": int,
            }
        )

        # Call the function
        processed_admissions = self.processor._calculate_derived_features(
            empty_df.copy()
        )
        result = self.processor._identify_readmissions(processed_admissions)

        # Assertions
        self.assertTrue(result.empty)
        self.assertIn("readmission_30day", result.columns)
        self.assertIn("readmission_90day", result.columns)
        self.assertIn("days_to_readmission", result.columns)

    def test_missing_column(self):
        """
        Test the function when an expected column ('dischtime') is missing.
        The code should handle this gracefully, log errors, and return default values.
        """
        admissions_missing_col = self.admissions.drop(columns=["dischtime"])

        # Call the functions - expect no KeyError now
        processed_admissions = self.processor._calculate_derived_features(
            admissions_missing_col.copy()
        )
        result = self.processor._identify_readmissions(processed_admissions)

        # Assertions: Check that processing completed and default values were assigned
        self.assertEqual(len(result), len(admissions_missing_col))
        # 'los_days' calculation would fail, check if it's NA (or how it's handled)
        self.assertTrue(result["los_days"].isna().all())
        # Readmission calculation fails due to missing dischtime, flags should be False
        self.assertEqual(result["readmission_30day"].sum(), 0)
        self.assertEqual(result["readmission_90day"].sum(), 0)
        self.assertTrue(result["days_to_readmission"].isna().all())

    def test_exact_30_day_readmission(self):
        """
        Test readmission exactly 30 days after discharge.
        """
        admissions_exact_30 = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [101, 102],
                "admittime": [
                    datetime(2023, 1, 1),
                    datetime(2023, 2, 5),
                ],  # 30 days after Jan 6
                "dischtime": [datetime(2023, 1, 6), datetime(2023, 2, 10)],
                "hospital_death": [False, False],
                "hospital_expire_flag": [0, 0],
            }
        )
        admissions_exact_30 = admissions_exact_30.sort_values(
            ["subject_id", "admittime"]
        )

        processed_admissions = self.processor._calculate_derived_features(
            admissions_exact_30.copy()
        )
        result = self.processor._identify_readmissions(processed_admissions)

        # Admission 101 should be flagged as 30-day readmission
        self.assertTrue(
            result.loc[result["hadm_id"] == 101, "readmission_30day"].iloc[0]
        )
        self.assertTrue(
            result.loc[result["hadm_id"] == 101, "readmission_90day"].iloc[0]
        )
        self.assertAlmostEqual(
            result.loc[result["hadm_id"] == 101, "days_to_readmission"].iloc[0],
            30.0,
            places=1,
        )

    def test_exact_90_day_readmission(self):
        """
        Test readmission exactly 90 days after discharge.
        """
        admissions_exact_90 = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [101, 102],
                "admittime": [
                    datetime(2023, 1, 1),
                    datetime(2023, 4, 6),
                ],  # 90 days after Jan 6
                "dischtime": [datetime(2023, 1, 6), datetime(2023, 4, 10)],
                "hospital_death": [False, False],
                "hospital_expire_flag": [0, 0],
            }
        )
        admissions_exact_90 = admissions_exact_90.sort_values(
            ["subject_id", "admittime"]
        )

        processed_admissions = self.processor._calculate_derived_features(
            admissions_exact_90.copy()
        )
        result = self.processor._identify_readmissions(processed_admissions)

        # Admission 101 should be flagged as 90-day but not 30-day readmission
        self.assertFalse(
            result.loc[result["hadm_id"] == 101, "readmission_30day"].iloc[0]
        )
        self.assertTrue(
            result.loc[result["hadm_id"] == 101, "readmission_90day"].iloc[0]
        )
        self.assertAlmostEqual(
            result.loc[result["hadm_id"] == 101, "days_to_readmission"].iloc[0],
            90.0,
            places=1,
        )

    def test_same_day_admission(self):
        """
        Test admission on the same day as discharge (should not be readmission).
        """
        admissions_same_day = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [101, 102],
                "admittime": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 5, 14, 0, 0),
                ],  # Admitted later same day
                "dischtime": [datetime(2023, 1, 5, 12, 0, 0), datetime(2023, 1, 10)],
                "hospital_death": [False, False],
                "hospital_expire_flag": [0, 0],
            }
        )
        admissions_same_day = admissions_same_day.sort_values(
            ["subject_id", "admittime"]
        )

        processed_admissions = self.processor._calculate_derived_features(
            admissions_same_day.copy()
        )
        result = self.processor._identify_readmissions(processed_admissions)

        # Admission 101 should NOT be flagged as readmission
        self.assertFalse(
            result.loc[result["hadm_id"] == 101, "readmission_30day"].iloc[0]
        )
        self.assertFalse(
            result.loc[result["hadm_id"] == 101, "readmission_90day"].iloc[0]
        )
        # Days to readmission should reflect the time difference
        expected_days = (
            admissions_same_day.loc[1, "admittime"]
            - admissions_same_day.loc[0, "dischtime"]
        ).total_seconds() / (24 * 60 * 60)
        self.assertAlmostEqual(
            result.loc[result["hadm_id"] == 101, "days_to_readmission"].iloc[0],
            expected_days,
            places=1,
        )

    def test_admission_before_discharge(self):
        """
        Test scenario where next admission time is before previous discharge time (data error).
        """
        admissions_error = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [101, 102],
                "admittime": [
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 4),
                ],  # Admitted before discharge
                "dischtime": [datetime(2023, 1, 5), datetime(2023, 1, 10)],
                "hospital_death": [False, False],
                "hospital_expire_flag": [0, 0],
            }
        )
        admissions_error = admissions_error.sort_values(["subject_id", "admittime"])

        processed_admissions = self.processor._calculate_derived_features(
            admissions_error.copy()
        )
        result = self.processor._identify_readmissions(processed_admissions)

        # Admission 101 should NOT be flagged as readmission
        self.assertFalse(
            result.loc[result["hadm_id"] == 101, "readmission_30day"].iloc[0]
        )
        self.assertFalse(
            result.loc[result["hadm_id"] == 101, "readmission_90day"].iloc[0]
        )
        # Days to readmission should be negative
        self.assertLess(
            result.loc[result["hadm_id"] == 101, "days_to_readmission"].iloc[0], 0
        )


if __name__ == "__main__":
    unittest.main()
