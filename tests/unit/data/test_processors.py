"""
Unit tests for the processors module.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.processors import AdmissionProcessor


class TestIdentifyReadmissions(unittest.TestCase):
    """
    Test cases for the _identify_readmissions function.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a sample admissions dataframe
        self.admissions = pd.DataFrame({
            "subject_id": [1, 1, 1, 2, 2, 3],
            "hadm_id": [101, 102, 103, 201, 202, 301],
            "admittime": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 15),
                datetime(2023, 3, 1),
                datetime(2023, 2, 1),
                datetime(2023, 4, 15),
                datetime(2023, 3, 10)
            ],
            "dischtime": [
                datetime(2023, 1, 5),
                datetime(2023, 1, 20),
                datetime(2023, 3, 10),
                datetime(2023, 2, 10),
                datetime(2023, 4, 25),
                datetime(2023, 3, 15)
            ],
            "hospital_death": [False, False, True, False, False, False]
        })
        
        # Sort by subject_id and admittime
        self.admissions = self.admissions.sort_values(["subject_id", "admittime"])
        
        # Create a processor instance
        self.processor = AdmissionProcessor()
    
    def test_identify_readmissions(self):
        """
        Test the _identify_readmissions function with a typical dataset.
        """
        # Call the function
        result = self.processor._identify_readmissions(self.admissions)
        
        # Assertions
        self.assertEqual(len(result), 6)  # Should have 6 rows
        
        # Check readmission flags
        # Patient 1: First admission (hadm_id 101) should have 30-day readmission
        self.assertTrue(result.loc[result["hadm_id"] == 101, "readmission_30day"].iloc[0])
        self.assertTrue(result.loc[result["hadm_id"] == 101, "readmission_90day"].iloc[0])
        
        # Patient 1: Second admission (hadm_id 102) should have 90-day readmission but not 30-day
        self.assertFalse(result.loc[result["hadm_id"] == 102, "readmission_30day"].iloc[0])
        self.assertTrue(result.loc[result["hadm_id"] == 102, "readmission_90day"].iloc[0])
        
        # Patient 1: Third admission (hadm_id 103) resulted in death, so no readmission
        self.assertFalse(result.loc[result["hadm_id"] == 103, "readmission_30day"].iloc[0])
        self.assertFalse(result.loc[result["hadm_id"] == 103, "readmission_90day"].iloc[0])
        
        # Patient 2: First admission (hadm_id 201) should not have 30-day readmission
        self.assertFalse(result.loc[result["hadm_id"] == 201, "readmission_30day"].iloc[0])
        self.assertTrue(result.loc[result["hadm_id"] == 201, "readmission_90day"].iloc[0])
        
        # Patient 3: Only one admission, so no readmission
        self.assertFalse(result.loc[result["hadm_id"] == 301, "readmission_30day"].iloc[0])
        self.assertFalse(result.loc[result["hadm_id"] == 301, "readmission_90day"].iloc[0])
    
    def test_days_to_readmission(self):
        """
        Test the calculation of days_to_readmission.
        """
        # Call the function
        result = self.processor._identify_readmissions(self.admissions)
        
        # Assertions
        # Patient 1: First admission to second admission
        days_to_readmission_1 = result.loc[result["hadm_id"] == 101, "days_to_readmission"].iloc[0]
        expected_days_1 = (datetime(2023, 1, 15) - datetime(2023, 1, 5)).total_seconds() / (24 * 60 * 60)
        self.assertAlmostEqual(days_to_readmission_1, expected_days_1, places=1)
        
        # Patient 1: Second admission to third admission
        days_to_readmission_2 = result.loc[result["hadm_id"] == 102, "days_to_readmission"].iloc[0]
        expected_days_2 = (datetime(2023, 3, 1) - datetime(2023, 1, 20)).total_seconds() / (24 * 60 * 60)
        self.assertAlmostEqual(days_to_readmission_2, expected_days_2, places=1)
        
        # Patient 2: First admission to second admission
        days_to_readmission_3 = result.loc[result["hadm_id"] == 201, "days_to_readmission"].iloc[0]
        expected_days_3 = (datetime(2023, 4, 15) - datetime(2023, 2, 10)).total_seconds() / (24 * 60 * 60)
        self.assertAlmostEqual(days_to_readmission_3, expected_days_3, places=1)
    
    def test_no_readmissions(self):
        """
        Test the function with a dataset that has no readmissions.
        """
        # Create a dataset with no readmissions (all different patients)
        admissions_no_readmissions = pd.DataFrame({
            "subject_id": [1, 2, 3, 4, 5],
            "hadm_id": [101, 201, 301, 401, 501],
            "admittime": [
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                datetime(2023, 3, 1),
                datetime(2023, 4, 1),
                datetime(2023, 5, 1)
            ],
            "dischtime": [
                datetime(2023, 1, 5),
                datetime(2023, 2, 5),
                datetime(2023, 3, 5),
                datetime(2023, 4, 5),
                datetime(2023, 5, 5)
            ],
            "hospital_death": [False, False, False, False, False]
        })
        
        # Call the function
        result = self.processor._identify_readmissions(admissions_no_readmissions)
        
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
        admissions_all_deaths = pd.DataFrame({
            "subject_id": [1, 1, 2, 3],
            "hadm_id": [101, 102, 201, 301],
            "admittime": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 15),
                datetime(2023, 2, 1),
                datetime(2023, 3, 1)
            ],
            "dischtime": [
                datetime(2023, 1, 5),
                datetime(2023, 1, 20),
                datetime(2023, 2, 5),
                datetime(2023, 3, 5)
            ],
            "hospital_death": [True, True, True, True]
        })
        
        # Sort by subject_id and admittime
        admissions_all_deaths = admissions_all_deaths.sort_values(["subject_id", "admittime"])
        
        # Call the function
        result = self.processor._identify_readmissions(admissions_all_deaths)
        
        # Assertions
        self.assertEqual(len(result), 4)  # Should have 4 rows
        
        # Check that no readmissions were identified (all resulted in death)
        self.assertEqual(result["readmission_30day"].sum(), 0)
        self.assertEqual(result["readmission_90day"].sum(), 0)


if __name__ == "__main__":
    unittest.main()