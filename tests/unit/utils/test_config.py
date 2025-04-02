"""
Unit tests for the config module.
"""

import logging  # Import logging for level comparison
import unittest
from unittest.mock import mock_open, patch

# Import functions being tested from config module
from utils.config import load_config, load_mappings

# Import the function that was misplaced from the logger module
from utils.logger import get_log_level_from_config

# Removed unused imports: os, Path, MagicMock, yaml, CONFIG_SCHEMA, MAPPINGS_SCHEMA


class TestConfig(unittest.TestCase):
    """
    Test cases for the config module.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Sample config data - Ensure it includes all required sections/subsections from CONFIG_SCHEMA
        self.sample_config = {
            "logging": {"level": "INFO", "file_output": True, "console_output": True},
            "data": {
                "raw": {"mimic_iii": "path/to/iii", "mimic_iv": "path/to/iv"},
                "processed": {
                    "base_path": "data/processed/",
                    "patient_data": "...",
                    "admission_data": "...",
                },
                "external": {
                    "some_external_data": "path/to/external"
                },  # Added missing subsection
            },
            "features": {
                "demographic": {"include": True, "age_bins": [0, 18, 65, 100]},
                "vitals": {
                    "include": True,
                    "window_hours": 24,
                    "aggregation_methods": ["mean", "std"],
                },  # Added missing subsection
                "labs": {
                    "include": True,
                    "window_hours": 24,
                    "aggregation_methods": ["mean"],
                },  # Added missing subsection
                "medications": {"include": False},  # Added missing subsection
                "procedures": {"include": False},  # Added missing subsection
                "diagnoses": {"include": True},  # Added missing subsection
                "temporal": {"include": False},  # Added missing subsection
            },
            "models": {  # Added missing section
                "readmission": {"type": "logistic", "params": {}},
                "mortality": {"type": "xgboost", "params": {}},
                "los": {"type": "linear", "params": {}},
            },
            "evaluation": {  # Added missing section
                "classification": {
                    "metrics": ["accuracy", "roc_auc", "precision", "recall"]
                },
                "regression": {"metrics": ["mae", "rmse"]},
            },
            "api": {  # Added missing section
                "host": "127.0.0.1",
                "port": 8000,
                "debug": False,
            },
            "dashboard": {  # Added missing section
                "host": "127.0.0.1",
                "port": 8501,
                "debug": False,
            },
        }

        # Sample mappings data - Ensure it includes all required sections/subsections from MAPPINGS_SCHEMA
        self.sample_mappings = {
            "lab_tests": {
                "common_labs": ["Glucose", "Potassium", "Sodium"],
                "mappings": {  # Added missing subsection (example structure)
                    "50809": "Glucose",
                    "50931": "Glucose",
                },
                "lab_name_variations": {  # Kept for compatibility if used elsewhere
                    "Glucose": ["Glucose", "Glucose, CSF"],
                    "Potassium": ["Potassium"],
                    "Sodium": ["Sodium"],
                },
            },
            "vital_signs": {
                "categories": {
                    "Heart Rate": [211, 220045],
                    "Systolic BP": [51, 442, 455],
                },
                "itemids": [211, 220045, 51, 442, 455],  # Added missing subsection
            },
            "icd9_categories": {  # Added missing section
                "ranges": {"Infectious": [1, 139]},
                "specific_codes": {"Diabetes": ["25000"]},
            },
        }

    # Patch pathlib.Path methods used in load_config
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_config(self, mock_yaml_load, mock_file_open, mock_exists):
        """
        Test loading the configuration file successfully.
        """
        # Configure mocks
        mock_exists.return_value = True  # Simulate file exists
        mock_yaml_load.return_value = self.sample_config
        load_config.cache_clear()  # Clear cache before test

        # Call the function
        config = load_config()  # Use default path

        # Assertions
        self.assertEqual(config, self.sample_config)
        mock_exists.assert_called_once()  # Check Path.exists was called
        mock_file_open.assert_called_once()  # Check Path.open was called
        mock_yaml_load.assert_called_once()
        load_config.cache_clear()  # Clear cache after test

    @patch("pathlib.Path.exists")  # Patch Path.exists
    def test_load_config_file_not_found(self, mock_exists):
        """
        Test loading a non-existent configuration file.
        """
        # Configure mock
        mock_exists.return_value = False  # Simulate file does not exist
        load_config.cache_clear()  # Clear cache before testing failure case

        # Assertions
        with self.assertRaises(FileNotFoundError):
            load_config()  # Use default path
        mock_exists.assert_called_once()  # Check Path.exists was called
        load_config.cache_clear()  # Clear cache after test

    # Patch pathlib.Path methods used in load_mappings
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_mappings(self, mock_yaml_load, mock_file_open, mock_exists):
        """
        Test loading the mappings file successfully.
        """
        # Configure mocks
        mock_exists.return_value = True  # Simulate file exists
        mock_yaml_load.return_value = self.sample_mappings

        # Call the function
        mappings = load_mappings()

        # Assertions
        self.assertEqual(mappings, self.sample_mappings)
        mock_exists.assert_called_once()  # Check Path.exists was called
        mock_file_open.assert_called_once()  # Check Path.open was called
        mock_yaml_load.assert_called_once()

    @patch("pathlib.Path.exists")  # Patch Path.exists
    def test_load_mappings_file_not_found(self, mock_exists):
        """
        Test loading a non-existent mappings file.
        """
        # Configure mock
        mock_exists.return_value = False  # Simulate file does not exist

        # Assertions
        with self.assertRaises(FileNotFoundError):
            load_mappings()
        mock_exists.assert_called_once()  # Check Path.exists was called

    # Patch load_config where it's defined (in utils.config)
    @patch("utils.config.load_config")
    def test_get_log_level_from_config(self, mock_load_config):
        """
        Test getting the log level from the configuration (function resides in logger).
        """
        # Test with INFO level
        mock_load_config.return_value = {"logging": {"level": "INFO"}}
        log_level = get_log_level_from_config()
        self.assertEqual(log_level, logging.INFO)

        # Test with DEBUG level
        mock_load_config.return_value = {"logging": {"level": "DEBUG"}}
        log_level = get_log_level_from_config()
        self.assertEqual(log_level, logging.DEBUG)

        # Test with WARNING level
        mock_load_config.return_value = {"logging": {"level": "WARNING"}}
        log_level = get_log_level_from_config()
        self.assertEqual(log_level, logging.WARNING)

        # Test with invalid level (should default to INFO)
        mock_load_config.return_value = {"logging": {"level": "INVALID"}}
        log_level = get_log_level_from_config()
        self.assertEqual(log_level, logging.INFO)

        # Test with missing logging section (should default to INFO)
        mock_load_config.return_value = {}
        log_level = get_log_level_from_config()
        self.assertEqual(log_level, logging.INFO)

        # Test with exception in load_config (should default to INFO)
        mock_load_config.side_effect = Exception("Test exception")
        log_level = get_log_level_from_config()
        self.assertEqual(log_level, logging.INFO)


if __name__ == "__main__":
    unittest.main()
