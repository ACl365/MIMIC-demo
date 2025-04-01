"""
Unit tests for the config module.
"""

import os
import unittest
from unittest.mock import patch, mock_open, MagicMock
import yaml

from src.utils.config import load_config, load_mappings, get_log_level_from_config


class TestConfig(unittest.TestCase):
    """
    Test cases for the config module.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Sample config data
        self.sample_config = {
            "logging": {
                "level": "INFO",
                "file_output": True,
                "console_output": True
            },
            "data": {
                "raw": {
                    "mimic_iii": "raw/MIMIC_iii/mimic-iii-clinical-database-demo-1.4/",
                    "mimic_iv": "raw/MIMIC_iv/"
                },
                "processed": {
                    "base_path": "data/processed/",
                    "patient_data": "data/processed/patient_data.csv",
                    "admission_data": "data/processed/admission_data.csv"
                }
            },
            "features": {
                "demographic": {
                    "include": True,
                    "age_bins": [0, 18, 30, 50, 70, 100]
                }
            }
        }
        
        # Sample mappings data
        self.sample_mappings = {
            "lab_tests": {
                "common_labs": [
                    "Glucose", "Potassium", "Sodium"
                ],
                "lab_name_variations": {
                    "Glucose": ["Glucose", "Glucose, CSF", "Glucose, Whole Blood"],
                    "Potassium": ["Potassium", "Potassium, Whole Blood"],
                    "Sodium": ["Sodium", "Sodium, Whole Blood"]
                }
            },
            "vital_signs": {
                "categories": {
                    "Heart Rate": [211, 220045],
                    "Systolic BP": [51, 442, 455, 6701, 220179, 220050]
                }
            }
        }
    
    @patch('src.utils.config.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_load_config(self, mock_yaml_load, mock_file_open, mock_exists):
        """
        Test loading the configuration file.
        """
        # Configure mocks
        mock_exists.return_value = True
        mock_yaml_load.return_value = self.sample_config
        
        # Call the function
        config = load_config()
        
        # Assertions
        self.assertEqual(config, self.sample_config)
        mock_exists.assert_called_once()
        mock_file_open.assert_called_once()
        mock_yaml_load.assert_called_once()
    
    @patch('src.utils.config.os.path.exists')
    def test_load_config_file_not_found(self, mock_exists):
        """
        Test loading a non-existent configuration file.
        """
        # Configure mock
        mock_exists.return_value = False
        
        # Assertions
        with self.assertRaises(FileNotFoundError):
            load_config()
    
    @patch('src.utils.config.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_load_mappings(self, mock_yaml_load, mock_file_open, mock_exists):
        """
        Test loading the mappings file.
        """
        # Configure mocks
        mock_exists.return_value = True
        mock_yaml_load.return_value = self.sample_mappings
        
        # Call the function
        mappings = load_mappings()
        
        # Assertions
        self.assertEqual(mappings, self.sample_mappings)
        mock_exists.assert_called_once()
        mock_file_open.assert_called_once()
        mock_yaml_load.assert_called_once()
    
    @patch('src.utils.config.os.path.exists')
    def test_load_mappings_file_not_found(self, mock_exists):
        """
        Test loading a non-existent mappings file.
        """
        # Configure mock
        mock_exists.return_value = False
        
        # Assertions
        with self.assertRaises(FileNotFoundError):
            load_mappings()
    
    @patch('src.utils.config.load_config')
    def test_get_log_level_from_config(self, mock_load_config):
        """
        Test getting the log level from the configuration.
        """
        import logging
        
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