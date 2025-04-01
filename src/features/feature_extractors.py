"""
Feature extractors for the MIMIC datasets.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import get_logger, load_config, get_data_path


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
                admissions[col] = pd.to_datetime(admissions[col], errors='coerce')
        
        # Merge patient and admission data
        data = pd.merge(
            admissions, 
            patients, 
            on=["subject_id", "source"], 
            how="left", 
            suffixes=("", "_patient")
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
        # Initialize features dataframe
        features = pd.DataFrame()
        
        # Add identifiers
        features["subject_id"] = data["subject_id"]
        features["hadm_id"] = data["hadm_id"]
        
        # Add basic demographics
        features["age"] = data["age"]
        features["gender"] = data["gender"]
        
        # Age bins
        age_bins = self.config["features"]["demographic"]["age_bins"]
        age_labels = [f"{age_bins[i]}-{age_bins[i+1]}" for i in range(len(age_bins) - 1)]
        features["age_group"] = pd.cut(
            data["age"], 
            bins=age_bins, 
            labels=age_labels, 
            right=False
        )
        
        # One-hot encode categorical variables
        categorical_features = ["gender", "age_group", "admission_type", "insurance", "marital_status"]
        for feature in categorical_features:
            if feature in data.columns:
                # Create dummy variables
                dummies = pd.get_dummies(data[feature], prefix=feature, dummy_na=True)
                # Add to features dataframe
                features = pd.concat([features, dummies], axis=1)
        
        # Add ethnicity/race features if available
        if "ethnicity" in data.columns:
            ethnicity_dummies = pd.get_dummies(data["ethnicity"], prefix="ethnicity", dummy_na=True)
            features = pd.concat([features, ethnicity_dummies], axis=1)
        elif "race" in data.columns:
            race_dummies = pd.get_dummies(data["race"], prefix="race", dummy_na=True)
            features = pd.concat([features, race_dummies], axis=1)
        
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
            parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]
        )
        
        # Load ICU stay data
        icu_path = get_data_path("processed", "icu_data", self.config)
        icustays = pd.read_csv(icu_path, parse_dates=["intime", "outtime"])
        
        # Extract lab features
        lab_features = self._extract_lab_features(admissions, icustays)
        
        # Extract vital sign features
        vital_features = self._extract_vital_features(admissions, icustays)
        
        # Combine features
        features = pd.merge(
            lab_features,
            vital_features,
            on=["subject_id", "hadm_id", "stay_id"],
            how="outer"
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
        
        # Load MIMIC-III lab data
        try:
            mimic3_path = get_data_path("raw", "mimic_iii", self.config)
            lab_path = os.path.join(mimic3_path, "LABEVENTS.csv")
            
            if os.path.exists(lab_path):
                # Load lab data - use lowercase column names for parse_dates
                labs = pd.read_csv(lab_path, parse_dates=["charttime"])
                
                # Ensure column names are lowercase
                labs.columns = labs.columns.str.lower()
                
                # Load lab items dictionary
                lab_items_path = os.path.join(mimic3_path, "D_LABITEMS.csv")
                lab_items = pd.read_csv(lab_items_path)
                lab_items.columns = lab_items.columns.str.lower()
                
                # Process lab data
                lab_features = self._process_lab_data(labs, lab_items, admissions, icustays)
            else:
                self.logger.warning("MIMIC-III lab data not found")
                # Create an empty dataframe with the required columns
                lab_features = pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])
                
                # Add all unique subject_id, hadm_id, stay_id combinations from icustays
                id_data = []
                for _, stay in icustays.iterrows():
                    id_data.append({
                        "subject_id": stay["subject_id"],
                        "hadm_id": stay["hadm_id"],
                        "stay_id": stay["stay_id"]
                    })
                if id_data:
                    lab_features = pd.concat([lab_features, pd.DataFrame(id_data)], ignore_index=True)
        except Exception as e:
            self.logger.warning(f"Error loading MIMIC-III lab data: {e}")
            # Create an empty dataframe with the required columns
            lab_features = pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])
            
            # Add all unique subject_id, hadm_id, stay_id combinations from icustays
            id_data = []
            for _, stay in icustays.iterrows():
                id_data.append({
                    "subject_id": stay["subject_id"],
                    "hadm_id": stay["hadm_id"],
                    "stay_id": stay["stay_id"]
                })
            if id_data:
                lab_features = pd.concat([lab_features, pd.DataFrame(id_data)], ignore_index=True)
        
        return lab_features
    def _process_lab_data(
        self,
        labs: pd.DataFrame,
        lab_items: pd.DataFrame,
        admissions: pd.DataFrame,
        icustays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process laboratory data and extract features using vectorized operations.
        
        Args:
            labs (pd.DataFrame): Laboratory data
            lab_items (pd.DataFrame): Laboratory items dictionary
            admissions (pd.DataFrame): Admission data
            icustays (pd.DataFrame): ICU stay data
        
        Returns:
            pd.DataFrame: Laboratory features
        """
        try:
            from src.utils import load_mappings
            
            self.logger.info("Processing lab data with optimised vectorized operations")
            
            # Merge lab data with lab items to get labels
            labs = pd.merge(labs, lab_items[["itemid", "label"]], on="itemid", how="left")
            
            # Load lab test mappings from configuration
            try:
                # Cache the mappings to avoid loading them repeatedly
                if not hasattr(self, '_lab_mappings'):
                    mappings = load_mappings()
                    self._lab_mappings = mappings.get('lab_tests', {})
                
                # Get common labs from mappings
                common_labs = self._lab_mappings.get('common_labs', [])
                
                # Get lab name variations from mappings
                lab_name_mapping = self._lab_mappings.get('lab_name_variations', {})
                
                self.logger.info(f"Loaded {len(common_labs)} common lab tests and {len(lab_name_mapping)} lab name mappings from configuration")
            except Exception as e:
                self.logger.warning(f"Error loading lab mappings from configuration: {e}")
                self.logger.warning("Falling back to hardcoded lab test lists")
                
                # Fallback to hardcoded lists if mappings fail
                common_labs = [
                    "Glucose", "Potassium", "Sodium", "Chloride", "Creatinine",
                    "BUN", "Bicarbonate", "Anion Gap", "Hemoglobin", "Hematocrit",
                    "WBC", "Platelet Count", "Magnesium", "Calcium", "Phosphate",
                    "Lactate", "pH", "pO2", "pCO2", "Base Excess", "Albumin",
                    "ALT", "AST", "Alkaline Phosphatase", "Bilirubin", "Troponin"
                ]
                
                # Create a mapping of common lab names to their variations in the dataset
                lab_name_mapping = {
                    "Glucose": ["Glucose", "Glucose, CSF", "Glucose, Whole Blood"],
                    "Potassium": ["Potassium", "Potassium, Whole Blood"],
                    "Sodium": ["Sodium", "Sodium, Whole Blood"],
                    "Chloride": ["Chloride", "Chloride, Whole Blood"],
                    "Creatinine": ["Creatinine"],
                    "BUN": ["BUN", "Urea Nitrogen"],
                    "Bicarbonate": ["Bicarbonate", "HCO3", "Calculated Bicarbonate, Whole Blood"],
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
                    "Troponin": ["Troponin I", "Troponin T", "Troponin"]
                }
            
            # Create a flat list of all lab name variations
            all_lab_variations = [variation for variations in lab_name_mapping.values() for variation in variations]
            
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
                how="inner"
            )
            
            # Filter to labs within ICU stay window
            labs = labs[
                (labs["charttime"] >= labs["intime"]) &
                (labs["charttime"] <= labs["outtime"])
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
            if isinstance(self.config["features"]["lab_values"]["aggregation_methods"], list):
                aggregation_methods = self.config["features"]["lab_values"]["aggregation_methods"]
            else:
                # Default to a list with a single method if it's not a list
                aggregation_methods = ["mean"]
            
            # If there are no labs after filtering, return an empty dataframe with ID columns
            if len(window_labs) == 0:
                self.logger.warning("No lab data found within the specified window")
                # Create a dataframe with all ICU stays
                features = icustays[["subject_id", "hadm_id", "stay_id"]].copy()
                return features
            
            # Use pivot_table for efficient reshaping - this replaces the loop over ICU stays
            # This is much more efficient than processing each ICU stay individually
            self.logger.info(f"Creating pivot table for {len(window_labs)} lab measurements")
            
            # Group by subject_id, hadm_id, stay_id, and standardized_label, then aggregate
            lab_features = pd.pivot_table(
                window_labs,
                values="valuenum",
                index=["subject_id", "hadm_id", "stay_id"],
                columns="standardized_label",
                aggfunc=aggregation_methods
            )
            
            # Flatten the multi-level column names
            if isinstance(lab_features.columns, pd.MultiIndex):
                lab_features.columns = [f"{col[1]}_{col[0]}" if isinstance(col, tuple) and len(col) > 1 else col for col in lab_features.columns]
            
            # Reset index to convert back to a regular dataframe
            lab_features = lab_features.reset_index()
            
            # Ensure all ICU stays are included, even those without lab data
            all_stays = icustays[["subject_id", "hadm_id", "stay_id"]].copy()
            features = pd.merge(
                all_stays,
                lab_features,
                on=["subject_id", "hadm_id", "stay_id"],
                how="left"
            )
            
            self.logger.info(f"Created lab features with {features.shape[1] - 3} measurements for {features.shape[0]} ICU stays")
            
            return features
        except Exception as e:
            self.logger.warning(f"Error in _process_lab_data: {e}")
            # Return a dataframe with just the ID columns
            return icustays[["subject_id", "hadm_id", "stay_id"]].copy()
            return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])
        
        return features
    
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
        
        # Load MIMIC-III chart data
        try:
            mimic3_path = get_data_path("raw", "mimic_iii", self.config)
            chart_path = os.path.join(mimic3_path, "CHARTEVENTS.csv")
            
            if os.path.exists(chart_path):
                # Load chart data (this could be very large, so we'll be selective)
                # Define vital sign itemids
                vital_itemids = [
                    # Heart rate
                    211, 220045,
                    # Systolic BP
                    51, 442, 455, 6701, 220179, 220050,
                    # Diastolic BP
                    8368, 8440, 8441, 8555, 220180, 220051,
                    # Mean BP
                    456, 52, 6702, 443, 220052, 220181,
                    # Respiratory rate
                    615, 618, 220210, 224690,
                    # Temperature
                    223761, 678, 223762,
                    # SpO2
                    646, 220277,
                    # Glucose
                    807, 811, 1529, 3745, 3744, 225664, 220621, 226537,
                    # GCS
                    198, 220739, 184, 220734, 723, 223900, 454, 223901
                ]
                
                # Read only the rows with these itemids
                # This is a simplification - in practice, you might need to chunk this
                # or use a database to handle the large file size
                chart_data = pd.read_csv(
                    chart_path,
                    parse_dates=["charttime"],
                    usecols=["subject_id", "hadm_id", "icustay_id", "itemid", "charttime", "value", "valuenum"],
                    dtype={"subject_id": int, "hadm_id": int, "icustay_id": float, "itemid": int}
                )
                
                # Filter to vital signs
                chart_data = chart_data[chart_data["itemid"].isin(vital_itemids)]
                
                # Convert column names to lowercase
                chart_data.columns = chart_data.columns.str.lower()
                
                # Load items dictionary
                items_path = os.path.join(mimic3_path, "D_ITEMS.csv")
                items = pd.read_csv(items_path)
                items.columns = items.columns.str.lower()
                
                # Process vital sign data
                vital_features = self._process_vital_data(chart_data, items, admissions, icustays)
            else:
                self.logger.warning("MIMIC-III chart data not found")
                vital_features = pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])
        except Exception as e:
            self.logger.warning(f"Error loading MIMIC-III chart data: {e}")
            vital_features = pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])
        
        return vital_features
    
    def _process_vital_data(
        self,
        chart_data: pd.DataFrame,
        items: pd.DataFrame,
        admissions: pd.DataFrame,
        icustays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process vital sign data and extract features using vectorized operations.
        
        Args:
            chart_data (pd.DataFrame): Chart event data
            items (pd.DataFrame): Items dictionary
            admissions (pd.DataFrame): Admission data
            icustays (pd.DataFrame): ICU stay data
        
        Returns:
            pd.DataFrame: Vital sign features
        """
        try:
            from src.utils import load_mappings
            
            self.logger.info("Processing vital sign data with optimised vectorized operations")
            
            # Merge chart data with items to get labels
            chart_data = pd.merge(chart_data, items[["itemid", "label"]], on="itemid", how="left")
            
            # Rename icustay_id to stay_id for consistency
            chart_data = chart_data.rename(columns={"icustay_id": "stay_id"})
            
            # Load vital sign mappings from configuration
            try:
                # Cache the mappings to avoid loading them repeatedly
                if not hasattr(self, '_vital_mappings'):
                    mappings = load_mappings()
                    self._vital_mappings = mappings.get('vital_signs', {})
                
                # Get vital sign categories from mappings
                vital_categories = self._vital_mappings.get('categories', {})
                
                self.logger.info(f"Loaded {len(vital_categories)} vital sign categories from configuration")
            except Exception as e:
                self.logger.warning(f"Error loading vital sign mappings from configuration: {e}")
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
                    "GCS Total": [198, 220739],
                    "GCS Motor": [184, 220734],
                    "GCS Verbal": [723, 223900],
                    "GCS Eye": [454, 223901]
                }
            
            # Create a reverse mapping from itemid to category
            itemid_to_category = {}
            for category, itemids in vital_categories.items():
                for itemid in itemids:
                    itemid_to_category[itemid] = category
            
            # Add category to chart data - use vectorized mapping instead of apply
            # First convert itemid to appropriate type for dictionary lookup
            chart_data["itemid_int"] = pd.to_numeric(chart_data["itemid"], errors="coerce")
            # Then create a Series mapping using the dictionary
            category_map = pd.Series(itemid_to_category)
            # Map the values
            chart_data["category"] = chart_data["itemid_int"].map(category_map)
            
            # Filter out rows with missing category
            chart_data = chart_data.dropna(subset=["category"])
            
            # Filter to numeric values only
            chart_data = chart_data.dropna(subset=["valuenum"])
            
            # Get vital window hours from config
            vital_window_hours = self.config["features"]["vitals"]["window_hours"]
            
            # Check if aggregation_methods is a list or a string
            if isinstance(self.config["features"]["vitals"]["aggregation_methods"], list):
                aggregation_methods = self.config["features"]["vitals"]["aggregation_methods"]
            else:
                # Default to a list with a single method if it's not a list
                aggregation_methods = ["mean"]
            
            # If there are no chart data, return an empty dataframe with ID columns
            if len(chart_data) == 0:
                self.logger.warning("No vital sign data found")
                # Create a dataframe with all ICU stays
                features = icustays[["subject_id", "hadm_id", "stay_id"]].copy()
                return features
            
            # Merge chart data with ICU stays to get intime for calculating hours from admission
            chart_data = pd.merge(
                chart_data,
                icustays[["subject_id", "hadm_id", "stay_id", "intime"]],
                on=["subject_id", "hadm_id", "stay_id"],
                how="inner"
            )
            
            # Calculate time from ICU admission for all vitals at once
            chart_data["hours_from_admission"] = (
                chart_data["charttime"] - chart_data["intime"]
            ).dt.total_seconds() / 3600
            
            # Filter to vitals within window for all patients at once
            window_vitals = chart_data[chart_data["hours_from_admission"] <= vital_window_hours]
            
            # If there are no vitals within the window, return an empty dataframe with ID columns
            if len(window_vitals) == 0:
                self.logger.warning("No vital sign data found within the specified window")
                features = icustays[["subject_id", "hadm_id", "stay_id"]].copy()
                return features
            
            self.logger.info(f"Creating pivot table for {len(window_vitals)} vital sign measurements")
            
            # Use pivot_table for efficient reshaping - this replaces the loop over ICU stays
            vital_features = pd.pivot_table(
                window_vitals,
                values="valuenum",
                index=["subject_id", "hadm_id", "stay_id"],
                columns="category",
                aggfunc=aggregation_methods
            )
            
            # Flatten the multi-level column names
            if isinstance(vital_features.columns, pd.MultiIndex):
                vital_features.columns = [f"{col[1]}_{col[0]}" if isinstance(col, tuple) and len(col) > 1 else col for col in vital_features.columns]
            
            # Reset index to convert back to a regular dataframe
            vital_features = vital_features.reset_index()
            
            # Ensure all ICU stays are included, even those without vital data
            all_stays = icustays[["subject_id", "hadm_id", "stay_id"]].copy()
            features = pd.merge(
                all_stays,
                vital_features,
                on=["subject_id", "hadm_id", "stay_id"],
                how="left"
            )
            
            self.logger.info(f"Created vital sign features with {features.shape[1] - 3} measurements for {features.shape[0]} ICU stays")
            
            return features
        except Exception as e:
            self.logger.warning(f"Error in _process_vital_data: {e}")
            # Return a dataframe with just the ID columns
            return icustays[["subject_id", "hadm_id", "stay_id"]].copy()


class DiagnosisFeatureExtractor(BaseFeatureExtractor):
    """
    Extractor for diagnosis features.
    """
    
    def extract(self) -> pd.DataFrame:
        """
        Extract diagnosis features.
        
        Returns:
            pd.DataFrame: Diagnosis features
        """
        self.logger.info("Extracting diagnosis features")
        
        # Load admission data
        admission_path = get_data_path("processed", "admission_data", self.config)
        admissions = pd.read_csv(admission_path)
        
        # Load MIMIC-III diagnosis data
        try:
            mimic3_path = get_data_path("raw", "mimic_iii", self.config)
            diagnosis_path = os.path.join(mimic3_path, "DIAGNOSES_ICD.csv")
            
            if os.path.exists(diagnosis_path):
                # Load diagnosis data
                diagnoses = pd.read_csv(diagnosis_path)
                
                # Convert column names to lowercase
                diagnoses.columns = diagnoses.columns.str.lower()
                
                # Load diagnosis dictionary
                icd_path = os.path.join(mimic3_path, "D_ICD_DIAGNOSES.csv")
                icd_diagnoses = pd.read_csv(icd_path)
                icd_diagnoses.columns = icd_diagnoses.columns.str.lower()
                
                # Process diagnosis data
                diagnosis_features = self._process_diagnosis_data(diagnoses, icd_diagnoses, admissions)
            else:
                self.logger.warning("MIMIC-III diagnosis data not found")
                diagnosis_features = pd.DataFrame(columns=["subject_id", "hadm_id"])
        except Exception as e:
            self.logger.warning(f"Error loading MIMIC-III diagnosis data: {e}")
            diagnosis_features = pd.DataFrame(columns=["subject_id", "hadm_id"])
        
        return diagnosis_features
    
    def _process_diagnosis_data(
        self, 
        diagnoses: pd.DataFrame, 
        icd_diagnoses: pd.DataFrame, 
        admissions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process diagnosis data and extract features.
        
        Args:
            diagnoses (pd.DataFrame): Diagnosis data
            icd_diagnoses (pd.DataFrame): ICD diagnosis dictionary
            admissions (pd.DataFrame): Admission data
        
        Returns:
            pd.DataFrame: Diagnosis features
        """
        # Merge diagnoses with ICD codes to get descriptions
        diagnoses = pd.merge(
            diagnoses, 
            icd_diagnoses[["icd9_code", "short_title"]], 
            on="icd9_code", 
            how="left"
        )
        
        # Create diagnosis categories based on ICD-9 chapters
        diagnoses["category"] = diagnoses["icd9_code"].apply(self._get_icd9_category)
        
        # One-hot encode diagnoses by category
        features = pd.DataFrame()
        
        # Add identifiers
        features["subject_id"] = admissions["subject_id"]
        features["hadm_id"] = admissions["hadm_id"]
        
        # Create binary features for each diagnosis category
        for subject_id, hadm_id in tqdm(
            zip(admissions["subject_id"], admissions["hadm_id"]),
            desc="Processing admissions for diagnosis features",
            total=len(admissions)
        ):
            # Get diagnoses for this admission
            admission_diagnoses = diagnoses[
                (diagnoses["subject_id"] == subject_id) & 
                (diagnoses["hadm_id"] == hadm_id)
            ]
            
            if len(admission_diagnoses) == 0:
                continue
            
            # Get unique categories for this admission
            categories = admission_diagnoses["category"].unique()
            
            # Update features for this admission
            for category in categories:
                if category:  # Skip None/NaN categories
                    col_name = f"diagnosis_{category}"
                    features.loc[
                        (features["subject_id"] == subject_id) & 
                        (features["hadm_id"] == hadm_id), 
                        col_name
                    ] = 1
        
        # Fill NaN values with 0 (no diagnosis in that category)
        diagnosis_cols = [col for col in features.columns if col.startswith("diagnosis_")]
        features[diagnosis_cols] = features[diagnosis_cols].fillna(0)
        
        return features
    
    def _get_icd9_category(self, icd9_code: str) -> Optional[str]:
        """
        Get the category for an ICD-9 code using the mappings configuration.
        
        Args:
            icd9_code (str): ICD-9 code
        
        Returns:
            Optional[str]: Category name or None if invalid code
        """
        from src.utils import load_mappings
        
        if not isinstance(icd9_code, str):
            return None
        
        # Remove dots if present
        icd9_code = icd9_code.replace(".", "")
        
        # Load ICD-9 category mappings
        try:
            # Cache the mappings to avoid loading them for each code
            if not hasattr(self, '_icd9_mappings'):
                mappings = load_mappings()
                self._icd9_mappings = mappings.get('icd9_categories', {})
            
            # Check for specific codes first (more precise categorization)
            specific_codes = self._icd9_mappings.get('specific_codes', {})
            for category, codes in specific_codes.items():
                if icd9_code in codes:
                    return category.lower().replace(" ", "_")
            
            # Then check ranges for general categorization
            if icd9_code.startswith("V"):
                return "supplementary"
            elif icd9_code.startswith("E"):
                return "external_causes"
            else:
                # Convert to integer for numeric codes
                try:
                    code_num = int(icd9_code)
                    
                    # Check each range in the mappings
                    ranges = self._icd9_mappings.get('ranges', {})
                    for category, range_list in ranges.items():
                        for range_pair in range_list:
                            if len(range_pair) == 2 and range_pair[0] <= code_num <= range_pair[1]:
                                return category.lower().replace(" ", "_")
                    
                    # If no match found
                    return "other"
                except ValueError:
                    # If code cannot be converted to integer
                    return None
        except Exception as e:
            self.logger.warning(f"Error categorizing ICD-9 code {icd9_code}: {e}")
            
            # Fallback to hardcoded categorization if mappings fail
            try:
                if icd9_code.startswith("V"):
                    return "supplementary"
                elif icd9_code.startswith("E"):
                    return "external_causes"
                else:
                    # Convert to integer for numeric codes
                    code_num = int(icd9_code)
                    
                    # Categorize based on ICD-9 chapters
                    if 1 <= code_num <= 139:
                        return "infectious"
                    elif 140 <= code_num <= 239:
                        return "neoplasms"
                    elif 240 <= code_num <= 279:
                        return "endocrine"
                    elif 280 <= code_num <= 289:
                        return "blood"
                    elif 290 <= code_num <= 319:
                        return "mental"
                    elif 320 <= code_num <= 389:
                        return "nervous"
                    elif 390 <= code_num <= 459:
                        return "circulatory"
                    elif 460 <= code_num <= 519:
                        return "respiratory"
                    elif 520 <= code_num <= 579:
                        return "digestive"
                    elif 580 <= code_num <= 629:
                        return "genitourinary"
                    elif 630 <= code_num <= 679:
                        return "pregnancy"
                    elif 680 <= code_num <= 709:
                        return "skin"
                    elif 710 <= code_num <= 739:
                        return "musculoskeletal"
                    elif 740 <= code_num <= 759:
                        return "congenital"
                    elif 760 <= code_num <= 779:
                        return "perinatal"
                    elif 780 <= code_num <= 799:
                        return "symptoms"
                    elif 800 <= code_num <= 999:
                        return "injury"
                    else:
                        return "other"
            except (ValueError, TypeError):
                return None