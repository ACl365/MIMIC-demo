"""
Data processors for the MIMIC datasets.

This module contains classes for processing raw MIMIC-III and potentially MIMIC-IV
data into structured formats suitable for feature extraction and modeling.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

from utils import get_data_path, get_logger, load_config  # Corrected direct import

logger = get_logger(__name__)


class BaseProcessor(ABC):
    """
    Abstract base class for data processors.

    Provides a common structure for loading configuration, processing data,
    and saving the results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the processor, loading configuration.

        Args:
            config (Optional[Dict], optional): Configuration dictionary.
                If None, loads the default configuration using load_config().
                Defaults to None.
        """
        self.config = config if config is not None else load_config()
        self.logger = logger

    @abstractmethod
    def process(self) -> pd.DataFrame:
        """
        Abstract method to process the specific data type.

        Must be implemented by subclasses.

        Returns:
            pd.DataFrame: The processed data as a pandas DataFrame.
        """
        pass

    def save(self, data: pd.DataFrame, output_path: str) -> None:
        """
        Save the processed DataFrame to a CSV file.

        Creates the output directory if it doesn't exist.

        Args:
            data (pd.DataFrame): The DataFrame to save.
            output_path (str): The full path (including filename) to save the data to.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save data
        data.to_csv(output_path, index=False)
        self.logger.info(f"Saved processed data to {output_path}")


class PatientProcessor(BaseProcessor):
    """
    Processor for consolidating and cleaning patient demographic data.

    Combines patient information from MIMIC-III and MIMIC-IV (if available),
    calculates approximate age for MIMIC-III, and standardizes column names
    and formats.
    """

    def process(self) -> pd.DataFrame:
        """
        Loads, processes, and combines patient data from MIMIC-III and MIMIC-IV.

        Returns:
            pd.DataFrame: A DataFrame containing combined and processed patient data
                          with columns like 'subject_id', 'gender', 'age', 'source'.
        """
        self.logger.info("Processing patient data")

        # Load MIMIC-III patient data
        mimic3_path = get_data_path("raw", "mimic_iii", self.config)
        mimic3_patients = pd.read_csv(
            os.path.join(mimic3_path, "PATIENTS.csv"),
            parse_dates=["dob", "dod", "dod_hosp", "dod_ssn"],
        )

        # Process MIMIC-III patient data
        mimic3_patients = self._process_mimic3_patients(mimic3_patients)

        # Try to load MIMIC-IV patient data if available
        try:
            mimic4_path = get_data_path("raw", "mimic_iv", self.config)
            # Adjust path based on MIMIC-IV structure (often nested)
            mimic4_patients_path = os.path.join(
                mimic4_path, "hosp", "patients.csv"  # Common MIMIC-IV structure
            )
            # Fallback if not in 'hosp' subdir
            if not os.path.exists(mimic4_patients_path):
                mimic4_patients_path = os.path.join(mimic4_path, "patients.csv")

            if os.path.exists(mimic4_patients_path):
                mimic4_patients = pd.read_csv(
                    mimic4_patients_path,
                    # MIMIC-IV uses anchor_year, anchor_year_group
                    parse_dates=["dod"],  # Only DOD is typically a full date
                )

                # Process MIMIC-IV patient data
                mimic4_patients = self._process_mimic4_patients(mimic4_patients)

                # Combine datasets
                patients = self._combine_patient_data(mimic3_patients, mimic4_patients)
            else:
                self.logger.warning(
                    f"MIMIC-IV patient data not found at expected path: {mimic4_patients_path}. Using only MIMIC-III data."
                )
                patients = mimic3_patients
        except Exception as e:
            self.logger.warning(
                f"Error loading or processing MIMIC-IV patient data: {e}", exc_info=True
            )
            self.logger.warning("Using only MIMIC-III patient data")
            patients = mimic3_patients

        return patients

    def _process_mimic3_patients(self, patients: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw MIMIC-III patient data.

        Converts column names to lowercase, adds a 'source' column, calculates
        approximate age (handling potential date overflows), and standardizes
        gender format.

        Args:
            patients (pd.DataFrame): Raw MIMIC-III PATIENTS.csv data.

        Returns:
            pd.DataFrame: Processed MIMIC-III patient data.
        """
        # Convert column names to lowercase for consistency
        patients.columns = patients.columns.str.lower()

        # Add source column
        patients["source"] = "mimic_iii"

        # Calculate age at admission (approximate since dates are shifted)
        # In MIMIC, dates are shifted but intervals are preserved
        try:
            # Try to calculate age from dob and dod
            # Note: This age is approximate and might represent age at death for some.
            # A more accurate age calculation requires linking with admission time.
            patients["age"] = (patients["dod"] - patients["dob"]).dt.days / 365.25
            # Handle biologically implausible ages (e.g., > 90 often due to date shifts)
            # MIMIC-III documentation suggests ages > 89 are shifted to 300.
            patients.loc[patients["age"] > 90, "age"] = 90
        except (OverflowError, pd.errors.OutOfBoundsDatetime, TypeError) as e:
            # If there's an error (e.g., dob/dod missing), handle gracefully
            self.logger.warning(
                f"Error calculating age from dates ({e}). Age column may contain NaNs or be incomplete."
            )
            # Ensure 'age' column exists even if calculation failed
            if "age" not in patients.columns:
                patients["age"] = pd.NA

        # Clean up gender
        patients["gender"] = patients["gender"].str.upper()

        # Select and rename columns for consistency
        patients = patients[["subject_id", "gender", "dob", "dod", "source", "age"]]

        return patients

    def _process_mimic4_patients(self, patients: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw MIMIC-IV patient data.

        Adds a 'source' column, uses 'anchor_age' for age, and standardizes
        gender format. Selects a subset of columns for consistency.

        Args:
            patients (pd.DataFrame): Raw MIMIC-IV patients.csv data.

        Returns:
            pd.DataFrame: Processed MIMIC-IV patient data.
        """
        # Convert column names to lowercase for consistency
        patients.columns = patients.columns.str.lower()

        # Add source column
        patients["source"] = "mimic_iv"

        # Calculate age (MIMIC-IV provides anchor_age)
        if "anchor_age" in patients.columns:
            patients["age"] = patients["anchor_age"]
        else:
            self.logger.warning("MIMIC-IV data missing 'anchor_age'. Age will be NaN.")
            patients["age"] = pd.NA

        # Clean up gender
        if "gender" in patients.columns:
            patients["gender"] = patients["gender"].str.upper()

        # Select and rename columns for consistency
        # Note: MIMIC-IV doesn't have DOB directly available in patients.csv
        patients = patients[
            ["subject_id", "gender", "dod", "source", "age", "anchor_year"]
        ]  # Keep anchor_year for potential future use

        return patients

    def _combine_patient_data(
        self, mimic3_patients: pd.DataFrame, mimic4_patients: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine processed MIMIC-III and MIMIC-IV patient data.

        Selects common columns and concatenates the DataFrames.

        Args:
            mimic3_patients (pd.DataFrame): Processed MIMIC-III patient data.
            mimic4_patients (pd.DataFrame): Processed MIMIC-IV patient data.

        Returns:
            pd.DataFrame: Combined patient data.
        """
        # Identify common columns, prioritizing MIMIC-III names if different but conceptually same
        # For patients, columns are generally aligned after processing.
        common_columns = list(
            set(mimic3_patients.columns) & set(mimic4_patients.columns)
        )

        # Ensure essential columns are present
        essential_cols = ["subject_id", "gender", "age", "dod", "source"]
        for col in essential_cols:
            if col not in common_columns:
                self.logger.warning(
                    f"Essential column '{col}' missing from common columns during patient data combination."
                )
                # Decide how to handle: add if missing in one? For now, proceed with intersection.

        self.logger.info(f"Combining patient data using columns: {common_columns}")

        # Keep only common columns
        mimic3_subset = mimic3_patients[common_columns]
        mimic4_subset = mimic4_patients[common_columns]

        # Combine datasets
        patients = pd.concat([mimic3_subset, mimic4_subset], ignore_index=True)

        # Handle potential duplicates (e.g., if a subject_id exists in both)
        # Strategy: Keep the first occurrence (could be MIMIC-III or IV depending on concat order)
        # A more robust strategy might involve prioritizing one source or merging details.
        patients = patients.drop_duplicates(subset=["subject_id"], keep="first")
        self.logger.info(
            f"Combined patient data shape after dropping duplicates: {patients.shape}"
        )

        return patients


class AdmissionProcessor(BaseProcessor):
    """
    Processor for consolidating and cleaning admission-level data.

    Combines admission information from MIMIC-III and MIMIC-IV (if available),
    calculates derived features like length of stay and hospital death,
    and identifies readmissions within 30 and 90 days.
    """

    def process(self) -> pd.DataFrame:
        """
        Loads, processes, and combines admission data from MIMIC-III and MIMIC-IV.

        Calculates length of stay, hospital death flag, and readmission flags.

        Returns:
            pd.DataFrame: A DataFrame containing combined and processed admission data,
                          with one row per unique hospital admission. Includes derived
                          features like 'los_days', 'hospital_death', 'readmission_30day',
                          'readmission_90day', 'days_to_readmission'.
        """
        self.logger.info("Processing admission data")

        # Load MIMIC-III admission data
        mimic3_path = get_data_path("raw", "mimic_iii", self.config)
        mimic3_admissions = pd.read_csv(
            os.path.join(mimic3_path, "ADMISSIONS.csv"),
            parse_dates=[
                "admittime",
                "dischtime",
                "deathtime",
                "edregtime",
                "edouttime",
            ],
        )

        # Process MIMIC-III admission data
        mimic3_admissions = self._process_mimic3_admissions(mimic3_admissions)

        # Try to load MIMIC-IV admission data if available
        try:
            mimic4_path = get_data_path("raw", "mimic_iv", self.config)
            # Adjust path based on MIMIC-IV structure
            mimic4_admissions_path = os.path.join(mimic4_path, "hosp", "admissions.csv")
            if not os.path.exists(mimic4_admissions_path):
                mimic4_admissions_path = os.path.join(mimic4_path, "admissions.csv")

            if os.path.exists(mimic4_admissions_path):
                mimic4_admissions = pd.read_csv(
                    mimic4_admissions_path,
                    parse_dates=[
                        "admittime",
                        "dischtime",
                        "deathtime",
                        "edregtime",
                        "edouttime",
                    ],
                )

                # Process MIMIC-IV admission data
                mimic4_admissions = self._process_mimic4_admissions(mimic4_admissions)

                # Combine datasets
                admissions = self._combine_admission_data(
                    mimic3_admissions, mimic4_admissions
                )
            else:
                self.logger.warning(
                    f"MIMIC-IV admission data not found at expected path: {mimic4_admissions_path}. Using only MIMIC-III data."
                )
                admissions = mimic3_admissions
        except Exception as e:
            self.logger.warning(
                f"Error loading or processing MIMIC-IV admission data: {e}",
                exc_info=True,
            )
            self.logger.warning("Using only MIMIC-III admission data")
            admissions = mimic3_admissions

        # Calculate length of stay and other derived features
        admissions = self._calculate_derived_features(admissions)

        # Identify readmissions
        admissions = self._identify_readmissions(admissions)

        return admissions

    def _process_mimic3_admissions(self, admissions: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw MIMIC-III admission data.

        Converts column names to lowercase, adds a 'source' column, and standardizes
        categorical variable formats (uppercase).

        Args:
            admissions (pd.DataFrame): Raw MIMIC-III ADMISSIONS.csv data.

        Returns:
            pd.DataFrame: Processed MIMIC-III admission data.
        """
        # Convert column names to lowercase for consistency
        admissions.columns = admissions.columns.str.lower()

        # Add source column
        admissions["source"] = "mimic_iii"

        # Clean up categorical variables
        admissions["admission_type"] = admissions["admission_type"].str.upper()
        admissions["discharge_location"] = admissions["discharge_location"].str.upper()
        # Add others if needed, e.g., insurance, marital_status

        return admissions

    def _process_mimic4_admissions(self, admissions: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw MIMIC-IV admission data.

        Adds a 'source' column, standardizes categorical variable formats,
        and renames 'ethnicity' to 'race' for consistency with MIMIC-III naming
        if 'race' doesn't already exist.

        Args:
            admissions (pd.DataFrame): Raw MIMIC-IV admissions.csv data.

        Returns:
            pd.DataFrame: Processed MIMIC-IV admission data.
        """
        # Convert column names to lowercase for consistency
        admissions.columns = admissions.columns.str.lower()

        # Add source column
        admissions["source"] = "mimic_iv"

        # Clean up categorical variables
        if "admission_type" in admissions.columns:
            admissions["admission_type"] = admissions["admission_type"].str.upper()
        if "discharge_location" in admissions.columns:
            admissions["discharge_location"] = admissions[
                "discharge_location"
            ].str.upper()
        # Add others if needed

        # Rename columns to match MIMIC-III if needed
        if "ethnicity" in admissions.columns and "race" not in admissions.columns:
            admissions = admissions.rename(columns={"ethnicity": "race"})

        return admissions

    def _combine_admission_data(
        self, mimic3_admissions: pd.DataFrame, mimic4_admissions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine processed MIMIC-III and MIMIC-IV admission data.

        Selects common columns and concatenates the DataFrames. Handles potential
        duplicate admissions by keeping the first occurrence.

        Args:
            mimic3_admissions (pd.DataFrame): Processed MIMIC-III admission data.
            mimic4_admissions (pd.DataFrame): Processed MIMIC-IV admission data.

        Returns:
            pd.DataFrame: Combined admission data.
        """
        # Identify common columns
        common_columns = list(
            set(mimic3_admissions.columns) & set(mimic4_admissions.columns)
        )

        # Ensure essential columns are present
        essential_cols = ["subject_id", "hadm_id", "admittime", "dischtime", "source"]
        for col in essential_cols:
            if col not in common_columns:
                self.logger.warning(
                    f"Essential column '{col}' missing from common columns during admission data combination."
                )

        self.logger.info(f"Combining admission data using columns: {common_columns}")

        # Keep only common columns
        mimic3_subset = mimic3_admissions[common_columns]
        mimic4_subset = mimic4_admissions[common_columns]

        # Combine datasets
        admissions = pd.concat([mimic3_subset, mimic4_subset], ignore_index=True)

        # Handle potential duplicates (e.g., if a hadm_id exists in both)
        admissions = admissions.drop_duplicates(
            subset=["subject_id", "hadm_id"], keep="first"
        )
        self.logger.info(
            f"Combined admission data shape after dropping duplicates: {admissions.shape}"
        )

        return admissions

    def _calculate_derived_features(self, admissions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features from admission data.

        Computes 'los_days' (length of stay), 'ed_los_hours' (ED length of stay,
        if ED times available), and 'hospital_death' flag (using
        'hospital_expire_flag' if available, otherwise inferring from 'deathtime').

        Args:
            admissions (pd.DataFrame): Admission data with 'admittime', 'dischtime',
                                       and potentially 'edregtime', 'edouttime',
                                       'hospital_expire_flag', 'deathtime'.

        Returns:
            pd.DataFrame: Admission data with added derived feature columns.
        """
        # Calculate length of stay in days
        if "dischtime" in admissions.columns and "admittime" in admissions.columns:
            admissions["los_days"] = (
                admissions["dischtime"] - admissions["admittime"]
            ).dt.total_seconds() / (24 * 60 * 60)
        else:
            self.logger.warning(
                "Cannot calculate 'los_days': missing 'dischtime' or 'admittime'."
            )
            admissions["los_days"] = pd.NA

        # Calculate ED length of stay in hours (if available)
        if "edregtime" in admissions.columns and "edouttime" in admissions.columns:
            # Only calculate for rows where both values are not null
            ed_mask = ~(admissions["edregtime"].isna() | admissions["edouttime"].isna())
            admissions.loc[ed_mask, "ed_los_hours"] = (
                admissions.loc[ed_mask, "edouttime"]
                - admissions.loc[ed_mask, "edregtime"]
            ).dt.total_seconds() / (60 * 60)
            admissions["ed_los_hours"] = admissions["ed_los_hours"].fillna(
                pd.NA
            )  # Fill non-calculable rows with NA
        else:
            admissions["ed_los_hours"] = (
                pd.NA
            )  # Column doesn't exist if times aren't present

        # Flag in-hospital deaths
        # Ensure 'hospital_expire_flag' exists before using it
        if "hospital_expire_flag" in admissions.columns:
            admissions["hospital_death"] = admissions["hospital_expire_flag"].astype(
                bool
            )
        elif "deathtime" in admissions.columns:
            # Handle case where flag is missing (e.g., set to False or based on 'deathtime')
            # Check if deathtime falls within the admission period (or slightly after discharge)
            admissions["hospital_death"] = (~admissions["deathtime"].isna()) & (
                admissions["deathtime"] <= admissions["dischtime"]
            )  # Basic check
            self.logger.warning(
                "'hospital_expire_flag' not found. Inferring 'hospital_death' from 'deathtime' <= 'dischtime'."
            )
        else:
            self.logger.warning(
                "Cannot determine 'hospital_death': missing 'hospital_expire_flag' and 'deathtime'. Setting to False."
            )
            admissions["hospital_death"] = False

        return admissions

    def _identify_readmissions(self, admissions: pd.DataFrame) -> pd.DataFrame:
        """
        Identify readmissions within 30 and 90 days using vectorized operations.

        Calculates 'days_to_readmission' between consecutive admissions for the
        same patient. Flags 'readmission_30day' and 'readmission_90day' if the
        days_to_readmission is between 1 and 30/90 days, respectively. Excludes
        admissions that resulted in hospital death.

        Args:
            admissions (pd.DataFrame): Admission data, sorted by subject_id and admittime.
                                       Must contain 'subject_id', 'admittime',
                                       'dischtime', 'hospital_death'.

        Returns:
            pd.DataFrame: Admission data with added readmission flag columns
                          ('readmission_30day', 'readmission_90day') and
                          'days_to_readmission'.
        """
        self.logger.info("Identifying readmissions using vectorized operations")

        # Ensure necessary columns exist
        required_cols = ["subject_id", "admittime", "dischtime", "hospital_death"]
        if not all(col in admissions.columns for col in required_cols):
            missing = [col for col in required_cols if col not in admissions.columns]
            self.logger.error(
                f"Missing required columns for readmission calculation: {missing}"
            )
            # Return dataframe without readmission flags or raise error
            admissions["readmission_30day"] = False
            admissions["readmission_90day"] = False
            admissions["days_to_readmission"] = float("nan")
            return admissions

        # Ensure sorting
        admissions = admissions.sort_values(["subject_id", "admittime"])

        # Initialize readmission columns
        admissions["readmission_30day"] = False
        admissions["readmission_90day"] = False
        admissions["days_to_readmission"] = float("nan")

        # Create shifted columns for the next admission's data within each patient group
        admissions["next_admittime"] = admissions.groupby("subject_id")[
            "admittime"
        ].shift(-1)
        admissions["next_subject_id"] = admissions.groupby("subject_id")[
            "subject_id"
        ].shift(
            -1
        )  # To confirm it's the same patient

        # Calculate days between discharge and next admission only for valid rows
        # Valid rows: Same patient, current admission did not result in death, next admission exists
        valid_rows = (
            (admissions["subject_id"] == admissions["next_subject_id"])
            & (~admissions["hospital_death"])
            & (admissions["next_admittime"].notna())
            & (admissions["dischtime"].notna())  # Ensure discharge time is not null
        )

        # Calculate days_to_readmission for valid rows
        admissions.loc[valid_rows, "days_to_readmission"] = (
            admissions.loc[valid_rows, "next_admittime"]
            - admissions.loc[valid_rows, "dischtime"]
        ).dt.total_seconds() / (24 * 60 * 60)

        # Flag readmissions within time windows (only if days >= 1)
        # Using .loc ensures alignment even if index is not sequential
        readmit_30_mask = (
            valid_rows
            & (admissions["days_to_readmission"] >= 1)
            & (admissions["days_to_readmission"] <= 30)
        )
        readmit_90_mask = (
            valid_rows
            & (admissions["days_to_readmission"] >= 1)
            & (admissions["days_to_readmission"] <= 90)
        )

        admissions.loc[readmit_30_mask, "readmission_30day"] = True
        admissions.loc[readmit_90_mask, "readmission_90day"] = True

        # Drop helper columns
        admissions = admissions.drop(columns=["next_admittime", "next_subject_id"])

        # Log statistics
        self.logger.info(
            f"Identified {admissions['readmission_30day'].sum()} 30-day readmissions"
        )
        self.logger.info(
            f"Identified {admissions['readmission_90day'].sum()} 90-day readmissions"
        )

        return admissions


class ICUStayProcessor(BaseProcessor):
    """
    Processor for ICU stay data.

    Combines ICU stay information from MIMIC-III and MIMIC-IV (if available),
    calculates length of stay for each ICU visit, and standardizes column names.
    """

    def process(self) -> pd.DataFrame:
        """
        Loads, processes, and combines ICU stay data from MIMIC-III and MIMIC-IV.

        Returns:
            pd.DataFrame: A DataFrame containing combined and processed ICU stay data,
                          with one row per unique ICU stay. Includes 'los' (length of stay in days).
        """
        self.logger.info("Processing ICU stay data")

        # Load MIMIC-III ICU stay data
        mimic3_path = get_data_path("raw", "mimic_iii", self.config)
        mimic3_icustays = pd.read_csv(
            os.path.join(mimic3_path, "ICUSTAYS.csv"), parse_dates=["intime", "outtime"]
        )

        # Process MIMIC-III ICU stay data
        mimic3_icustays = self._process_mimic3_icustays(mimic3_icustays)

        # Try to load MIMIC-IV ICU stay data if available
        try:
            mimic4_path = get_data_path("raw", "mimic_iv", self.config)
            # Adjust path based on MIMIC-IV structure
            mimic4_icustays_path = os.path.join(
                mimic4_path, "icu", "icustays.csv"  # Common MIMIC-IV structure
            )
            if not os.path.exists(mimic4_icustays_path):
                mimic4_icustays_path = os.path.join(mimic4_path, "icustays.csv")

            if os.path.exists(mimic4_icustays_path):
                mimic4_icustays = pd.read_csv(
                    mimic4_icustays_path, parse_dates=["intime", "outtime"]
                )

                # Process MIMIC-IV ICU stay data
                mimic4_icustays = self._process_mimic4_icustays(mimic4_icustays)

                # Combine datasets
                icustays = self._combine_icustay_data(mimic3_icustays, mimic4_icustays)
            else:
                self.logger.warning(
                    f"MIMIC-IV icustay data not found at expected path: {mimic4_icustays_path}. Using only MIMIC-III data."
                )
                icustays = mimic3_icustays
        except Exception as e:
            self.logger.warning(
                f"Error loading or processing MIMIC-IV icustay data: {e}", exc_info=True
            )
            self.logger.warning("Using only MIMIC-III icustay data")
            icustays = mimic3_icustays

        return icustays

    def _process_mimic3_icustays(self, icustays: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw MIMIC-III ICU stay data.

        Converts column names to lowercase, adds a 'source' column, and calculates
        length of stay ('los').

        Args:
            icustays (pd.DataFrame): Raw MIMIC-III ICUSTAYS.csv data.

        Returns:
            pd.DataFrame: Processed MIMIC-III ICU stay data.
        """
        # Convert column names to lowercase
        icustays.columns = icustays.columns.str.lower()

        # Add source column
        icustays["source"] = "mimic_iii"

        # Calculate length of stay (LOS) in days
        icustays["los"] = (
            icustays["outtime"] - icustays["intime"]
        ).dt.total_seconds() / (24 * 60 * 60)

        # Select relevant columns
        icustays = icustays[
            [
                "subject_id",
                "hadm_id",
                "icustay_id",
                "intime",
                "outtime",
                "los",
                "source",
            ]
        ]
        return icustays

    def _process_mimic4_icustays(self, icustays: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw MIMIC-IV ICU stay data.

        Converts column names to lowercase, adds a 'source' column, and uses the
        pre-calculated 'los' column if available.

        Args:
            icustays (pd.DataFrame): Raw MIMIC-IV icustays.csv data.

        Returns:
            pd.DataFrame: Processed MIMIC-IV ICU stay data.
        """
        # Convert column names to lowercase
        icustays.columns = icustays.columns.str.lower()

        # Add source column
        icustays["source"] = "mimic_iv"

        # MIMIC-IV often has 'los' pre-calculated
        if (
            "los" not in icustays.columns
            and "outtime" in icustays.columns
            and "intime" in icustays.columns
        ):
            icustays["los"] = (
                icustays["outtime"] - icustays["intime"]
            ).dt.total_seconds() / (24 * 60 * 60)
            self.logger.info("Calculated 'los' for MIMIC-IV ICU stays.")

        # Select relevant columns (ensure consistency with MIMIC-III processing)
        # Rename stay_id if needed (MIMIC-IV sometimes uses icustay_id)
        if "icustay_id" in icustays.columns and "stay_id" not in icustays.columns:
            icustays = icustays.rename(columns={"icustay_id": "stay_id"})

        cols_to_keep = [
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "los",
            "source",
        ]
        missing_cols = [col for col in cols_to_keep if col not in icustays.columns]
        if missing_cols:
            self.logger.warning(
                f"Missing expected columns in MIMIC-IV ICU data: {missing_cols}"
            )
            # Add missing columns with NaNs if necessary before selection
            for col in missing_cols:
                icustays[col] = pd.NA

        return icustays[cols_to_keep]

    def _combine_icustay_data(
        self, mimic3_icustays: pd.DataFrame, mimic4_icustays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine processed MIMIC-III and MIMIC-IV ICU stay data.

        Selects common columns and concatenates the DataFrames. Handles potential
        duplicate stays by keeping the first occurrence.

        Args:
            mimic3_icustays (pd.DataFrame): Processed MIMIC-III ICU stay data.
            mimic4_icustays (pd.DataFrame): Processed MIMIC-IV ICU stay data.

        Returns:
            pd.DataFrame: Combined ICU stay data.
        """
        # Identify common columns
        common_columns = list(
            set(mimic3_icustays.columns) & set(mimic4_icustays.columns)
        )

        self.logger.info(f"Combining ICU stay data using columns: {common_columns}")

        # Keep only common columns
        mimic3_subset = mimic3_icustays[common_columns]
        mimic4_subset = mimic4_icustays[common_columns]

        # Combine datasets
        icustays = pd.concat([mimic3_subset, mimic4_subset], ignore_index=True)

        # Handle potential duplicates (e.g., if a stay_id exists in both)
        icustays = icustays.drop_duplicates(
            subset=["subject_id", "hadm_id", "stay_id"], keep="first"
        )
        self.logger.info(
            f"Combined ICU stay data shape after dropping duplicates: {icustays.shape}"
        )

        return icustays
