"""
Model classes for the MIMIC project.
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple  # Removed Any, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # Removed cross_val_score
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm  # Added for progress bar in preprocess

from src.models.temporal_modeling import (  # Import PyTorch components; Add other necessary imports from temporal_modeling.py if needed
    TemporalEHRDataset,
    TimeAwarePatientLSTM,
)

# Local application imports
from src.utils import get_logger, load_config  # Removed get_data_path, get_project_root

logger = get_logger(__name__)


def _reshape_data_to_admission_level(
    data: pd.DataFrame, id_cols: List[str], target_cols: List[str], logger_instance
) -> pd.DataFrame:
    """
    Reshapes data from potentially multiple rows per admission to one row per admission.
    Handles pivoting for clinical features and aggregation for demographic features.
    """
    logger_instance.info("Detected multiple rows per admission. Restructuring data...")

    # Ensure required ID columns exist
    if "subject_id" not in data.columns or "hadm_id" not in data.columns:
        raise ValueError(
            "Input data must contain 'subject_id' and 'hadm_id' for reshaping."
        )

    # --- 1. Handle Base Identifiers and Targets ---
    base_cols = list(set(id_cols + target_cols))  # Combine and unique
    existing_base_cols = [col for col in base_cols if col in data.columns]

    # Group by admission and take the first non-null value for base columns
    # This assumes these values are constant per admission
    agg_dict_base = {
        col: "first"
        for col in existing_base_cols
        if col not in ["subject_id", "hadm_id"]
    }

    if not agg_dict_base:  # Handle case where only IDs are present initially
        admission_base = (
            data[["subject_id", "hadm_id"]]
            .drop_duplicates()
            .set_index(["subject_id", "hadm_id"])
        )
    else:
        admission_base = data.groupby(["subject_id", "hadm_id"]).agg(agg_dict_base)

    logger_instance.debug(f"Aggregated base columns: {admission_base.columns.tolist()}")

    # --- 2. Identify Feature Columns ---
    all_cols = set(data.columns)
    base_cols_set = set(existing_base_cols)
    feature_cols = list(all_cols - base_cols_set)
    logger_instance.debug(f"Identified {len(feature_cols)} potential feature columns.")

    # --- 3. Identify Demographic Features (Assume they don't need pivoting) ---
    # Define typical demographic prefixes/names (adjust as needed)
    demographic_indicators = [
        "gender_",
        "age",
        "admission_type_",
        "insurance_",
        "marital_status_",
        "ethnicity_",
        "race_",
    ]
    demographic_cols = [
        col
        for col in feature_cols
        if any(indicator in col for indicator in demographic_indicators)
    ]
    logger_instance.debug(f"Identified demographic columns: {demographic_cols}")

    # Aggregate demographic features (take first non-null value per admission)
    if demographic_cols:
        agg_dict_demo = {col: "first" for col in demographic_cols}
        admission_demo = data.groupby(["subject_id", "hadm_id"]).agg(agg_dict_demo)
        logger_instance.debug(
            f"Aggregated demographic features shape: {admission_demo.shape}"
        )
    else:
        admission_demo = pd.DataFrame(
            index=admission_base.index
        )  # Empty DF with correct index

    # --- 4. Identify Clinical/Other Features for Pivoting ---
    pivot_candidate_cols = list(set(feature_cols) - set(demographic_cols))

    # Check if pivoting based on 'category' and 'level_0' is feasible
    pivot_features = pd.DataFrame(index=admission_base.index)  # Initialize empty
    if (
        "category" in data.columns
        and "level_0" in data.columns
        and "0" in data.columns
        and pivot_candidate_cols
    ):
        logger_instance.info(
            "Attempting pivot based on 'category' and 'level_0' columns..."
        )

        # Select only necessary columns for pivoting to save memory
        pivot_data = data[["subject_id", "hadm_id", "category", "level_0", "0"]].copy()
        pivot_data["feature_id"] = (
            pivot_data["category"] + "_" + pivot_data["level_0"].astype(str)
        )

        try:
            pivot_features_temp = pivot_data.pivot_table(
                index=["subject_id", "hadm_id"],
                columns="feature_id",
                values="0",
                aggfunc="first",  # Or 'mean', 'max' etc. if appropriate
            )
            # Merge with base index to ensure all admissions are present, fill NaNs later
            pivot_features = pd.merge(
                admission_base[[]],
                pivot_features_temp,
                left_index=True,
                right_index=True,
                how="left",
            )
            logger_instance.info(
                f"Successfully pivoted clinical features. Shape: {pivot_features.shape}"
            )

        except Exception as e:
            logger_instance.warning(
                f"Pivoting failed: {e}. Clinical features might be missing or incomplete."
            )
            pivot_features = pd.DataFrame(
                index=admission_base.index
            )  # Ensure it's an empty DF with index

    else:
        logger_instance.warning(
            "Columns 'category', 'level_0', '0' not found or no pivot candidates. Skipping pivot."
        )
        # If pivot columns aren't present, maybe aggregate other candidates?
        # For now, we just won't have pivoted features.
        pivot_features = pd.DataFrame(index=admission_base.index)

    # --- 5. Combine Aggregated/Pivoted Features ---
    logger_instance.info("Combining base, demographic, and pivoted features...")
    # Start with base features
    restructured_data = admission_base
    # Merge demographic features
    restructured_data = pd.merge(
        restructured_data, admission_demo, left_index=True, right_index=True, how="left"
    )
    # Merge pivoted features
    restructured_data = pd.merge(
        restructured_data, pivot_features, left_index=True, right_index=True, how="left"
    )

    restructured_data = (
        restructured_data.reset_index()
    )  # Make subject_id, hadm_id columns again

    logger_instance.info(
        f"Data restructured: {len(restructured_data)} unique admissions, {len(restructured_data.columns)} columns."
    )
    logger_instance.debug(
        f"Final columns after restructuring: {restructured_data.columns.tolist()}"
    )
    return restructured_data


class BaseModel(ABC):
    """
    Base class for models.
    """

    def __init__(
        self, model_type: str, config: Optional[Dict] = None, random_state: int = 42
    ):
        """
        Initialize the model.

        Args:
            model_type (str): Type of model ('readmission', 'mortality', or 'los')
            config (Optional[Dict], optional): Configuration dictionary.
                If None, loads the default configuration. Defaults to None.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        self.model_type = model_type
        self.config = config if config is not None else load_config()
        self.random_state = random_state
        self.logger = logger
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()  # Default scaler for non-temporal models

        # Set model-specific configuration
        # Handle potential KeyError if model_type isn't in config (e.g., during direct instantiation)
        if model_type in self.config["models"]:
            self.model_config = self.config["models"][model_type]
            self.target = self.model_config.get("target")  # Use .get for safety
            self.algorithms = self.model_config.get(
                "algorithms", []
            )  # Default to empty list
            self.cv_folds = self.model_config.get("cv_folds", 5)
            self.hyperparameter_tuning = self.model_config.get(
                "hyperparameter_tuning", False
            )
        else:
            self.logger.warning(
                f"Model type '{model_type}' not found in config['models']. Using defaults."
            )
            self.model_config = {}
            self.target = None
            self.algorithms = []
            self.cv_folds = 5
            self.hyperparameter_tuning = False

    @abstractmethod
    def preprocess(
        self, data: pd.DataFrame, for_prediction: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess the data for modelling.

        Args:
            data (pd.DataFrame): Input data
            for_prediction (bool, optional): Whether preprocessing is for prediction.
                If True, returns None for the target. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Preprocessed features and target
        """
        pass

    @abstractmethod
    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, algorithm: str = None
    ) -> BaseEstimator:
        """
        Train the model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            algorithm (str, optional): Algorithm to use. If None, uses the first
                algorithm in the configuration. Defaults to None.

        Returns:
            BaseEstimator: Trained model
        """
        pass

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        pass

    def fit(
        self, data: pd.DataFrame, algorithm: str = None, test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Fit the model to the data. Handles preprocessing, splitting, scaling, training, evaluation.
        Note: This default implementation assumes non-temporal data and scikit-learn style models.
              Temporal models should override this.
        """
        # Preprocess data
        X, y = self.preprocess(data)
        if y is None:
            raise ValueError(
                "Target variable 'y' is None after preprocessing during fit."
            )

        self.feature_names = (
            X.columns.tolist()
        )  # Store feature names after preprocessing

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if y.nunique() > 1 else None,  # Stratify only for classification
        )

        # Scale features
        self.logger.info("Scaling features...")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        self.logger.info("Features scaled.")

        # Train model
        self.model = self.train(X_train_scaled, y_train, algorithm)

        # Evaluate model
        metrics = self.evaluate(X_test_scaled, y_test)

        return metrics

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model. Handles preprocessing and scaling.
        Note: This default implementation assumes non-temporal data and scikit-learn style models.
              Temporal models should override this.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        if self.feature_names is None:
            raise ValueError("Model has not been fitted yet (feature names not set)")
        if self.scaler is None:  # Check the default scaler
            raise ValueError("Scaler has not been fitted yet")

        # Preprocess data - crucial to use the same steps as training
        X, _ = self.preprocess(data, for_prediction=True)

        # Ensure all feature columns used during training are present
        missing_cols = set(self.feature_names) - set(X.columns)
        for col in missing_cols:
            self.logger.warning(
                f"Missing feature column during prediction: {col}. Filling with 0."
            )
            X[col] = 0

        # Ensure columns are in the same order as during training
        # Also drop any extra columns not seen during training
        extra_cols = set(X.columns) - set(self.feature_names)
        if extra_cols:
            self.logger.warning(
                f"Extra columns found during prediction: {extra_cols}. Dropping them."
            )
            X = X.drop(columns=list(extra_cols))

        X = X[self.feature_names]

        # Scale features using the fitted scaler
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),  # Use transform, not fit_transform
            columns=X.columns,
            index=X.index,
        )

        # Make predictions
        return self.model.predict(X_scaled)

    def save(self, path: str) -> None:
        """
        Save the model to disk using pickle.
        Note: Temporal models should override this to save PyTorch state_dict.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model and metadata
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "target": self.target,
            "config": self.config,  # Save config used for training
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Load a model from disk. Handles both pickled scikit-learn models
        and saved PyTorch model state_dicts (for TemporalReadmissionModel).
        """
        # Try loading as PyTorch state_dict first (specifically for temporal)
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Load onto CPU first to avoid GPU memory issues if the saving GPU differs
            checkpoint = torch.load(path, map_location="cpu")

            # Check if it looks like our PyTorch model save format
            if (
                isinstance(checkpoint, dict)
                and "model_state_dict" in checkpoint
                and checkpoint.get("model_type") == "temporal_readmission"
            ):
                logger.info(f"Loading TemporalReadmissionModel state_dict from {path}")
                config = checkpoint["config"]
                # Instantiate the correct class using cls() which refers to the class calling load
                # This assumes TemporalReadmissionModel.load(path) is called
                if cls.__name__ != "TemporalReadmissionModel":
                    # If BaseModel.load was called directly, we need to find the right class
                    # This scenario is less likely if using TemporalReadmissionModel.load()
                    logger.warning(
                        f"BaseModel.load called for temporal model. Instantiating TemporalReadmissionModel directly."
                    )
                    model_instance = TemporalReadmissionModel(config=config)
                else:
                    model_instance = cls(config=config)

                # Recreate model architecture using saved parameters
                arch_params = checkpoint["model_architecture_params"]
                model_instance.model = TimeAwarePatientLSTM(**arch_params)
                model_instance.model.load_state_dict(checkpoint["model_state_dict"])
                model_instance.model.to(device)  # Move model to the correct device
                model_instance.device = device  # Store the device

                # Load metadata
                model_instance.feature_names = checkpoint.get("feature_names")
                model_instance.sequence_scaler = checkpoint.get("sequence_scaler")
                model_instance.static_scaler = checkpoint.get("static_scaler")
                model_instance.target = checkpoint.get("target")

                logger.info(
                    f"Loaded Temporal model '{model_instance.model_type}' from {path} onto {device}"
                )
                return model_instance
            else:
                # If it's not our specific PyTorch format, assume it might be a pickle file
                logger.debug(
                    "File loaded via torch.load but not a temporal model state_dict. Trying pickle."
                )
                pass  # Fall through to pickle loading

        except Exception as e_torch:
            logger.debug(
                f"Failed to load as PyTorch state_dict ({e_torch}). Trying pickle..."
            )
            pass  # Fall through to pickle loading

        # --- Fallback to Pickle Loading (for non-temporal models) ---
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            # Ensure it's a dictionary as expected
            if not isinstance(model_data, dict):
                raise TypeError(f"Pickle file at {path} did not contain a dictionary.")

            model_type = model_data.get("model_type")
            config = model_data.get("config")

            if model_type is None or config is None:
                raise ValueError(
                    f"Pickle file at {path} missing 'model_type' or 'config'."
                )

            # Instantiate the correct model class (excluding temporal as it's handled above)
            if model_type == "readmission":
                model_instance = ReadmissionModel(config=config)
            elif model_type == "mortality":
                model_instance = MortalityModel(config=config)
            elif model_type == "los":
                model_instance = LengthOfStayModel(config=config)
            elif model_type == "temporal_readmission":
                # This case should ideally be caught by torch.load, but handle defensively
                logger.warning(
                    f"Pickle file contains 'temporal_readmission' type. Loading via pickle is deprecated for this type."
                )
                model_instance = TemporalReadmissionModel(
                    config=config
                )  # Will likely fail later if model is PyTorch
            else:
                raise ValueError(
                    f"Unknown or unsupported model type in pickle file: {model_type}"
                )

            # Load model and metadata
            model_instance.model = model_data.get("model")
            model_instance.feature_names = model_data.get("feature_names")
            model_instance.scaler = model_data.get("scaler")
            model_instance.target = model_data.get("target")

            # Basic validation
            if model_instance.model is None:
                logger.warning(f"Model object not found in pickle file: {path}")
            if model_instance.feature_names is None:
                logger.warning(f"Feature names not found in pickle file: {path}")
            if model_instance.scaler is None:
                logger.warning(f"Scaler not found in pickle file: {path}")

            logger.info(f"Loaded pickled model '{model_type}' from {path}")
            return model_instance

        except pickle.UnpicklingError as e_pickle:
            logger.error(
                f"Failed to load model from {path}. Not a valid PyTorch state_dict or pickle file."
            )
            # Combine error messages if torch loading also failed
            e_combined = f"Pickle Error: {e_pickle}"
            if "e_torch" in locals():
                e_combined += f" | Torch Error: {e_torch}"
            raise IOError(f"Could not load model file: {path}. Errors: {e_combined}")
        except Exception as e_other:
            logger.error(
                f"An unexpected error occurred during model loading from {path}: {e_other}",
                exc_info=True,
            )
            raise

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the model.
        Note: This default implementation assumes scikit-learn style models.
              Temporal models should override this if applicable.
        Args:
            top_n (int, optional): Number of top features to return. Defaults to 20.

        Returns:
            pd.DataFrame: Feature importance
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        if self.feature_names is None:
            raise ValueError("Model has not been fitted yet (feature names not set)")

        # Get feature importance based on model type
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # Handle multi-class coefficients (shape [n_classes, n_features])
            if self.model.coef_.ndim > 1 and self.model.coef_.shape[0] > 1:
                # For binary classification, often use the coefficients for the positive class
                # Or take the absolute mean across classes
                importance = np.abs(
                    self.model.coef_[1]
                    if self.model.coef_.shape[0] == 2
                    else np.mean(np.abs(self.model.coef_), axis=0)
                )
            else:  # Binary classification or regression
                importance = (
                    np.abs(self.model.coef_[0])
                    if self.model.coef_.ndim > 1
                    else np.abs(self.model.coef_)
                )
        else:
            self.logger.warning(
                f"Model type {type(self.model)} does not have standard feature_importances_ or coef_ attribute. Cannot get feature importance."
            )
            return pd.DataFrame()

        # Ensure feature_names is a flat list (handle dict case for temporal if needed)
        f_names = (
            self.feature_names["static"]
            if isinstance(self.feature_names, dict)
            else self.feature_names
        )

        # Create dataframe
        if len(importance) != len(f_names):
            self.logger.error(
                f"Mismatch between number of importance values ({len(importance)}) and feature names ({len(f_names)})"
            )
            # Attempt to return partial results if lengths mismatch but importance exists
            min_len = min(len(importance), len(f_names))
            if min_len > 0:
                importance_df = pd.DataFrame(
                    {"feature": f_names[:min_len], "importance": importance[:min_len]}
                )
            else:
                return pd.DataFrame()

        else:
            importance_df = pd.DataFrame({"feature": f_names, "importance": importance})

        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)

        # Return top N features
        return importance_df.head(top_n)

    def get_shap_values(
        self, data: pd.DataFrame, num_examples: Optional[int] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Calculate SHAP values for the model.
        Note: This default implementation assumes scikit-learn style models and data.
              Temporal models should override this.
        Args:
            data (pd.DataFrame): Data to explain (should be preprocessed and scaled)
            num_examples (Optional[int], optional): Number of examples to use for SHAP calculation.
                If None, uses all data. Defaults to None.

        Returns:
            Tuple[np.ndarray, pd.DataFrame]: SHAP values and the data subset used
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        if self.feature_names is None:
            raise ValueError("Feature names not set. Model needs to be fitted.")

        # Ensure data is a DataFrame with the correct columns
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
        if not all(f in data.columns for f in self.feature_names):
            missing = set(self.feature_names) - set(data.columns)
            raise ValueError(f"Input data missing required feature columns: {missing}")
        data = data[self.feature_names]  # Ensure correct column order

        if num_examples is not None and num_examples < len(data):
            data_subset = shap.sample(
                data, num_examples, random_state=self.random_state
            )
        else:
            data_subset = data

        self.logger.info(f"Calculating SHAP values for {len(data_subset)} examples...")

        # Create explainer based on model type
        try:
            if isinstance(
                self.model,
                (
                    xgb.XGBClassifier,
                    xgb.XGBRegressor,
                    lgb.LGBMClassifier,
                    lgb.LGBMRegressor,
                    RandomForestClassifier,
                    RandomForestRegressor,
                ),
            ):
                explainer = shap.TreeExplainer(self.model)
            elif isinstance(self.model, (LogisticRegression, LinearRegression)):
                # Use KernelExplainer for linear models, requires a background dataset
                background_data = shap.sample(
                    data, min(100, len(data)), random_state=self.random_state
                )  # Sample background data
                explainer = shap.KernelExplainer(self.model.predict, background_data)
            else:
                # Fallback or raise error for unsupported models
                self.logger.warning(
                    f"SHAP explainer not automatically configured for model type {type(self.model)}. Using KernelExplainer as fallback."
                )
                background_data = shap.sample(
                    data, min(100, len(data)), random_state=self.random_state
                )
                # Need predict_proba for classifiers if available
                if hasattr(self.model, "predict_proba"):

                    def predict_fn(data_np):
                        # KernelExplainer expects numpy array
                        df = pd.DataFrame(data_np, columns=self.feature_names)
                        return self.model.predict_proba(df)[
                            :, 1
                        ]  # Prob of positive class

                    explainer = shap.KernelExplainer(predict_fn, background_data)
                else:
                    explainer = shap.KernelExplainer(
                        self.model.predict, background_data
                    )
        except Exception as e:
            self.logger.error(f"Error creating SHAP explainer: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to create SHAP explainer for model type {type(self.model)}"
            ) from e

        # Calculate SHAP values
        try:
            shap_values = explainer.shap_values(data_subset)
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {e}", exc_info=True)
            # Provide more context in the error
            raise RuntimeError(
                f"Failed to calculate SHAP values using {type(explainer).__name__}. Check model compatibility and data format."
            ) from e

        # Adjust shape for binary classification if needed (TreeExplainer might return list)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Use SHAP values for the positive class

        self.logger.info("SHAP values calculated.")
        return shap_values, data_subset


# --- Standard Model Implementations ---


class ReadmissionModel(BaseModel):
    """
    Model for predicting 30-day readmission.
    """

    def __init__(self, config: Optional[Dict] = None, random_state: int = 42):
        super().__init__(
            model_type="readmission", config=config, random_state=random_state
        )
        # Ensure evaluation metrics are suitable for classification
        self.evaluation_metrics = (
            self.config.get("evaluation", {})
            .get("classification", {})
            .get("metrics", [])
        )

    def preprocess(
        self, data: pd.DataFrame, for_prediction: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess data for readmission prediction.
        """
        self.logger.info(f"Preprocessing data for {self.model_type}...")
        self.logger.debug(f"Initial data shape: {data.shape}")
        self.logger.debug(f"Initial columns: {data.columns.tolist()}")

        # Check if data needs reshaping (multiple rows per admission)
        id_cols_to_check = ["subject_id", "hadm_id"]
        if (
            all(col in data.columns for col in id_cols_to_check)
            and data.duplicated(subset=id_cols_to_check).any()
        ):
            self.logger.info("Reshaping data to admission level.")
            id_cols = ["subject_id", "hadm_id", "stay_id"]
            target_cols = [
                self.target,
                "los_days",
                "hospital_death",
                "readmission_90day",
                "days_to_readmission",
            ]  # Include related targets
            # Filter target_cols to only those present in the data
            target_cols_present = [col for col in target_cols if col in data.columns]
            data = _reshape_data_to_admission_level(
                data, id_cols, target_cols_present, self.logger
            )
            self.logger.debug(f"Data shape after reshaping: {data.shape}")
            self.logger.debug(f"Columns after reshaping: {data.columns.tolist()}")

        # Extract target variable
        y = None
        if not for_prediction:
            if self.target in data.columns:
                y = data[self.target].copy()
                self.logger.info(f"Target variable '{self.target}' extracted.")
            else:
                raise ValueError(
                    f"Target column '{self.target}' not found in input data."
                )

        # Select features (all numeric columns excluding IDs and targets)
        exclude_cols = [
            "subject_id",
            "hadm_id",
            "stay_id",
            "admittime",
            "dischtime",
            "deathtime",
            "edregtime",
            "edouttime",
            "dod",  # Date of death from patient table
            self.target,
            "readmission_90day",  # Exclude other potential targets/leakage
            "days_to_readmission",
            "hospital_death",
            "los_days",
        ]
        # Include only numeric types for standard models
        X = data.select_dtypes(include=np.number).drop(
            columns=[col for col in exclude_cols if col in data.columns],
            errors="ignore",
        )

        # Handle potential infinite values
        X = X.replace([np.inf, -np.inf], np.nan)

        # Impute remaining NaNs (simple median imputation for baseline)
        # More sophisticated imputation could be done here or in feature engineering
        if X.isnull().any().any():
            self.logger.info("Imputing remaining NaN values with column medians...")
            for col in X.columns[X.isnull().any()]:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                self.logger.debug(f"Imputed NaNs in '{col}' with median {median_val}")

        self.logger.info(f"Preprocessing complete. Feature shape: {X.shape}")
        self.logger.debug(f"Final feature columns: {X.columns.tolist()}")

        return X, y

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, algorithm: str = None
    ) -> BaseEstimator:
        """
        Train a classification model for readmission.
        """
        if algorithm is None:
            # Ensure algorithms list is not empty
            if not self.algorithms:
                raise ValueError(
                    f"No algorithms specified in config for model type '{self.model_type}'"
                )
            algorithm = self.algorithms[0]  # Use the first algorithm if none specified
        self.logger.info(f"Training {algorithm} model for {self.model_type}...")

        # Define models
        model_map = {
            "logistic_regression": LogisticRegression(
                random_state=self.random_state,
                class_weight=self.model_config.get("class_weight", "balanced"),
                max_iter=1000,  # Increase max_iter for convergence
                solver="liblinear",  # Good default solver
            ),
            "random_forest": RandomForestClassifier(
                random_state=self.random_state,
                class_weight=self.model_config.get("class_weight", "balanced"),
                n_estimators=100,  # Default n_estimators
                n_jobs=-1,  # Use all available cores
            ),
            "xgboost": xgb.XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=(
                    (y_train == 0).sum() / (y_train == 1).sum()
                    if self.model_config.get("class_weight") == "balanced"
                    and (y_train == 1).sum() > 0
                    else 1
                ),
            ),
            "lightgbm": lgb.LGBMClassifier(
                random_state=self.random_state,
                class_weight=self.model_config.get("class_weight", "balanced"),
                n_jobs=-1,
            ),
        }

        if algorithm not in model_map:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. Choose from {list(model_map.keys())}"
            )

        model = model_map[algorithm]

        # Hyperparameter tuning (optional)
        if self.hyperparameter_tuning:
            self.logger.info(f"Performing hyperparameter tuning for {algorithm}...")
            # Define parameter grid (example for Logistic Regression)
            # TODO: Define proper grids for each algorithm in config or here
            param_grid = {}
            if algorithm == "logistic_regression":
                param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
            elif algorithm == "random_forest":
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                }
            # Add grids for xgboost, lightgbm

            if param_grid:
                cv = StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
                )
                grid_search = GridSearchCV(
                    model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                self.logger.info(f"Best parameters found: {grid_search.best_params_}")
            else:
                self.logger.warning(
                    f"No parameter grid defined for {algorithm}. Skipping tuning."
                )
                model.fit(X_train, y_train)  # Fit with default params
        else:
            model.fit(X_train, y_train)

        self.logger.info(f"{algorithm} model training complete.")
        return model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the classification model.
        """
        self.logger.info(f"Evaluating {self.model_type} model...")
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]  # Probability of positive class

        metrics = {}
        if "accuracy" in self.evaluation_metrics:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
        if "precision" in self.evaluation_metrics:
            metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
        if "recall" in self.evaluation_metrics:
            metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
        if "f1" in self.evaluation_metrics:
            metrics["f1"] = f1_score(y_test, y_pred, zero_division=0)
        if "roc_auc" in self.evaluation_metrics:
            if len(np.unique(y_test)) > 1:  # Check for multiple classes
                metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
            else:
                metrics["roc_auc"] = np.nan
                self.logger.warning(
                    "ROC AUC not defined for single-class results in test set."
                )
        if "pr_auc" in self.evaluation_metrics:
            metrics["pr_auc"] = average_precision_score(y_test, y_prob)

        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics


class MortalityModel(BaseModel):
    """
    Model for predicting ICU mortality.
    Inherits much from ReadmissionModel, potentially overriding preprocess if needed.
    """

    def __init__(self, config: Optional[Dict] = None, random_state: int = 42):
        super().__init__(
            model_type="mortality", config=config, random_state=random_state
        )
        # Ensure evaluation metrics are suitable for classification
        self.evaluation_metrics = (
            self.config.get("evaluation", {})
            .get("classification", {})
            .get("metrics", [])
        )

    def preprocess(
        self, data: pd.DataFrame, for_prediction: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess data for mortality prediction.
        Currently uses the same logic as ReadmissionModel preprocess.
        Override if mortality requires different feature selection or handling.
        """
        self.logger.info(
            f"Preprocessing data for {self.model_type} using ReadmissionModel logic..."
        )
        # For now, delegate to ReadmissionModel's preprocessing logic
        # Create a temporary ReadmissionModel instance to call its preprocess
        # This isn't ideal, consider refactoring preprocessing logic if it diverges significantly
        temp_readmission_model = ReadmissionModel(
            config=self.config, random_state=self.random_state
        )
        temp_readmission_model.target = self.target  # Ensure correct target is used
        return temp_readmission_model.preprocess(data, for_prediction)

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, algorithm: str = None
    ) -> BaseEstimator:
        """
        Train a classification model for mortality.
        Uses the same logic as ReadmissionModel train.
        """
        self.logger.info(
            f"Training model for {self.model_type} using ReadmissionModel logic..."
        )
        # Delegate to ReadmissionModel's training logic
        temp_readmission_model = ReadmissionModel(
            config=self.config, random_state=self.random_state
        )
        temp_readmission_model.target = self.target
        temp_readmission_model.evaluation_metrics = self.evaluation_metrics
        # We need to pass the already scaled data
        return temp_readmission_model.train(X_train, y_train, algorithm)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the classification model for mortality.
        Uses the same logic as ReadmissionModel evaluate.
        """
        self.logger.info(
            f"Evaluating {self.model_type} model using ReadmissionModel logic..."
        )
        if self.model is None:
            # If train was delegated, the model is stored in the instance calling evaluate
            raise ValueError("Model has not been trained yet.")

        # Delegate to ReadmissionModel's evaluation logic
        # Need to temporarily assign the trained model to a ReadmissionModel instance
        # This is getting complicated - refactoring evaluate might be better
        temp_readmission_model = ReadmissionModel(
            config=self.config, random_state=self.random_state
        )
        temp_readmission_model.model = self.model  # Assign the trained model
        temp_readmission_model.target = self.target
        temp_readmission_model.evaluation_metrics = self.evaluation_metrics
        return temp_readmission_model.evaluate(X_test, y_test)


class LengthOfStayModel(BaseModel):
    """
    Model for predicting Length of Stay (Regression).
    """

    def __init__(self, config: Optional[Dict] = None, random_state: int = 42):
        super().__init__(model_type="los", config=config, random_state=random_state)
        # Ensure evaluation metrics are suitable for regression
        self.evaluation_metrics = (
            self.config.get("evaluation", {}).get("regression", {}).get("metrics", [])
        )

    def preprocess(
        self, data: pd.DataFrame, for_prediction: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess data for Length of Stay prediction.
        Similar to ReadmissionModel but target is different.
        """
        self.logger.info(f"Preprocessing data for {self.model_type}...")
        self.logger.debug(f"Initial data shape: {data.shape}")

        # Reshaping logic (if needed) - same as ReadmissionModel
        id_cols_to_check = ["subject_id", "hadm_id"]
        if (
            all(col in data.columns for col in id_cols_to_check)
            and data.duplicated(subset=id_cols_to_check).any()
        ):
            self.logger.info("Reshaping data to admission level.")
            id_cols = ["subject_id", "hadm_id", "stay_id"]
            target_cols = [
                self.target,  # los_days
                "readmission_30day",
                "hospital_death",
                "readmission_90day",
                "days_to_readmission",
            ]
            target_cols_present = [col for col in target_cols if col in data.columns]
            data = _reshape_data_to_admission_level(
                data, id_cols, target_cols_present, self.logger
            )
            self.logger.debug(f"Data shape after reshaping: {data.shape}")

        # Extract target variable (Length of Stay)
        y = None
        if not for_prediction:
            if self.target in data.columns:
                y = data[self.target].copy()
                # Optional: Apply transformation like log(LOS + 1) if distribution is skewed
                # y = np.log1p(y)
                self.logger.info(f"Target variable '{self.target}' extracted.")
                # Check for NaN/inf in target
                if y.isnull().any() or np.isinf(y).any():
                    self.logger.warning(
                        f"Target variable '{self.target}' contains NaN or infinite values. Check data processing."
                    )
                    # Optionally drop rows with invalid target
                    # data = data.dropna(subset=[self.target])
                    # y = data[self.target].copy()

            else:
                raise ValueError(
                    f"Target column '{self.target}' not found in input data."
                )

        # Select features (same logic as ReadmissionModel)
        exclude_cols = [
            "subject_id",
            "hadm_id",
            "stay_id",
            "admittime",
            "dischtime",
            "deathtime",
            "edregtime",
            "edouttime",
            "dod",
            self.target,  # los_days
            "readmission_30day",
            "readmission_90day",
            "days_to_readmission",
            "hospital_death",
        ]
        X = data.select_dtypes(include=np.number).drop(
            columns=[col for col in exclude_cols if col in data.columns],
            errors="ignore",
        )
        X = X.replace([np.inf, -np.inf], np.nan)

        # Impute remaining NaNs
        if X.isnull().any().any():
            self.logger.info("Imputing remaining NaN values with column medians...")
            for col in X.columns[X.isnull().any()]:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                self.logger.debug(f"Imputed NaNs in '{col}' with median {median_val}")

        self.logger.info(f"Preprocessing complete. Feature shape: {X.shape}")
        return X, y

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, algorithm: str = None
    ) -> BaseEstimator:
        """
        Train a regression model for Length of Stay.
        """
        if algorithm is None:
            if not self.algorithms:
                raise ValueError(
                    f"No algorithms specified in config for model type '{self.model_type}'"
                )
            algorithm = self.algorithms[0]
        self.logger.info(f"Training {algorithm} model for {self.model_type}...")

        # Define models
        model_map = {
            "linear_regression": LinearRegression(n_jobs=-1),
            "random_forest": RandomForestRegressor(
                random_state=self.random_state, n_jobs=-1
            ),
            "xgboost": xgb.XGBRegressor(
                random_state=self.random_state, objective="reg:squarederror", n_jobs=-1
            ),
            "lightgbm": lgb.LGBMRegressor(random_state=self.random_state, n_jobs=-1),
        }

        if algorithm not in model_map:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. Choose from {list(model_map.keys())}"
            )

        model = model_map[algorithm]

        # Hyperparameter tuning (optional)
        if self.hyperparameter_tuning:
            self.logger.info(f"Performing hyperparameter tuning for {algorithm}...")
            # Define parameter grid (example for Random Forest Regressor)
            # TODO: Define proper grids for each algorithm
            param_grid = {}
            if algorithm == "random_forest":
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                }
            # Add grids for xgboost, lightgbm, linear regression (e.g., regularization)

            if param_grid:
                # Use standard KFold for regression
                cv = self.cv_folds  # Can directly pass integer to GridSearchCV
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                self.logger.info(f"Best parameters found: {grid_search.best_params_}")
            else:
                self.logger.warning(
                    f"No parameter grid defined for {algorithm}. Skipping tuning."
                )
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        self.logger.info(f"{algorithm} model training complete.")
        return model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the regression model.
        """
        self.logger.info(f"Evaluating {self.model_type} model...")
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        y_pred = self.model.predict(X_test)

        # Optional: Inverse transform predictions if target was transformed (e.g., log)
        # y_pred = np.expm1(y_pred)
        # y_test = np.expm1(y_test) # Apply to test set as well for comparison

        metrics = {}
        if "rmse" in self.evaluation_metrics:
            metrics["rmse"] = mean_squared_error(y_test, y_pred, squared=False)
        if "mae" in self.evaluation_metrics:
            metrics["mae"] = mean_absolute_error(y_test, y_pred)
        if "r2" in self.evaluation_metrics:
            metrics["r2"] = r2_score(y_test, y_pred)
        if "explained_variance" in self.evaluation_metrics:
            metrics["explained_variance"] = explained_variance_score(y_test, y_pred)

        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics


# --- Temporal Model Implementation ---


class TemporalReadmissionModel(BaseModel):
    """
    Model for readmission prediction using temporal (LSTM) approaches.
    """

    def __init__(self, config: Optional[Dict] = None, random_state: int = 42):
        """Initialize the temporal readmission model."""
        # Call super().__init__ AFTER defining temporal-specific attributes if needed by super()
        # For now, call super first
        super().__init__("temporal_readmission", config, random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(
            f"TemporalReadmissionModel initialized on device: {self.device}"
        )
        # Scalers for temporal data need careful handling (fit on training sequences/static)
        self.sequence_scaler = None  # Placeholder for potential sequence-specific scaler (e.g., fitted per feature)
        self.static_scaler = StandardScaler()  # Scaler for static features

    def _pad_sequence(self, seq: List, max_len: int, value: float = 0.0) -> List:
        """Pads or truncates a sequence to a maximum length."""
        if len(seq) >= max_len:
            return seq[:max_len]  # Truncate
        else:
            return seq + [value] * (max_len - len(seq))  # Pad

    def _calculate_intervals(self, timestamps: List) -> List:
        """
        Calculates time intervals between consecutive timestamps.
        Ensures the output length matches the timestamp list length (after padding).
        """
        if not timestamps:
            return []
        # Calculate intervals for existing timestamps
        intervals = [0.0] + [
            timestamps[i] - timestamps[i - 1]
            for i in range(1, len(timestamps))
            if timestamps[i] > 0 and timestamps[i - 1] > 0
        ]  # Avoid calculating diff on padded zeros
        # Pad intervals to match the full length of the (potentially padded) timestamps list
        padded_intervals = intervals + [0.0] * (len(timestamps) - len(intervals))
        return padded_intervals

    def preprocess(
        self, data: pd.DataFrame, for_prediction: bool = False
    ) -> Tuple[Dict, Optional[pd.Series]]:  # Return type changed for sequences
        """
        Preprocess data for the temporal model.
        Separates static and sequence features, pads/truncates sequences,
        calculates time intervals, scales static features.
        """
        self.logger.info(
            f"Preprocessing data for TemporalReadmissionModel (for_prediction={for_prediction})..."
        )

        # Configuration
        temporal_config = self.config["models"]["temporal_readmission"]
        max_len = temporal_config["sequence_length"]
        padding_value = temporal_config["padding_value"]
        id_col = (
            "hadm_id"  # Assuming this is the primary ID linking sequences/static data
        )

        # --- 1. Extract Target ---
        target_col = self.target
        y = (
            data[target_col].copy()
            if target_col in data.columns and not for_prediction
            else None
        )
        if y is not None:
            self.logger.info(f"Target variable '{target_col}' extracted.")
            # Ensure target is numeric
            y = pd.to_numeric(y, errors="coerce")
            if y.isnull().any():
                self.logger.warning(
                    f"Target column '{target_col}' contains non-numeric values. Check data."
                )
                # Decide handling: raise error or drop rows? For now, log warning.
        elif not for_prediction:
            raise ValueError(
                f"Target column '{target_col}' not found in input data during training."
            )

        # --- 2. Identify Columns ---
        all_cols = data.columns.tolist()
        sequence_cols = [col for col in all_cols if col.endswith("_sequence_data")]
        # Identify static columns: exclude IDs, target, and sequence columns
        # Ensure static cols identified during training are used for prediction
        if not for_prediction or self.feature_names is None:
            static_cols_potential = [
                col
                for col in all_cols
                if col
                not in [
                    self.target,
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                ]  # Common ID/target cols
                and not col.endswith("_sequence_data")
            ]
            # Keep only numeric static features for now (handle categorical later if needed)
            static_cols = (
                data[static_cols_potential]
                .select_dtypes(include=np.number)
                .columns.tolist()
            )
            if not for_prediction:
                self.feature_names = {"static": static_cols, "sequence": sequence_cols}
                self.logger.info(
                    f"Identified {len(static_cols)} static and {len(sequence_cols)} sequence features during training."
                )
        else:
            # Use stored feature names during prediction
            static_cols = self.feature_names["static"]
            sequence_cols = self.feature_names["sequence"]
            self.logger.info(
                f"Using {len(static_cols)} static and {len(sequence_cols)} sequence features from training."
            )

        # --- 3. Process Static Features ---
        static_features_df = data[[id_col] + static_cols].copy()
        # Handle potential missing/extra static columns during prediction
        if for_prediction:
            # Use stored static feature names
            stored_static_cols = self.feature_names["static"]
            missing_static = set(stored_static_cols) - set(static_features_df.columns)
            for col in missing_static:
                self.logger.warning(
                    f"Static feature '{col}' missing during prediction. Filling with 0."
                )
                static_features_df[col] = 0
            extra_static = (
                set(static_features_df.columns) - set(stored_static_cols) - {id_col}
            )
            if extra_static:
                self.logger.warning(
                    f"Dropping extra static features found during prediction: {extra_static}"
                )
                static_features_df = static_features_df.drop(columns=list(extra_static))
            static_features_df = static_features_df[
                [id_col] + stored_static_cols
            ]  # Ensure order

        # Scale static features
        static_values = static_features_df.drop(columns=[id_col])
        # Impute NaNs before scaling (using median from training set if predicting)
        if not for_prediction:
            self.logger.info("Fitting static feature scaler and storing medians...")
            self.static_scaler = StandardScaler()
            self._static_medians = (
                static_values.median()
            )  # Store medians for prediction imputation
            static_values_imputed = static_values.fillna(self._static_medians)
            scaled_static_values = self.static_scaler.fit_transform(
                static_values_imputed
            )
            self.logger.info("Static feature scaler fitted.")
        else:
            if self.static_scaler is None or not hasattr(self, "_static_medians"):
                raise ValueError(
                    "Static scaler or medians have not been fitted. Train the model first."
                )
            self.logger.info(
                "Transforming static features using fitted scaler and stored medians..."
            )
            static_values_imputed = static_values.fillna(
                self._static_medians
            )  # Impute with training medians
            scaled_static_values = self.static_scaler.transform(static_values_imputed)

        scaled_static_df = pd.DataFrame(
            scaled_static_values,
            index=static_features_df.index,
            columns=static_values.columns,
        )
        scaled_static_df[id_col] = static_features_df[id_col]
        # Convert scaled static features into a dictionary keyed by hadm_id
        static_features_dict = {
            row[id_col]: row.drop(id_col).values.astype(np.float32)
            for _, row in scaled_static_df.iterrows()
        }

        # --- 4. Process Sequences ---
        self.logger.info(
            f"Processing sequences (padding/truncating to {max_len}, calculating intervals)..."
        )
        num_sequence_features = len(sequence_cols)
        processed_sequences = {}
        processed_intervals = {}
        hadm_ids = data[id_col].tolist()

        # TODO: Implement sequence scaling (fit scaler during training, transform during prediction)
        # This is complex as scaling needs to happen per feature across all sequences.
        # For now, sequences remain unscaled.

        for _, row in tqdm(
            data.iterrows(), total=len(data), desc="Processing Sequences"
        ):
            hadm_id = row[id_col]
            # List to hold padded sequences for each feature for this admission
            admission_padded_seq_values = []
            # List to hold padded intervals for each feature for this admission
            admission_padded_intervals = []

            for seq_col_name in sequence_cols:
                # Default to empty lists if column is missing or value is NaN/None
                sequence_tuple = row.get(seq_col_name, ([], []))
                if not isinstance(sequence_tuple, tuple) or len(sequence_tuple) != 2:
                    self.logger.warning(
                        f"Invalid sequence data format for {hadm_id}, column {seq_col_name}. Using empty sequence."
                    )
                    timestamps, values = [], []
                else:
                    timestamps, values = sequence_tuple

                # Pad/Truncate values and timestamps
                padded_values = self._pad_sequence(values, max_len, padding_value)
                padded_timestamps = self._pad_sequence(
                    timestamps, max_len, 0.0
                )  # Pad time with 0

                # Calculate Intervals based on padded timestamps
                intervals = self._calculate_intervals(padded_timestamps)

                admission_padded_seq_values.append(padded_values)
                admission_padded_intervals.append(intervals)

            # Combine features for this admission
            # Resulting shape: (max_len, num_sequence_features) for values
            # Resulting shape: (max_len, num_sequence_features) for intervals (or maybe just 1 interval dim?)
            # Let's keep intervals separate for now, assuming TimeEncoder handles [batch, seq, 1]
            # Need to transpose the list of lists before converting to numpy array
            final_seq_array = np.array(
                admission_padded_seq_values, dtype=np.float32
            ).T  # Shape: (max_len, num_seq_feats)
            # For intervals, let's average or take the first? Or does the model expect intervals per feature?
            # Assuming model expects one interval sequence: take intervals from the first feature? Or average?
            # Let's take the first one for now, assuming time alignment across features.
            final_interval_array = np.array(
                admission_padded_intervals[0], dtype=np.float32
            ).reshape(
                -1, 1
            )  # Shape: (max_len, 1)

            processed_sequences[hadm_id] = final_seq_array
            processed_intervals[hadm_id] = final_interval_array

        # --- 5. Structure Output ---
        processed_data = {
            "sequences": processed_sequences,  # Dict[hadm_id, np.array[max_len, num_seq_feats]]
            "time_intervals": processed_intervals,  # Dict[hadm_id, np.array[max_len, 1]]
            "static_features": static_features_dict,  # Dict[hadm_id, np.array[num_static_feats]]
            "hadm_ids": hadm_ids,
            "labels": y.to_dict() if y is not None else None,  # Pass labels as dict too
        }

        self.logger.info("Preprocessing complete.")
        return processed_data, y  # Return dict and Series/None

    def train(
        self,
        train_data: Dict,
        y_train: pd.Series = None,
        algorithm: str = None,  # y_train ignored, algorithm ignored for PyTorch
    ) -> nn.Module:  # Return type changed to PyTorch module
        """
        Train the Time-Aware LSTM model.
        """
        self.logger.info(f"Training TemporalReadmissionModel on device: {self.device}")

        # --- Get Data & Config ---
        # Data comes preprocessed from self.preprocess
        sequences = train_data["sequences"]
        time_intervals = train_data["time_intervals"]
        static_features = train_data["static_features"]
        hadm_ids = train_data["hadm_ids"]
        labels = train_data["labels"]  # Labels are now in the dict

        if labels is None:
            raise ValueError("Labels are missing from the preprocessed training data.")

        # Config
        temporal_config = self.config["models"]["temporal_readmission"]
        num_epochs = temporal_config.get("num_epochs", 10)
        batch_size = temporal_config.get("batch_size", 32)
        learning_rate = temporal_config.get("learning_rate", 0.001)
        optimizer_name = temporal_config.get("optimizer", "Adam").lower()
        loss_fn_name = temporal_config.get("loss_function", "BCEWithLogitsLoss")
        class_weight_setting = temporal_config.get("class_weight", None)

        # --- Determine Model Dimensions ---
        # Get dimensions from the first example (assuming consistency)
        if not hadm_ids:
            raise ValueError(
                "No data found in train_data['hadm_ids']. Cannot determine model dimensions."
            )
        first_id = hadm_ids[0]
        if first_id not in sequences or first_id not in static_features:
            # Try finding the first valid ID if the absolute first one is missing data
            valid_id_found = False
            for hid in hadm_ids:
                if hid in sequences and hid in static_features:
                    first_id = hid
                    valid_id_found = True
                    break
            if not valid_id_found:
                raise ValueError(
                    f"Could not find any hadm_id with both sequence and static feature data."
                )
            self.logger.warning(
                f"First hadm_id '{hadm_ids[0]}' missing data, using '{first_id}' instead to determine dimensions."
            )

        seq_example = sequences[first_id]  # Shape: (max_len, num_seq_feats)
        static_example = static_features[first_id]  # Shape: (num_static_feats,)

        # Ensure examples are numpy arrays before checking shape
        if not isinstance(seq_example, np.ndarray) or not isinstance(
            static_example, np.ndarray
        ):
            raise TypeError(
                f"Expected numpy arrays for sequence and static features, but got {type(seq_example)} and {type(static_example)} for id {first_id}"
            )

        if seq_example.ndim != 2:
            raise ValueError(
                f"Expected sequence features to have 2 dimensions (max_len, num_seq_feats), but got {seq_example.ndim} for id {first_id}"
            )
        if static_example.ndim != 1:
            raise ValueError(
                f"Expected static features to have 1 dimension (num_static_feats,), but got {static_example.ndim} for id {first_id}"
            )

        input_dim = seq_example.shape[1]  # num_seq_feats
        static_dim = static_example.shape[0]  # num_static_feats
        self.logger.info(
            f"Determined dimensions: input_dim={input_dim}, static_dim={static_dim}"
        )

        # Model Hyperparameters from config
        hidden_dim = temporal_config.get("hidden_dim", 64)
        time_embed_dim = temporal_config.get("time_embed_dim", 16)
        num_layers = temporal_config.get("num_layers", 2)
        dropout = temporal_config.get("dropout", 0.2)

        # --- Create Dataset and DataLoader ---
        # Ensure labels are in a format compatible with the dataset (e.g., dict)
        if isinstance(labels, pd.Series):
            labels_dict = labels.to_dict()
        elif isinstance(labels, dict):
            labels_dict = labels
        else:
            raise TypeError(
                f"Unsupported type for labels: {type(labels)}. Expected dict or pd.Series."
            )

        train_dataset = TemporalEHRDataset(
            sequences=sequences,
            time_intervals=time_intervals,
            static_features=static_features,
            labels=labels_dict,  # Pass labels dict
            hadm_ids=hadm_ids,
            # Pass necessary params like max_len if Dataset handles padding internally
            # Currently assumes padding is done in preprocess
        )
        # Use the custom collate function defined in the class
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        # --- Initialize Model, Loss, Optimizer ---
        self.model = TimeAwarePatientLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            static_dim=static_dim,
            time_embed_dim=time_embed_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # Loss Function
        pos_weight = None
        if class_weight_setting == "balanced":
            # Calculate positive weight based on labels in the training set
            label_values = np.array(list(labels_dict.values()))
            neg_count = (label_values == 0).sum()
            pos_count = (label_values == 1).sum()
            if pos_count > 0 and neg_count > 0:  # Avoid division by zero
                pos_weight = torch.tensor([neg_count / pos_count], device=self.device)
                self.logger.info(
                    f"Using balanced class weights. Pos_weight: {pos_weight.item():.2f}"
                )
            else:
                self.logger.warning(
                    "Cannot calculate balanced class weight: pos_count or neg_count is zero."
                )

        if loss_fn_name == "BCEWithLogitsLoss":
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            # Add other loss functions if needed
            raise ValueError(f"Unsupported loss function: {loss_fn_name}")

        # Optimizer
        if optimizer_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == "rmsprop":
            optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        # Add other optimizers like AdamW if needed
        # elif optimizer_name == "adamw":
        #      optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.logger.info(f"Using optimizer: {optimizer_name}, loss: {loss_fn_name}")

        # --- Training Loop ---
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        train_losses = []
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            # Wrap train_loader with tqdm for progress bar
            batch_iterator = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
            )
            for batch in batch_iterator:
                # Unpack batch from collate_fn
                seq_batch, interval_batch, static_batch, label_batch = batch

                # Move to device
                seq_batch = seq_batch.to(self.device)
                interval_batch = interval_batch.to(self.device)
                static_batch = static_batch.to(self.device)
                label_batch = (
                    label_batch.to(self.device).float().view(-1, 1)
                )  # Ensure float and correct shape [batch_size, 1]

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(
                    seq_batch, interval_batch, static_batch
                )  # Model returns logits
                loss = criterion(outputs, label_batch)

                # Backward pass and optimize
                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                # Update tqdm progress bar description with current loss
                batch_iterator.set_postfix(
                    {"loss": f"{loss.item():.4f}"}
                )  # Format loss in progress bar

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            train_losses.append(avg_epoch_loss)
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_epoch_loss:.4f}"
            )

        self.logger.info("Training complete.")
        # Feature names and scalers are already stored during preprocess
        return self.model

    def _collate_fn(self, batch):
        """
        Custom collate function to handle variable length sequences before padding
        and combine dictionary-based data into batches.
        Assumes preprocess has already padded/processed individual sequences.
        """
        # Batch is a list of tuples: [(seq_tensor, interval_tensor, static_tensor, label_tensor), ...]
        # where tensors are for individual patients from TemporalEHRDataset.__getitem__

        # Separate the components
        sequences = [item[0] for item in batch]
        intervals = [item[1] for item in batch]
        statics = [item[2] for item in batch]
        labels = [item[3] for item in batch]

        # Stack tensors along a new batch dimension
        # Assumes sequences and intervals from __getitem__ are already correctly shaped [max_len, num_features] etc.
        # Need to ensure they are tensors before stacking
        sequences_batch = torch.stack(
            [torch.as_tensor(s, dtype=torch.float32) for s in sequences], dim=0
        )
        intervals_batch = torch.stack(
            [torch.as_tensor(i, dtype=torch.float32) for i in intervals], dim=0
        )
        statics_batch = torch.stack(
            [torch.as_tensor(st, dtype=torch.float32) for st in statics], dim=0
        )
        labels_batch = torch.stack(
            [torch.as_tensor(l, dtype=torch.float32) for l in labels], dim=0
        )

        return sequences_batch, intervals_batch, statics_batch, labels_batch

    def evaluate(
        self, test_data: Dict, y_test: pd.Series = None
    ) -> Dict[
        str, float
    ]:  # Input type changed, y_test ignored (use labels from test_data)
        """
        Evaluate the temporal model on the provided test data.
        """
        self.logger.info("Evaluating TemporalReadmissionModel...")
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        self.model.eval()  # Set model to evaluation mode

        # --- Get Data ---
        sequences = test_data.get("sequences")
        time_intervals = test_data.get("time_intervals")
        static_features = test_data.get("static_features")
        hadm_ids = test_data.get("hadm_ids")
        labels = test_data.get("labels")  # Labels from preprocessed data

        if not all([sequences, time_intervals, static_features, hadm_ids, labels]):
            missing_keys = [k for k, v in test_data.items() if v is None]
            raise ValueError(
                f"Test data dictionary is missing required keys: {missing_keys}"
            )

        # --- Create Dataset and DataLoader ---
        # Ensure labels are in a format compatible with the dataset (e.g., dict)
        if isinstance(labels, pd.Series):
            labels_dict = labels.to_dict()
        elif isinstance(labels, dict):
            labels_dict = labels
        else:
            raise TypeError(
                f"Unsupported type for labels: {type(labels)}. Expected dict or pd.Series."
            )

        # Ensure all hadm_ids in labels_dict are present in other data dicts
        valid_hadm_ids = [
            hid
            for hid in hadm_ids
            if hid in sequences
            and hid in time_intervals
            and hid in static_features
            and hid in labels_dict
        ]
        if len(valid_hadm_ids) != len(hadm_ids):
            self.logger.warning(
                f"Found {len(hadm_ids) - len(valid_hadm_ids)} hadm_ids with incomplete data. Evaluating only on complete cases."
            )
            # Filter all data structures to only include valid IDs
            sequences = {hid: sequences[hid] for hid in valid_hadm_ids}
            time_intervals = {hid: time_intervals[hid] for hid in valid_hadm_ids}
            static_features = {hid: static_features[hid] for hid in valid_hadm_ids}
            labels_dict = {hid: labels_dict[hid] for hid in valid_hadm_ids}
            hadm_ids = valid_hadm_ids  # Update hadm_ids list

        if not hadm_ids:
            self.logger.error("No valid HADM IDs found for evaluation after filtering.")
            return {
                metric: np.nan
                for metric in self.config.get("evaluation", {})
                .get("classification", {})
                .get("metrics", [])
            }

        test_dataset = TemporalEHRDataset(
            sequences=sequences,
            time_intervals=time_intervals,
            static_features=static_features,
            labels=labels_dict,  # Pass filtered labels dict
            hadm_ids=hadm_ids,  # Pass filtered IDs
        )
        # Use a reasonable batch size for evaluation
        eval_batch_size = (
            self.config["models"]["temporal_readmission"].get("batch_size", 32) * 2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        all_preds_probs = []
        all_true_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                seq_batch, interval_batch, static_batch, label_batch = batch

                # Move to device
                seq_batch = seq_batch.to(self.device)
                interval_batch = interval_batch.to(self.device)
                static_batch = static_batch.to(self.device)
                # label_batch is already a tensor from collate_fn

                # Forward pass
                logits = self.model(seq_batch, interval_batch, static_batch)
                probs = torch.sigmoid(logits).cpu().numpy()  # Get probabilities

                all_preds_probs.extend(probs.flatten())
                all_true_labels.extend(
                    label_batch.cpu().numpy().flatten()
                )  # Collect true labels from the batch

        # Calculate metrics
        all_true_labels = np.array(all_true_labels)
        all_preds_probs = np.array(all_preds_probs)

        if len(all_true_labels) == 0:
            self.logger.warning("No predictions generated during evaluation.")
            return {
                metric: np.nan
                for metric in self.config.get("evaluation", {})
                .get("classification", {})
                .get("metrics", [])
            }

        # Use a threshold (e.g., 0.5) for classification metrics
        threshold = 0.5
        all_preds_binary = (all_preds_probs >= threshold).astype(int)

        metrics = {}
        # Use evaluation metrics specified in config if available
        eval_metrics_config = (
            self.config.get("evaluation", {})
            .get("classification", {})
            .get("metrics", [])
        )
        if not eval_metrics_config:  # Default if not specified
            eval_metrics_config = [
                "roc_auc",
                "pr_auc",
                "f1",
                "precision",
                "recall",
                "accuracy",
            ]

        try:
            # Ensure there are labels to evaluate against
            if len(all_true_labels) > 0:
                if "accuracy" in eval_metrics_config:
                    metrics["accuracy"] = accuracy_score(
                        all_true_labels, all_preds_binary
                    )
                if "precision" in eval_metrics_config:
                    metrics["precision"] = precision_score(
                        all_true_labels, all_preds_binary, zero_division=0
                    )
                if "recall" in eval_metrics_config:
                    metrics["recall"] = recall_score(
                        all_true_labels, all_preds_binary, zero_division=0
                    )
                if "f1" in eval_metrics_config:
                    metrics["f1"] = f1_score(
                        all_true_labels, all_preds_binary, zero_division=0
                    )

                # Check if both classes are present for ROC AUC and PR AUC
                if len(np.unique(all_true_labels)) > 1:
                    if "roc_auc" in eval_metrics_config:
                        metrics["roc_auc"] = roc_auc_score(
                            all_true_labels, all_preds_probs
                        )
                    if "pr_auc" in eval_metrics_config:
                        metrics["pr_auc"] = average_precision_score(
                            all_true_labels, all_preds_probs
                        )
                else:
                    self.logger.warning(
                        "ROC AUC / PR AUC not defined for single-class results."
                    )
                    if "roc_auc" in eval_metrics_config:
                        metrics["roc_auc"] = np.nan
                    if "pr_auc" in eval_metrics_config:
                        metrics["pr_auc"] = np.nan
            else:
                self.logger.warning("No true labels available for metric calculation.")
                for m in eval_metrics_config:
                    metrics[m] = np.nan

        except ValueError as e:
            self.logger.error(
                f"Error calculating evaluation metrics: {e}", exc_info=True
            )
            # Set metrics to NaN or 0 if calculation fails
            for m in eval_metrics_config:
                metrics[m] = np.nan

        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def predict(
        self, data: pd.DataFrame
    ) -> np.ndarray:  # Input likely needs to be richer than just DataFrame
        """
        Make predictions with the temporal model.
        Input data format needs careful consideration (might need sequences directly).
        """
        self.logger.info("Predicting with TemporalReadmissionModel...")
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if self.static_scaler is None or not hasattr(self, "_static_medians"):
            raise ValueError(
                "Static scaler or medians have not been fitted. Train the model first."
            )
        if self.feature_names is None:
            raise ValueError("Feature names not set. Train the model first.")

        self.model.eval()  # Set model to evaluation mode

        # --- Preprocess Input Data ---
        # Use the same preprocess method used for training/evaluation
        processed_data, _ = self.preprocess(data, for_prediction=True)

        sequences = processed_data["sequences"]
        time_intervals = processed_data["time_intervals"]
        static_features = processed_data["static_features"]
        hadm_ids = processed_data["hadm_ids"]

        # Create a dummy labels dictionary for the Dataset/DataLoader
        dummy_labels = {
            hid: 0 for hid in hadm_ids
        }  # Value doesn't matter for prediction

        # --- Create Dataset and DataLoader ---
        pred_dataset = TemporalEHRDataset(
            sequences=sequences,
            time_intervals=time_intervals,
            static_features=static_features,
            labels=dummy_labels,  # Pass dummy labels
            hadm_ids=hadm_ids,
        )
        pred_batch_size = (
            self.config["models"]["temporal_readmission"].get("batch_size", 32) * 2
        )
        pred_loader = DataLoader(
            pred_dataset,
            batch_size=pred_batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        all_preds_probs = []
        with torch.no_grad():
            for batch in tqdm(pred_loader, desc="Predicting", leave=False):
                seq_batch, interval_batch, static_batch, _ = (
                    batch  # Ignore dummy labels
                )

                # Move to device
                seq_batch = seq_batch.to(self.device)
                interval_batch = interval_batch.to(self.device)
                static_batch = static_batch.to(self.device)

                # Forward pass
                logits = self.model(seq_batch, interval_batch, static_batch)
                probs = torch.sigmoid(logits).cpu().numpy()  # Get probabilities

                all_preds_probs.extend(probs.flatten())

        return np.array(all_preds_probs)

    def save(self, path: str) -> None:
        """
        Save the PyTorch model state_dict and associated metadata.
        Overrides BaseModel.save for PyTorch specifics.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        if self.static_scaler is None or not hasattr(self, "_static_medians"):
            raise ValueError("Static scaler or medians have not been fitted.")
        if self.feature_names is None:
            raise ValueError("Feature names not set.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Ensure model is on CPU before saving state_dict for better portability
        self.model.to("cpu")

        # Gather model architecture parameters dynamically
        try:
            lstm_layer = getattr(self.model, "lstm")
            time_encoder_layer = getattr(self.model, "time_encoder")
            classifier_layer = getattr(self.model, "classifier")

            # Infer dimensions carefully
            lstm_input_size = lstm_layer.input_size
            time_embed_dim = time_encoder_layer.embed_dim
            input_dim = (
                lstm_input_size - time_embed_dim
            )  # Infer original sequence feature dim

            hidden_dim = lstm_layer.hidden_size
            num_layers = lstm_layer.num_layers
            lstm_dropout = lstm_layer.dropout  # This is the dropout between LSTM layers

            # Infer static_dim from the first linear layer of the classifier
            # Assumes classifier structure: Linear -> ReLU -> Dropout -> Linear
            classifier_input_features = classifier_layer[0].in_features
            static_dim = (
                classifier_input_features - hidden_dim
            )  # Infer static feature dim

            output_dim = classifier_layer[
                -1
            ].out_features  # Get output dim from last layer

            # Get overall dropout from the Dropout layer in the classifier
            dropout_layer = [m for m in classifier_layer if isinstance(m, nn.Dropout)]
            classifier_dropout = (
                dropout_layer[0].p if dropout_layer else 0.2
            )  # Default if not found

            model_arch_params = {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "static_dim": static_dim,
                "time_embed_dim": time_embed_dim,
                "num_layers": num_layers,
                # Use classifier dropout as the main 'dropout' param? Or LSTM dropout? Clarify.
                # Let's assume the config 'dropout' refers to the classifier dropout for now.
                "dropout": classifier_dropout,
                "output_dim": output_dim,
            }
        except AttributeError as e:
            self.logger.error(
                f"Could not dynamically determine model architecture parameters: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                "Failed to gather model architecture parameters for saving."
            ) from e

        # Save model state_dict and metadata
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "feature_names": self.feature_names,  # Store sequence/static feature names
            "static_scaler": self.static_scaler,  # Save static scaler state
            "_static_medians": getattr(
                self, "_static_medians", None
            ),  # Save medians used for imputation
            # Add sequence scaler saving if implemented
            "model_type": self.model_type,
            "target": self.target,
            "config": self.config,  # Save config used for training (includes hyperparameters)
            "model_architecture_params": model_arch_params,
        }

        # Use torch.save for PyTorch components
        try:
            torch.save(model_data, path)
            self.logger.info(f"Saved Temporal model state and metadata to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save model to {path}: {e}", exc_info=True)
            raise
        finally:
            # Move model back to original device if needed
            self.model.to(self.device)

    # Override load method in TemporalReadmissionModel specifically
    @classmethod
    def load(cls, path: str) -> "TemporalReadmissionModel":
        """
        Load a TemporalReadmissionModel from disk (expects PyTorch saved format).
        """
        logger.info(f"Attempting to load TemporalReadmissionModel from: {path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Load onto CPU first
            checkpoint = torch.load(path, map_location="cpu")

            if (
                not isinstance(checkpoint, dict)
                or "model_state_dict" not in checkpoint
                or checkpoint.get("model_type") != "temporal_readmission"
            ):
                raise ValueError(
                    f"File {path} is not a valid TemporalReadmissionModel checkpoint."
                )

            config = checkpoint["config"]
            model_instance = cls(config=config)  # Instantiate using cls

            # Recreate model architecture
            arch_params = checkpoint["model_architecture_params"]
            model_instance.model = TimeAwarePatientLSTM(**arch_params)
            model_instance.model.load_state_dict(checkpoint["model_state_dict"])
            model_instance.model.to(device)  # Move to target device
            model_instance.device = device

            # Load metadata
            model_instance.feature_names = checkpoint.get("feature_names")
            model_instance.static_scaler = checkpoint.get("static_scaler")
            model_instance._static_medians = checkpoint.get("_static_medians")
            # Load sequence scaler if implemented
            model_instance.target = checkpoint.get("target")

            # Validate loaded components
            if model_instance.model is None:
                raise ValueError("Model state_dict loaded but model object is None.")
            if model_instance.static_scaler is None:
                logger.warning("Static scaler not found in checkpoint.")
            if model_instance.feature_names is None:
                logger.warning("Feature names not found in checkpoint.")

            logger.info(
                f"Successfully loaded Temporal model '{model_instance.model_type}' from {path} onto {device}"
            )
            return model_instance

        except FileNotFoundError:
            logger.error(f"Model file not found at {path}")
            raise
        except Exception as e:
            logger.error(
                f"Error loading TemporalReadmissionModel from {path}: {e}",
                exc_info=True,
            )
            raise

    # --- Override methods specific to PyTorch models ---
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Feature importance for LSTM is complex (e.g., SHAP on sequences). Not implemented."""
        self.logger.warning(
            "Standard feature importance not applicable to LSTM. Consider SHAP or attention analysis."
        )
        return pd.DataFrame()  # Return empty DataFrame

    def get_shap_values(
        self, data: pd.DataFrame, num_examples: Optional[int] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """SHAP for temporal models requires specialized explainers (e.g., DeepExplainer or GradientExplainer). Not implemented."""
        # TODO: Implement SHAP for temporal models if needed, likely using DeepExplainer or similar
        self.logger.warning(
            "SHAP for temporal models requires specific implementation (e.g., DeepExplainer). Not implemented."
        )
        # Placeholder return
        return np.array([]), pd.DataFrame()
