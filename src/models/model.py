"""
Model classes for the MIMIC project.

This module defines base model classes and specific implementations for
predicting readmission, mortality, length of stay, and temporal readmission.
It includes preprocessing, training, evaluation, prediction, saving, and loading logic.
"""

import logging  # Import logging for type hint
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union  # Added Union, Any

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset  # Added Dataset for type hint
from tqdm import tqdm

from src.models.temporal_modeling import TemporalEHRDataset, TimeAwarePatientLSTM

# Local application imports
from src.utils import get_logger, load_config

logger = get_logger(__name__)


def _reshape_data_to_admission_level(
    data: pd.DataFrame,
    id_cols: List[str],
    target_cols: List[str],
    logger_instance: logging.Logger,
) -> pd.DataFrame:
    """
    Reshapes data potentially containing multiple rows per admission (e.g., ICU stay features)
    to a single row per admission.

    It aggregates demographic features by taking the first non-null value and attempts
    to pivot clinical features based on 'category' and 'level_0' columns if present.

    Args:
        data (pd.DataFrame): Input DataFrame, potentially with multiple rows per admission.
                             Must contain 'subject_id' and 'hadm_id'.
        id_cols (List[str]): List of identifier columns (e.g., ['subject_id', 'hadm_id']).
        target_cols (List[str]): List of target variable column names.
        logger_instance (logging.Logger): Logger instance for logging messages.

    Returns:
        pd.DataFrame: A DataFrame with one row per unique admission (subject_id, hadm_id),
                      containing aggregated base/target columns, aggregated demographic
                      features, and potentially pivoted clinical features.

    Raises:
        ValueError: If 'subject_id' or 'hadm_id' are missing from the input data.
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
        and "0"
        in data.columns  # Assuming '0' holds the value after some upstream processing
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
                admission_base[[]],  # Use empty slice to just get the index
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
    Abstract Base Class for all predictive models in the project.

    Defines the common interface for preprocessing, training, evaluation,
    prediction, saving, and loading models. Also handles configuration loading
    and basic attribute initialization.

    Attributes:
        model_type (str): The type of model (e.g., 'readmission', 'mortality').
        config (Dict): The loaded configuration dictionary.
        random_state (int): Random state for reproducibility.
        logger (logging.Logger): Logger instance.
        model (Optional[BaseEstimator | nn.Module]): The trained model object. None until trained.
        feature_names (Optional[List[str]]): List of feature names used by the model. None until fitted.
        scaler (StandardScaler): Default scaler for non-temporal models. Temporal models use specific scalers.
        model_config (Dict): Configuration specific to this model type from the main config.
        target (Optional[str]): Name of the target variable column.
        algorithms (List[str]): List of algorithms specified for this model type in the config.
        cv_folds (int): Number of cross-validation folds specified in the config.
        hyperparameter_tuning (bool): Whether hyperparameter tuning is enabled in the config.
    """

    def __init__(
        self, model_type: str, config: Optional[Dict] = None, random_state: int = 42
    ) -> None:
        """
        Initialize the BaseModel.

        Args:
            model_type (str): Type of model ('readmission', 'mortality', 'los', 'temporal_readmission').
            config (Optional[Dict], optional): Configuration dictionary. If None, loads default. Defaults to None.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        self.model_type = model_type
        self.config = config if config is not None else load_config()
        self.random_state = random_state
        self.logger = logger
        self.model: Optional[Union[BaseEstimator, nn.Module]] = (
            None  # More specific type hint
        )
        self.feature_names: Optional[List[str]] = None
        self.scaler = StandardScaler()  # Default scaler for non-temporal models

        # Set model-specific configuration
        # Handle potential KeyError if model_type isn't in config (e.g., during direct instantiation)
        if "models" in self.config and model_type in self.config["models"]:
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
    ) -> Union[
        Tuple[pd.DataFrame, Optional[pd.Series]], Tuple[Dict, Optional[pd.Series]]
    ]:
        """
        Preprocess the input data for modelling.

        This method should handle tasks like feature selection, imputation,
        reshaping (if necessary), and extraction of the target variable.
        Temporal models might return a dictionary of processed data instead
        of a single DataFrame.

        Args:
            data (pd.DataFrame): Input data, typically the output of feature building.
            for_prediction (bool, optional): If True, preprocessing is for prediction,
                and the target variable (y) should be returned as None. Defaults to False.

        Returns:
            Union[Tuple[pd.DataFrame, Optional[pd.Series]], Tuple[Dict, Optional[pd.Series]]]:
                A tuple containing:
                - Preprocessed features (pd.DataFrame for standard models, Dict for temporal).
                - Target variable (pd.Series) if not for_prediction, otherwise None.
        """
        pass

    @abstractmethod
    def train(
        self,
        X_train: Union[pd.DataFrame, Dict],
        y_train: pd.Series,
        algorithm: Optional[str] = None,
    ) -> Union[BaseEstimator, nn.Module]:
        """
        Train the model on the provided training data.

        Args:
            X_train (Union[pd.DataFrame, Dict]): Training features (DataFrame for standard models,
                                                 Dict for temporal models).
            y_train (pd.Series): Training target variable.
            algorithm (Optional[str], optional): The specific algorithm to use for training,
                if applicable (e.g., 'logistic_regression'). If None, the first algorithm
                listed in the configuration for this model type might be used. Defaults to None.

        Returns:
            Union[BaseEstimator, nn.Module]: The trained model object (scikit-learn estimator or PyTorch module).
        """
        pass

    @abstractmethod
    def evaluate(
        self, X_test: Union[pd.DataFrame, Dict], y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the trained model on the test data.

        Args:
            X_test (Union[pd.DataFrame, Dict]): Test features (DataFrame for standard models,
                                                Dict for temporal models).
            y_test (pd.Series): Test target variable.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics (e.g., accuracy, AUC).
        """
        pass

    def fit(
        self,
        data: pd.DataFrame,
        algorithm: Optional[str] = None,
        test_size: float = 0.2,
    ) -> Dict[str, float]:
        """
        Fit the model to the data (preprocess, split, scale, train, evaluate).

        This is the main entry point for training standard (non-temporal) models.
        It orchestrates the preprocessing, data splitting, feature scaling,
        model training, and evaluation steps.

        Note:
            This default implementation assumes non-temporal data and scikit-learn
            style models where `preprocess` returns a DataFrame `X`. Temporal models
            (like `TemporalReadmissionModel`) override this method due to their
            different data structures and training loops.

        Args:
            data (pd.DataFrame): The input DataFrame containing features and the target variable.
            algorithm (Optional[str], optional): The specific algorithm to train. If None,
                delegates selection to the `train` method implementation. Defaults to None.
            test_size (float, optional): The proportion of the data to use for the test set.
                Defaults to 0.2.

        Returns:
            Dict[str, float]: Evaluation metrics calculated on the test set.

        Raises:
            ValueError: If preprocessing fails to return features (X) or the target (y)
                        when expected.
        """
        # Preprocess data
        X, y = self.preprocess(data)
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"BaseModel.fit expects preprocess to return a DataFrame for X, but got {type(X)}. Override fit for different preprocess outputs."
            )
        if y is None:
            raise ValueError(
                "Target variable 'y' is None after preprocessing during fit."
            )

        # Ensure X is not empty after preprocessing
        if X.empty:
            raise ValueError("Feature set 'X' is empty after preprocessing during fit.")

        self.feature_names = (
            X.columns.tolist()
        )  # Store feature names after preprocessing

        # Split data
        # Adjust stratify condition to handle cases with few samples per class
        stratify_cond = None
        if y.nunique() > 1 and len(y.value_counts()) > 1:
            min_samples_per_class = y.value_counts().min()
            # Ensure test set size allows at least 1 sample per class if possible, ideally 2 for StratifiedKFold
            n_splits_needed = (
                self.cv_folds if self.hyperparameter_tuning else 2
            )  # Need at least 2 for basic split
            # Also check if test_size * total_samples is large enough for stratification
            min_test_samples = max(
                n_splits_needed, int(np.ceil(test_size * len(y)))
            )  # Heuristic
            if min_samples_per_class >= min_test_samples:
                stratify_cond = y
            else:
                self.logger.warning(
                    f"Cannot stratify: Minimum samples per class ({min_samples_per_class}) might be too few for test size ({test_size}) and CV folds ({n_splits_needed})."
                )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_cond,
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
        Make predictions with the trained model.

        Handles preprocessing and scaling consistent with the training phase.
        Aligns columns, fills missing features with 0, and drops extra columns.

        Note:
            This default implementation assumes non-temporal data and scikit-learn
            style models where `preprocess` returns a DataFrame `X`. Temporal models
            override this method.

        Args:
            data (pd.DataFrame): Input data for prediction. Should contain columns
                                 required for preprocessing.

        Returns:
            np.ndarray: NumPy array of predictions.

        Raises:
            ValueError: If the model, feature names, or scaler have not been set (i.e., `fit` was not called).
            TypeError: If `preprocess` does not return a DataFrame for X.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        if self.feature_names is None:
            raise ValueError("Model has not been fitted yet (feature names not set)")
        if self.scaler is None:  # Check the default scaler
            raise ValueError("Scaler has not been fitted yet")

        # Preprocess data - crucial to use the same steps as training
        X, _ = self.preprocess(data, for_prediction=True)
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"BaseModel.predict expects preprocess to return a DataFrame for X, but got {type(X)}. Override predict for different preprocess outputs."
            )

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
        self.logger.info(f"Making predictions using model: {type(self.model).__name__}")
        return self.model.predict(X_scaled)

    def save(self, path: str) -> None:
        """
        Save the trained model, scaler, feature names, and config to disk using pickle.

        Note:
            Temporal models override this method to save the PyTorch model's state_dict
            and specific scalers instead of pickling the entire model object.

        Args:
            path (str): The file path to save the model data to.

        Raises:
            ValueError: If the model has not been trained yet.
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
        Load a model from disk.

        Handles loading both standard pickled scikit-learn models (saved via the
        default `save` method) and PyTorch models saved as state_dict checkpoints
        (specifically for `TemporalReadmissionModel`). It determines the loading
        method based on the file content and metadata.

        Args:
            path (str): The file path from which to load the model data.

        Returns:
            BaseModel: An instance of the appropriate model subclass (e.g.,
                       ReadmissionModel, TemporalReadmissionModel) with loaded
                       model object, scaler(s), feature names, and config.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            TypeError: If the loaded pickle file does not contain a dictionary.
            ValueError: If the loaded data is missing essential keys ('model_type', 'config')
                        or if the model object itself is missing after loading.
            Exception: Re-raises other unexpected errors during loading.
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
                    model_instance = cls(config=config)  # type: ignore # cls() is valid here

                # Recreate model architecture using saved parameters
                arch_params = checkpoint["model_architecture_params"]
                # Ensure device is part of arch_params or update it
                arch_params["device"] = device
                model_instance.model = TimeAwarePatientLSTM(**arch_params)
                model_instance.model.load_state_dict(checkpoint["model_state_dict"])
                model_instance.model.to(device)
                model_instance.device = device

                # Load metadata
                model_instance.feature_names = checkpoint.get("feature_names")
                model_instance.sequence_feature_names = checkpoint.get(
                    "sequence_feature_names"
                )  # Load specific feature lists
                model_instance.static_feature_names = checkpoint.get(
                    "static_feature_names"
                )
                model_instance.sequence_scaler = checkpoint.get("sequence_scaler")
                model_instance.static_scaler = checkpoint.get("static_scaler")
                model_instance.target = checkpoint.get("target")
                model_instance.model_architecture_params = (
                    arch_params  # Store loaded params
                )

                # Basic checks for loaded components
                if model_instance.model is None:
                    raise ValueError(
                        "Model state_dict loaded but model object is None."
                    )
                if model_instance.sequence_scaler is None:
                    logger.warning("Sequence scaler not found in loaded model data.")
                if model_instance.static_scaler is None:
                    logger.warning("Static scaler not found in loaded model data.")

                logger.info(
                    f"Loaded Temporal model '{model_instance.model_type}' from {path} onto {device}"
                )
                return model_instance  # type: ignore # Instance type matches TemporalReadmissionModel
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

            # Instantiate the correct class based on model_type
            if model_type == "readmission":
                model_instance = ReadmissionModel(config=config)
            elif model_type == "mortality":
                model_instance = MortalityModel(config=config)
            elif model_type == "los":
                model_instance = LengthOfStayModel(config=config)
            # Add other model types here if needed
            else:
                # Fallback or raise error if type is unknown
                logger.warning(
                    f"Unknown model_type '{model_type}' found in {path}. Attempting to load as generic BaseModel subclass if possible."
                )
                # Attempt to instantiate the class that called load (cls)
                try:
                    # Check if cls is the abstract BaseModel itself
                    if cls is BaseModel:
                        raise ValueError(
                            f"Cannot directly instantiate abstract BaseModel for unknown type '{model_type}'"
                        )
                    model_instance = cls(model_type=model_type, config=config)
                except (
                    TypeError
                ):  # Handle case where cls might be BaseModel (ABC) or other instantiation issues
                    raise ValueError(
                        f"Could not instantiate class '{cls.__name__}' for unknown model type '{model_type}'"
                    )

            # Load the actual model object and metadata
            model_instance.model = model_data.get("model")
            model_instance.feature_names = model_data.get("feature_names")
            model_instance.scaler = model_data.get("scaler")
            model_instance.target = model_data.get("target")  # Load target if saved

            if model_instance.model is None:
                raise ValueError(f"Model object not found in pickle file at {path}")
            if model_instance.feature_names is None:
                logger.warning(f"Feature names not found in pickle file at {path}")
            if model_instance.scaler is None:
                logger.warning(
                    f"Scaler not found in pickle file at {path}. Using default StandardScaler."
                )
                model_instance.scaler = StandardScaler()

            logger.info(f"Loaded model '{model_type}' from {path}")
            return model_instance

        except FileNotFoundError:
            logger.error(f"Model file not found at {path}")
            raise
        except (pickle.UnpicklingError, TypeError, ValueError) as e:
            logger.error(f"Error loading model from {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred loading model from {path}: {e}")
            raise

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Supports tree-based models (like RF, XGBoost, LightGBM) with
        `feature_importances_` attribute and linear models with `coef_` attribute.
        For linear models with multi-class classification (coef_ shape > 1),
        it averages the absolute coefficients across classes.

        Args:
            top_n (int, optional): The number of top features to return. Defaults to 20.

        Returns:
            pd.DataFrame: A DataFrame with 'feature' and 'importance' columns,
                          sorted by importance in descending order. Returns an empty
                          DataFrame if importance cannot be determined.

        Raises:
            ValueError: If the model has not been trained or feature names are not set.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if self.feature_names is None:
            raise ValueError("Feature names not set. Fit the model first.")

        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # For linear models, use absolute coefficient values
            if self.model.coef_.ndim == 1:  # Binary classification/regression
                importances = np.abs(self.model.coef_)
            elif self.model.coef_.ndim == 2:  # Multiclass classification
                # Average absolute coefficients across classes
                importances = np.mean(np.abs(self.model.coef_), axis=0)

        if importances is not None:
            # Ensure importances array matches feature_names length
            if len(importances) != len(self.feature_names):
                self.logger.error(
                    f"Mismatch between number of importances ({len(importances)}) and feature names ({len(self.feature_names)})."
                )
                return pd.DataFrame(columns=["feature", "importance"])

            feature_importance_df = pd.DataFrame(
                {"feature": self.feature_names, "importance": importances}
            )
            feature_importance_df = feature_importance_df.sort_values(
                "importance", ascending=False
            ).head(top_n)
            return feature_importance_df
        else:
            self.logger.warning(
                f"Model type {type(self.model)} does not have standard feature_importances_ or coef_ attribute."
            )
            return pd.DataFrame(columns=["feature", "importance"])

    def get_shap_values(
        self, X: pd.DataFrame, background_data: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Any]]:  # Changed shap.Explainer to Any
        """
        Calculate SHAP values for the given data using shap.KernelExplainer.

        Note: This method currently uses KernelExplainer, which can be slow for large datasets.
              It requires a background dataset for initialization.

        Args:
            X (pd.DataFrame): Data for which to calculate SHAP values (should be preprocessed and scaled).
            background_data (Optional[pd.DataFrame]): Background dataset for initializing the explainer
                                                     (often the training set or a sample). Should be
                                                     preprocessed and scaled. If None, attempts to use
                                                     a default background set if available.

        Returns:
            Tuple[Optional[np.ndarray], Optional[Any]]:
                - SHAP values array (or None if calculation fails). For classification,
                  this might be a list of arrays (one per class).
                - SHAP explainer object (or None if calculation fails). Type is Any due to shap library variations.

        Raises:
            ValueError: If the model or feature names are not set.
            KeyError: If expected feature columns are missing in X or background_data.
        """
        if self.model is None:
            raise ValueError("Cannot calculate SHAP values: Model not trained.")
        if self.feature_names is None:
            raise ValueError("Cannot calculate SHAP values: Feature names not set.")

        # Ensure columns match feature names used during training
        try:
            X = X[self.feature_names]
            if background_data is not None:
                background_data = background_data[self.feature_names]
        except KeyError as e:
            self.logger.error(
                f"Missing expected feature column for SHAP calculation: {e}"
            )
            return None, None

        self.logger.info("Calculating SHAP values...")
        explainer = None
        shap_values = None

        try:
            # Use KernelExplainer for models not directly supported by TreeExplainer or LinearExplainer
            # Requires a predict_proba function (for classification) or predict (for regression)
            if hasattr(self.model, "predict_proba"):
                # Wrap predict_proba for KernelExplainer
                def predict_fn(data_np: np.ndarray) -> np.ndarray:
                    # SHAP passes numpy arrays
                    data_df = pd.DataFrame(data_np, columns=self.feature_names)
                    return self.model.predict_proba(data_df)  # type: ignore # Assume predict_proba exists

                # Use a sample of the background data if it's large
                if background_data is not None:
                    if len(background_data) > 100:
                        background_sample = shap.sample(background_data, 100)
                    else:
                        background_sample = background_data
                    explainer = shap.KernelExplainer(predict_fn, background_sample)
                else:
                    self.logger.warning(
                        "No background data provided for SHAP KernelExplainer. Results may be less reliable."
                    )
                    # Create a dummy background if none provided (less ideal)
                    dummy_background = pd.DataFrame(
                        np.zeros((1, len(self.feature_names))),
                        columns=self.feature_names,
                    )
                    explainer = shap.KernelExplainer(predict_fn, dummy_background)

            elif hasattr(self.model, "predict"):  # Regression case

                def predict_fn_reg(data_np: np.ndarray) -> np.ndarray:
                    data_df = pd.DataFrame(data_np, columns=self.feature_names)
                    return self.model.predict(data_df)  # type: ignore # Assume predict exists

                if background_data is not None:
                    if len(background_data) > 100:
                        background_sample = shap.sample(background_data, 100)
                    else:
                        background_sample = background_data
                    explainer = shap.KernelExplainer(predict_fn_reg, background_sample)
                else:
                    self.logger.warning(
                        "No background data provided for SHAP KernelExplainer. Results may be less reliable."
                    )
                    dummy_background = pd.DataFrame(
                        np.zeros((1, len(self.feature_names))),
                        columns=self.feature_names,
                    )
                    explainer = shap.KernelExplainer(predict_fn_reg, dummy_background)
            else:
                self.logger.error(
                    "Model has neither predict_proba nor predict method for SHAP."
                )
                return None, None

            # Calculate SHAP values for the input data X
            # Use appropriate slicing/indexing based on model output shape if needed
            shap_values = explainer.shap_values(X)
            self.logger.info("SHAP values calculated successfully.")

        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {e}", exc_info=True)
            return None, None

        return shap_values, explainer


class ReadmissionModel(BaseModel):
    """
    Model for predicting hospital readmission.

    Inherits from BaseModel and provides specific implementations for
    preprocessing, training, and evaluation tailored for readmission prediction.
    Defaults to 'readmission_30day' as the target variable.
    """

    def __init__(self, config: Optional[Dict] = None, random_state: int = 42) -> None:
        """
        Initialize the readmission model.

        Args:
            config (Optional[Dict], optional): Configuration dictionary. Defaults to None.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        super().__init__(
            model_type="readmission", config=config, random_state=random_state
        )
        if self.target is None:  # Set default target if not in config
            self.logger.warning(
                "Readmission target not specified in config, defaulting to 'readmission_30day'."
            )
            self.target = "readmission_30day"

    def preprocess(
        self, data: pd.DataFrame, for_prediction: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess data specifically for the readmission model.

        Handles potential reshaping to one row per admission, selects features based
        on configuration ('numeric_features', 'categorical_features'), performs
        basic imputation (median for numeric, mode for categorical), and extracts
        the target variable ('readmission_30day' by default).

        Args:
            data (pd.DataFrame): Input DataFrame, potentially containing multiple rows
                                 per admission and columns from feature extraction.
            for_prediction (bool, optional): If True, preprocessing is for prediction,
                                             and the target variable (y) is None. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]:
                - X (pd.DataFrame): DataFrame containing only the selected and imputed features.
                - y (Optional[pd.Series]): Series containing the target variable (integer type)
                                            if not for_prediction, otherwise None.
        """
        self.logger.info(
            f"Preprocessing data for {'prediction' if for_prediction else 'training'}..."
        )

        # --- Reshape if necessary ---
        # Check if data is already one row per admission
        id_cols = ["subject_id", "hadm_id"]
        is_reshaped = not data.duplicated(subset=id_cols).any()

        if not is_reshaped:
            self.logger.info("Data is not one row per admission. Reshaping...")
            target_cols = (
                [self.target] if self.target and self.target in data.columns else []
            )
            try:
                data = _reshape_data_to_admission_level(
                    data, id_cols, target_cols, self.logger
                )
            except Exception as e:
                self.logger.error(f"Error during data reshaping: {e}", exc_info=True)
                # Return empty dataframes if reshaping fails critically
                return pd.DataFrame(), (
                    None
                    if for_prediction
                    else pd.Series(dtype=int)  # Return empty Series on error
                )
        else:
            self.logger.info(
                "Data is already one row per admission. Skipping reshaping."
            )

        # --- Feature Selection ---
        # Define features based on config (or use all available if not specified)
        # Attempt to load features from config
        conf_numeric_features = self.model_config.get("numeric_features", [])
        conf_categorical_features = self.model_config.get("categorical_features", [])
        all_expected_features = conf_numeric_features + conf_categorical_features

        if all_expected_features:
            # Features ARE specified in config: Select available ones among expected
            self.logger.info("Using features specified in configuration.")
            available_features = [
                col for col in all_expected_features if col in data.columns
            ]
            missing_expected = set(all_expected_features) - set(available_features)
            if missing_expected:
                self.logger.warning(
                    f"Expected features missing from data after reshaping: {missing_expected}. They will be excluded."
                )
        else:
            # Features ARE NOT specified in config: Dynamically identify
            self.logger.info(
                "No features specified in config. Dynamically identifying numeric and categorical features."
            )
            potential_features = data.columns.tolist()
            # Exclude IDs and target
            exclude_cols = id_cols + (
                [self.target] if self.target and self.target in data.columns else []
            )
            available_features = [
                col for col in potential_features if col not in exclude_cols
            ]
            # Further refine based on dtype (optional but good practice)
            numeric_features = (
                data[available_features]
                .select_dtypes(include=np.number)
                .columns.tolist()
            )
            categorical_features = (
                data[available_features]
                .select_dtypes(include="object")
                .columns.tolist()
            )  # Include 'category' as well if needed
            available_features = (
                numeric_features + categorical_features
            )  # Use only identified numeric/cat features
            self.logger.info(
                f"Dynamically identified {len(numeric_features)} numeric and {len(categorical_features)} categorical features."
            )

        if not available_features:
            self.logger.error(
                "No features found in the data (either configured or dynamically identified)."
            )
            # Return empty DataFrame and empty Series if no features are found
            return pd.DataFrame(), (
                None
                if for_prediction
                else pd.Series(dtype=int)  # Return empty Series on error
            )

        # Select the final set of features
        self.logger.info(f"Using {len(available_features)} features for the model.")
        self.feature_names = available_features  # Store feature names used
        X = data[self.feature_names].copy()

        # --- Handle Target Variable ---
        y = None
        if not for_prediction:
            if self.target and self.target in data.columns:
                y = data[self.target].astype(int)  # Ensure target is integer
                self.logger.info(f"Target variable '{self.target}' extracted.")
            else:
                self.logger.error(
                    f"Target variable '{self.target}' not found in data columns: {data.columns.tolist()}"
                )
                # Return empty dataframes if target is missing during training
                return pd.DataFrame(), pd.Series(dtype=int)

        # --- Imputation (Example: Simple mean/median/mode imputation) ---
        # More sophisticated imputation might be needed
        for col in X.select_dtypes(include=np.number).columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                self.logger.debug(
                    f"Imputed missing numeric values in '{col}' with median ({median_val})."
                )

        for col in X.select_dtypes(include="object").columns:
            if X[col].isnull().any():
                mode_val = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
                X[col] = X[col].fillna(mode_val)
                self.logger.debug(
                    f"Imputed missing categorical values in '{col}' with mode ('{mode_val}')."
                )

        self.logger.info(f"Preprocessing complete. Feature shape: {X.shape}")
        return X, y

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, algorithm: Optional[str] = None
    ) -> BaseEstimator:
        """
        Train the readmission model using the specified algorithm.

        Supports 'logistic_regression', 'random_forest', 'xgboost', 'lightgbm'.
        Handles class weighting based on configuration ('balanced' or custom dict).
        Optionally performs hyperparameter tuning if enabled in config (currently
        only implemented for logistic regression).

        Args:
            X_train (pd.DataFrame): Training features (scaled).
            y_train (pd.Series): Training target variable.
            algorithm (Optional[str], optional): Algorithm name. If None, uses the first
                                                 algorithm in the config or defaults to
                                                 'logistic_regression'. Defaults to None.

        Returns:
            BaseEstimator: The trained scikit-learn compatible model estimator.

        Raises:
            ValueError: If an unsupported algorithm is specified.
        """
        if algorithm is None:
            algorithm = self.algorithms[0] if self.algorithms else "logistic_regression"
            self.logger.info(f"No algorithm specified, using default: {algorithm}")

        self.logger.info(f"Training {algorithm} model for {self.model_type}...")

        # Get class weight parameter if configured
        class_weight_config = self.model_config.get("class_weight", None)
        if class_weight_config == "balanced":
            class_weight = "balanced"
        elif isinstance(class_weight_config, dict):
            # Ensure keys are integers if they represent classes 0 and 1
            class_weight = {int(k): v for k, v in class_weight_config.items()}
        else:
            class_weight = None

        # Define models
        if algorithm == "logistic_regression":
            model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,  # Increase max_iter for convergence
                class_weight=class_weight,
                solver="liblinear",  # Good default for small datasets and L1/L2 penalties
            )
        elif algorithm == "random_forest":
            model = RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,  # Default, consider tuning
                class_weight=class_weight,
            )
        elif algorithm == "xgboost":
            # Calculate scale_pos_weight for imbalance if class_weight is 'balanced'
            scale_pos_weight = None
            if class_weight == "balanced":
                counts = y_train.value_counts()
                if 1 in counts and 0 in counts and counts[1] > 0:
                    scale_pos_weight = counts[0] / counts[1]
                else:
                    self.logger.warning(
                        "Cannot calculate scale_pos_weight for XGBoost, using default."
                    )

            model = xgb.XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,  # Use calculated weight
            )
        elif algorithm == "lightgbm":
            # Calculate scale_pos_weight for imbalance if class_weight is 'balanced'
            scale_pos_weight = None
            if class_weight == "balanced":
                counts = y_train.value_counts()
                if 1 in counts and 0 in counts and counts[1] > 0:
                    scale_pos_weight = counts[0] / counts[1]
                else:
                    self.logger.warning(
                        "Cannot calculate scale_pos_weight for LightGBM, using default."
                    )

            model = lgb.LGBMClassifier(
                random_state=self.random_state,
                scale_pos_weight=scale_pos_weight,  # Use calculated weight
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Hyperparameter tuning (optional)
        if self.hyperparameter_tuning:
            # Define parameter grid (example for logistic regression)
            # TODO: Define grids for other algorithms
            if algorithm == "logistic_regression":
                param_grid = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}
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
                    f"Hyperparameter tuning not implemented for {algorithm}. Using default parameters."
                )
                model.fit(
                    X_train, y_train
                )  # Fit with default params if tuning not implemented
        else:
            model.fit(X_train, y_train)

        self.logger.info(f"{algorithm} model training complete.")
        return model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the trained readmission model on the test set.

        Calculates and logs standard classification metrics: accuracy, precision,
        recall, F1-score, ROC AUC, and PR AUC. Also logs the confusion matrix.

        Args:
            X_test (pd.DataFrame): Test features (scaled).
            y_test (pd.Series): Test target variable.

        Returns:
            Dict[str, float]: Dictionary containing the calculated metric names and values.
                              AUC metrics will be NaN if the model lacks `predict_proba`.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        self.logger.info(f"Evaluating {self.model_type} model...")
        y_pred = self.model.predict(X_test)
        y_prob = (
            self.model.predict_proba(X_test)[:, 1]
            if hasattr(self.model, "predict_proba")
            else y_pred  # Use predictions if predict_proba not available
        )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": (
                roc_auc_score(y_test, y_prob)
                if hasattr(self.model, "predict_proba")
                else np.nan
            ),
            "pr_auc": (
                average_precision_score(y_test, y_prob)
                if hasattr(self.model, "predict_proba")
                else np.nan
            ),
        }

        # Log metrics
        for metric, value in metrics.items():
            self.logger.info(f"{metric.replace('_', ' ').title()}: {value:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self.logger.info(f"Confusion Matrix:\n{cm}")

        return metrics


class MortalityModel(BaseModel):
    """
    Model for predicting in-hospital mortality.

    Inherits from BaseModel and provides specific implementations for
    preprocessing, training, and evaluation tailored for mortality prediction.
    Defaults to 'hospital_death' as the target variable.
    """

    def __init__(self, config: Optional[Dict] = None, random_state: int = 42) -> None:
        """
        Initialize the mortality model.

        Args:
            config (Optional[Dict], optional): Configuration dictionary. Defaults to None.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        super().__init__(
            model_type="mortality", config=config, random_state=random_state
        )
        if self.target is None:  # Set default target if not in config
            self.logger.warning(
                "Mortality target not specified in config, defaulting to 'hospital_death'."
            )
            self.target = "hospital_death"

    def preprocess(
        self, data: pd.DataFrame, for_prediction: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess data specifically for the mortality model.

        (Currently delegates to ReadmissionModel's preprocess method, assuming
         similar feature requirements and handling. Can be specialized if needed.)

        Args:
            data (pd.DataFrame): Input DataFrame.
            for_prediction (bool, optional): If True, return None for target. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Preprocessed features (X) and target (y).
        """
        self.logger.info(
            f"Preprocessing data for mortality {'prediction' if for_prediction else 'training'}..."
        )
        # Use ReadmissionModel's preprocessing logic for now
        temp_model = ReadmissionModel(
            config=self.config, random_state=self.random_state
        )
        temp_model.target = self.target  # Ensure correct target is used
        temp_model.model_config = (
            self.model_config
        )  # Ensure correct config section is used
        return temp_model.preprocess(data, for_prediction=for_prediction)

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, algorithm: Optional[str] = None
    ) -> BaseEstimator:
        """
        Train the mortality model.

        (Currently delegates to ReadmissionModel's train method, assuming the same
         algorithms and training procedures apply. Can be specialized if needed.)

        Args:
            X_train (pd.DataFrame): Training features (scaled).
            y_train (pd.Series): Training target variable.
            algorithm (Optional[str], optional): Algorithm name. Defaults to None.

        Returns:
            BaseEstimator: The trained scikit-learn compatible model estimator.
        """
        if algorithm is None:
            algorithm = self.algorithms[0] if self.algorithms else "logistic_regression"
            self.logger.info(f"No algorithm specified, using default: {algorithm}")

        # Delegate to the ReadmissionModel's train method logic
        temp_model = ReadmissionModel(
            config=self.config, random_state=self.random_state
        )
        temp_model.model_type = self.model_type  # Ensure correct type logging
        temp_model.model_config = self.model_config
        temp_model.target = self.target
        return temp_model.train(X_train, y_train, algorithm)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the mortality model.

        (Currently delegates to ReadmissionModel's evaluate method, assuming the same
         metrics are relevant. Can be specialized if needed.)

        Args:
            X_test (pd.DataFrame): Test features (scaled).
            y_test (pd.Series): Test target variable.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Delegate to the ReadmissionModel's evaluate method logic
        temp_model = ReadmissionModel(
            config=self.config, random_state=self.random_state
        )
        temp_model.model = self.model  # Use the trained model
        temp_model.model_type = self.model_type
        return temp_model.evaluate(X_test, y_test)


class LengthOfStayModel(BaseModel):
    """
    Model for predicting length of stay (regression).

    Inherits from BaseModel and provides specific implementations for
    preprocessing, training, and evaluation tailored for LOS prediction.
    Defaults to 'los_days' as the target variable and supports optional
    log transformation of the target. Uses regression algorithms.
    """

    def __init__(self, config: Optional[Dict] = None, random_state: int = 42) -> None:
        """
        Initialize the length of stay model.

        Args:
            config (Optional[Dict], optional): Configuration dictionary. Defaults to None.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        super().__init__(model_type="los", config=config, random_state=random_state)
        if self.target is None:  # Set default target if not in config
            self.logger.warning(
                "Length of Stay target not specified in config, defaulting to 'los_days'."
            )
            self.target = "los_days"

    def preprocess(
        self, data: pd.DataFrame, for_prediction: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess data specifically for the length of stay model.

        Handles reshaping, feature selection, imputation, and extracts the
        continuous target variable ('los_days' by default). Optionally applies
        log transformation (log1p) to the target variable based on config.

        Args:
            data (pd.DataFrame): Input DataFrame.
            for_prediction (bool, optional): If True, return None for target. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Preprocessed features (X) and target (y).
        """
        self.logger.info(
            f"Preprocessing data for LOS {'prediction' if for_prediction else 'training'}..."
        )

        # --- Reshape if necessary ---
        id_cols = ["subject_id", "hadm_id"]
        is_reshaped = not data.duplicated(subset=id_cols).any()

        if not is_reshaped:
            self.logger.info("Data is not one row per admission. Reshaping...")
            target_cols = (
                [self.target] if self.target and self.target in data.columns else []
            )
            try:
                data = _reshape_data_to_admission_level(
                    data, id_cols, target_cols, self.logger
                )
            except Exception as e:
                self.logger.error(f"Error during data reshaping: {e}", exc_info=True)
                return pd.DataFrame(), (
                    None if for_prediction else (pd.DataFrame(), pd.Series(dtype=float))
                )
        else:
            self.logger.info(
                "Data is already one row per admission. Skipping reshaping."
            )

        # --- Feature Selection ---
        numeric_features = self.model_config.get("numeric_features", [])
        categorical_features = self.model_config.get("categorical_features", [])
        all_expected_features = numeric_features + categorical_features

        available_features = [
            col for col in all_expected_features if col in data.columns
        ]
        missing_expected = set(all_expected_features) - set(available_features)
        if missing_expected:
            self.logger.warning(
                f"Expected features missing from data after reshaping: {missing_expected}. They will be excluded."
            )

        if not available_features:
            self.logger.error(
                "No configured features found in the data after reshaping."
            )
            return pd.DataFrame(), (
                None if for_prediction else (pd.DataFrame(), pd.Series(dtype=float))
            )

        X = data[available_features].copy()

        # --- Handle Target Variable ---
        y = None
        if not for_prediction:
            if self.target and self.target in data.columns:
                y = data[self.target]  # Target is continuous
                # Optional: Log transform LOS if distribution is skewed
                if self.model_config.get("log_transform_target", False):
                    y = np.log1p(y)
                    self.logger.info(
                        f"Log-transformed target variable '{self.target}'."
                    )
                self.logger.info(f"Target variable '{self.target}' extracted.")
            else:
                self.logger.error(
                    f"Target variable '{self.target}' not found in data columns: {data.columns.tolist()}"
                )
                return pd.DataFrame(), pd.Series(dtype=float)

        # --- Imputation ---
        for col in X.select_dtypes(include=np.number).columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                self.logger.debug(
                    f"Imputed missing numeric values in '{col}' with median ({median_val})."
                )

        for col in X.select_dtypes(include="object").columns:
            if X[col].isnull().any():
                mode_val = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
                X[col] = X[col].fillna(mode_val)
                self.logger.debug(
                    f"Imputed missing categorical values in '{col}' with mode ('{mode_val}')."
                )

        self.logger.info(f"Preprocessing complete. Feature shape: {X.shape}")
        return X, y

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, algorithm: Optional[str] = None
    ) -> BaseEstimator:
        """
        Train the length of stay model (regression).

        Supports 'linear_regression', 'random_forest_regressor', 'xgboost_regressor',
        'lightgbm_regressor'.

        Args:
            X_train (pd.DataFrame): Training features (scaled).
            y_train (pd.Series): Training target variable (potentially log-transformed).
            algorithm (Optional[str], optional): Algorithm name. If None, uses the first
                                                 algorithm in the config or defaults to
                                                 'linear_regression'. Defaults to None.

        Returns:
            BaseEstimator: The trained scikit-learn compatible regression model.

        Raises:
            ValueError: If an unsupported algorithm is specified.
        """
        if algorithm is None:
            algorithm = self.algorithms[0] if self.algorithms else "linear_regression"
            self.logger.info(f"No algorithm specified, using default: {algorithm}")

        self.logger.info(f"Training {algorithm} model for {self.model_type}...")

        # Define models
        if algorithm == "linear_regression":
            model = LinearRegression()
        elif algorithm == "random_forest_regressor":
            model = RandomForestRegressor(
                random_state=self.random_state,
                n_estimators=100,  # Default, consider tuning
            )
        elif algorithm == "xgboost_regressor":
            model = xgb.XGBRegressor(
                random_state=self.random_state,
                objective="reg:squarederror",  # Common objective for regression
                eval_metric="rmse",
            )
        elif algorithm == "lightgbm_regressor":
            model = lgb.LGBMRegressor(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported algorithm for regression: {algorithm}")

        # Hyperparameter tuning (optional) - Example for Linear Regression (none needed)
        if self.hyperparameter_tuning:
            # TODO: Add hyperparameter grids and tuning for regression models
            self.logger.warning(
                f"Hyperparameter tuning not implemented for {algorithm}. Using default parameters."
            )
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        self.logger.info(f"{algorithm} model training complete.")
        return model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the length of stay regression model.

        Calculates and logs standard regression metrics: MAE, MSE, RMSE, R2,
        and Explained Variance. Handles inverse transformation of predictions
        and test target if log transformation was applied during preprocessing.

        Args:
            X_test (pd.DataFrame): Test features (scaled).
            y_test (pd.Series): Test target variable (potentially log-transformed).

        Returns:
            Dict[str, float]: Dictionary containing the calculated metric names and values.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        self.logger.info(f"Evaluating {self.model_type} model...")
        y_pred = self.model.predict(X_test)

        # Inverse transform predictions if target was log-transformed
        if self.model_config.get("log_transform_target", False):
            self.logger.info(
                "Applying inverse log transformation (expm1) to predictions and target for evaluation."
            )
            y_pred = np.expm1(y_pred)
            y_test = np.expm1(
                y_test
            )  # Also inverse transform test set for correct metric calculation
            # Clip negative predictions after inverse transform (LOS cannot be negative)
            y_pred = np.maximum(y_pred, 0)

        metrics = {
            "mean_absolute_error": mean_absolute_error(y_test, y_pred),
            "mean_squared_error": mean_squared_error(y_test, y_pred),
            "root_mean_squared_error": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2_score": r2_score(y_test, y_pred),
            "explained_variance": explained_variance_score(y_test, y_pred),
        }

        # Log metrics
        for metric, value in metrics.items():
            self.logger.info(f"{metric.replace('_', ' ').title()}: {value:.4f}")

        return metrics


# --- Temporal Model ---
# Note: This model has significantly different preprocessing, training, and evaluation logic.


class TemporalReadmissionModel(BaseModel):
    """
    Model for predicting readmission using temporal data (Time-Aware LSTM).

    Inherits from BaseModel but overrides `preprocess`, `train`, `evaluate`,
    `predict`, `save`, and `load` to handle sequence data and PyTorch models.
    Uses specific scalers for sequence and static features.
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, random_state: int = 42
    ) -> None:
        """
        Initialize the temporal readmission model.

        Sets up device (CPU/GPU) and specific scalers for sequence/static data.

        Args:
            config (Optional[Dict], optional): Configuration dictionary. Defaults to None.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        super().__init__(
            model_type="temporal_readmission", config=config, random_state=random_state
        )
        if self.target is None:
            self.logger.warning(
                "Temporal Readmission target not specified, defaulting to 'readmission_30day'."
            )
            self.target = "readmission_30day"

        # Temporal models use PyTorch, set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Initialize scalers specific to temporal data (sequence and static)
        self.sequence_scaler = StandardScaler()
        self.static_scaler = StandardScaler()
        # Override the default scaler from BaseModel as it's not used here
        self.scaler = None

        # Attributes specific to temporal model structure
        self.sequence_feature_names: Optional[List[str]] = None
        self.static_feature_names: Optional[List[str]] = None
        self.model_architecture_params: Optional[Dict[str, Any]] = None

    def _pad_sequence(
        self, seq: List[Tuple[float, float]], max_len: int, value: float = 0.0
    ) -> List[Tuple[float, float]]:
        """
        Pads a sequence of (time, value) tuples to the specified max length.

        Args:
            seq (List[Tuple[float, float]]): The input sequence of (time, value) tuples.
            max_len (int): The target length to pad to.
            value (float, optional): The value to use for padding. Time is padded with 0.0. Defaults to 0.0.

        Returns:
            List[Tuple[float, float]]: The padded sequence.
        """
        padding_needed = max_len - len(seq)
        if padding_needed > 0:
            # Pad with (time=0.0, value=specified_padding_value)
            return seq + [(0.0, value)] * padding_needed
        return seq[:max_len]  # Truncate if longer

    def _calculate_intervals(self, timestamps: List[float]) -> List[float]:
        """
        Calculates time intervals (delta time) between consecutive events in a sequence.
        The first interval is always 0.0.

        Args:
            timestamps (List[float]): A list of timestamps (e.g., hours from admission).

        Returns:
            List[float]: A list of time intervals between consecutive timestamps.
        """
        if not timestamps:
            return []
        intervals = [0.0] + [
            timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))
        ]
        return intervals

    def preprocess(
        self, data: pd.DataFrame, for_prediction: bool = False
    ) -> Tuple[Dict[str, Any], Optional[pd.Series]]:
        """
        Preprocess data for the temporal model.

        Extracts sequences, static features, calculates sequence lengths, pads sequences,
        calculates time intervals, scales sequence values and static features separately.

        Args:
            data (pd.DataFrame): Input DataFrame containing sequence columns
                                 (ending in '_sequence_data') and static feature columns.
                                 Assumes data is one row per admission.
            for_prediction (bool, optional): If True, preprocessing is for prediction,
                                             and the target variable (y) is None. Defaults to False.

        Returns:
            Tuple[Dict[str, Any], Optional[pd.Series]]:
                - X_processed (Dict[str, Any]): A dictionary containing:
                    - 'sequences' (Dict[str, np.ndarray]): Dict of numpy arrays for scaled values,
                      timestamps, and intervals for each sequence feature. Shape: (n_samples, max_len).
                    - 'static_features' (pd.DataFrame): DataFrame of scaled static features.
                    - 'sequence_lengths' (List[int]): List of original lengths for each sequence.
                    - 'admission_ids' (pd.DataFrame): DataFrame with 'subject_id', 'hadm_id'.
                - y (Optional[pd.Series]): Series containing the target variable (integer type)
                                            if not for_prediction, otherwise None.
        """
        self.logger.info(
            f"Preprocessing data for temporal {'prediction' if for_prediction else 'training'}..."
        )

        # Ensure data is one row per admission (should be done before calling this)
        id_cols = ["subject_id", "hadm_id"]
        if data.duplicated(subset=id_cols).any():
            self.logger.warning(
                "Input data contains multiple rows per admission. Ensure data is pre-aggregated."
            )
            # Attempt aggregation if needed (though ideally done upstream)
            target_cols = (
                [self.target] if self.target and self.target in data.columns else []
            )
            try:
                data = _reshape_data_to_admission_level(
                    data, id_cols, target_cols, self.logger
                )
            except Exception as e:
                self.logger.error(f"Error during data reshaping: {e}", exc_info=True)
                # Return empty dict and empty Series (int type for labels) if not predicting
                return {}, None if for_prediction else pd.Series(dtype=int)

        # --- Identify Feature Columns ---
        sequence_cols = [col for col in data.columns if col.endswith("_sequence_data")]
        static_cols_config = self.model_config.get("static_features", [])
        # Select static features that actually exist in the data
        static_cols = [col for col in static_cols_config if col in data.columns]
        missing_static = set(static_cols_config) - set(static_cols)
        if missing_static:
            self.logger.warning(
                f"Configured static features not found in data: {missing_static}"
            )

        self.logger.info(f"Identified {len(sequence_cols)} sequence features.")
        self.logger.info(
            f"Identified {len(static_cols)} static features: {static_cols}"
        )

        # Store feature names (distinguish sequence and static)
        self.sequence_feature_names = sequence_cols
        self.static_feature_names = static_cols
        self.feature_names = (
            self.sequence_feature_names + self.static_feature_names
        )  # Combined list for compatibility

        # --- Extract Target ---
        y = None
        if not for_prediction:
            if self.target and self.target in data.columns:
                y = data[self.target].astype(int)
                self.logger.info(f"Target variable '{self.target}' extracted.")
            else:
                self.logger.error(f"Target variable '{self.target}' not found.")
                return {}, None  # Return empty dict and None

        # --- Process Sequences ---
        processed_sequences = {}
        all_sequence_lengths = []
        max_len = 0

        # First pass: Extract sequences and find max length
        for col in self.sequence_feature_names:
            # Handle potential NaN/missing sequences gracefully -> empty list
            # Ensure evaluation of string representation if data loaded from CSV
            def _eval_seq(x: Any) -> List[Any]:
                if isinstance(x, list):
                    return x
                try:
                    return eval(x) if isinstance(x, str) else []
                except:
                    return []

            sequences = data[col].apply(_eval_seq)
            processed_sequences[col] = sequences.tolist()
            current_lengths = sequences.apply(len).tolist()
            all_sequence_lengths.append(current_lengths)
            max_len = max(max_len, max(current_lengths) if current_lengths else 0)

        self.logger.info(f"Maximum sequence length found: {max_len}")

        # --- Pad Sequences and Extract Values/Timestamps/Intervals ---
        final_sequences = {}
        num_admissions = len(data)

        for col in self.sequence_feature_names:
            padded_values = np.zeros((num_admissions, max_len))
            padded_timestamps = np.zeros((num_admissions, max_len))
            padded_intervals = np.zeros((num_admissions, max_len))

            for i, seq in enumerate(processed_sequences[col]):
                seq_len = len(seq)
                if seq_len > 0:
                    # Unzip the list of (time, value) tuples
                    try:
                        # Ensure timestamps and values are lists from the start
                        timestamps, values = map(list, zip(*seq))
                        intervals = self._calculate_intervals(list(timestamps))

                        # Pad if necessary
                        if seq_len < max_len:
                            values = list(values) + [0.0] * (max_len - seq_len)
                            timestamps = list(timestamps) + [0.0] * (max_len - seq_len)
                            intervals = list(intervals) + [0.0] * (max_len - seq_len)

                        padded_values[i, :] = values[:max_len]  # Ensure correct length
                        padded_timestamps[i, :] = timestamps[:max_len]
                        padded_intervals[i, :] = intervals[:max_len]
                    except Exception as e:
                        self.logger.warning(
                            f"Error processing sequence for {col}, admission index {i}: {e}. Sequence: {seq}"
                        )
                # else: arrays remain zeros

            final_sequences[f"{col}_values"] = padded_values
            final_sequences[f"{col}_timestamps"] = padded_timestamps
            final_sequences[f"{col}_intervals"] = padded_intervals

        # --- Scale Sequence Data ---
        # Reshape for scaler: (num_admissions * max_len, num_sequence_features)
        num_sequence_features = len(self.sequence_feature_names)
        # Handle case where there are no sequence features
        if num_sequence_features > 0:
            value_data_to_scale = np.zeros(
                (num_admissions * max_len, num_sequence_features)
            )
            for idx, col in enumerate(self.sequence_feature_names):
                value_data_to_scale[:, idx] = final_sequences[f"{col}_values"].ravel()

            # Fit scaler only during training, otherwise transform
            if not for_prediction:
                self.logger.info("Fitting sequence scaler...")
                scaled_value_data = self.sequence_scaler.fit_transform(
                    value_data_to_scale
                )
            else:
                if not hasattr(
                    self.sequence_scaler, "mean_"
                ):  # Check if scaler is fitted
                    self.logger.error(
                        "Sequence scaler has not been fitted. Cannot preprocess for prediction."
                    )
                    return {}, None
                self.logger.info("Transforming sequence data with fitted scaler...")
                scaled_value_data = self.sequence_scaler.transform(value_data_to_scale)

            # Reshape back and update final_sequences dictionary
            scaled_value_data = scaled_value_data.reshape(
                num_admissions, max_len, num_sequence_features
            )
            for idx, col in enumerate(self.sequence_feature_names):
                final_sequences[f"{col}_values_scaled"] = scaled_value_data[:, :, idx]
        else:
            self.logger.warning("No sequence features found to scale.")

        # --- Process Static Features ---
        static_features_df = pd.DataFrame(index=data.index)
        if static_cols:
            static_features_df = data[static_cols].copy()
            # Impute missing static features (using median for numeric)
            for col in static_features_df.select_dtypes(include=np.number).columns:
                if static_features_df[col].isnull().any():
                    median_val = static_features_df[col].median()
                    static_features_df[col] = static_features_df[col].fillna(median_val)
                    self.logger.debug(
                        f"Imputed missing static numeric values in '{col}' with median ({median_val})."
                    )
            # Scale static features
            if not for_prediction:
                self.logger.info("Fitting static feature scaler...")
                static_features_scaled = self.static_scaler.fit_transform(
                    static_features_df
                )
            else:
                if not hasattr(self.static_scaler, "mean_"):
                    self.logger.error(
                        "Static feature scaler has not been fitted. Cannot preprocess for prediction."
                    )
                    return {}, None
                self.logger.info("Transforming static features with fitted scaler...")
                static_features_scaled = self.static_scaler.transform(
                    static_features_df
                )

            static_features_df = pd.DataFrame(
                static_features_scaled,
                columns=static_cols,
                index=static_features_df.index,
            )
        else:
            self.logger.warning("No static features selected or found.")
            # Create an empty DataFrame with the correct index if no static features
            static_features_df = pd.DataFrame(index=data.index)

        # --- Prepare Output ---
        # Calculate actual sequence lengths (non-padded length) for each admission
        # Use the first sequence column's lengths as representative
        actual_lengths = (
            all_sequence_lengths[0] if all_sequence_lengths else [0] * num_admissions
        )

        X_processed = {
            "sequences": final_sequences,  # Contains scaled values, timestamps, intervals
            "static_features": static_features_df,
            "sequence_lengths": actual_lengths,
            "admission_ids": data[id_cols],  # Keep original IDs for reference
        }

        self.logger.info("Temporal preprocessing complete.")
        return X_processed, y

    def train(
        self,
        # Widen X_train type hint to match BaseModel for LSP compliance
        X_train: pd.DataFrame | Dict[Any, Any],
        y_train: pd.Series,
        algorithm: Optional[str] = None,  # Algorithm ignored for now
    ) -> nn.Module:
        """
        Train the Time-Aware LSTM model.

        Initializes the LSTM model based on configuration parameters, sets up
        the dataset and dataloader, defines the loss function (BCEWithLogitsLoss)
        and optimizer (Adam), and runs the training loop for the specified number
        of epochs.

        Args:
            X_train (Dict[str, Any]): The preprocessed training data dictionary containing
                                      'sequences', 'static_features', 'sequence_lengths'.
            y_train (pd.Series): The training target variable.
            algorithm (Optional[str], optional): Ignored for this model type. Defaults to None.

        Returns:
            nn.Module: The trained PyTorch TimeAwarePatientLSTM model.
        """
        self.logger.info(
            f"Training TimeAwarePatientLSTM model for {self.model_type}..."
        )

        # --- Model Initialization ---
        # Determine input sizes based on preprocessed data
        num_sequence_features = (
            len(self.sequence_feature_names) if self.sequence_feature_names else 0
        )
        num_static_features = X_train["static_features"].shape[1]
        input_dim = num_sequence_features  # Each time step receives sequence features

        # Get model hyperparameters from config
        hidden_dim = self.model_config.get("hidden_dim", 128)
        num_layers = self.model_config.get("num_layers", 1)
        dropout = self.model_config.get("dropout", 0.1)
        use_time_features = self.model_config.get(
            "use_time_features", True
        )  # Use time intervals by default

        self.model = TimeAwarePatientLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_static_features=num_static_features,
            num_layers=num_layers,
            dropout=dropout,
            use_time_features=use_time_features,
            device=self.device,
        )
        self.model.to(self.device)

        # Store architecture params for saving/loading
        self.model_architecture_params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_static_features": num_static_features,
            "num_layers": num_layers,
            "dropout": dropout,
            "use_time_features": use_time_features,
            "device": str(
                self.device
            ),  # Store device as string for JSON compatibility if needed later
        }

        # --- Dataset and DataLoader ---
        train_dataset = TemporalEHRDataset(
            # Ensure y_train is converted to a numpy array
            X_train,
            y_train.to_numpy(),
        )  # Pass numpy array for target
        batch_size = self.model_config.get("batch_size", 64)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        # --- Loss and Optimizer ---
        # Use BCEWithLogitsLoss for binary classification (more numerically stable)
        criterion = nn.BCEWithLogitsLoss()
        learning_rate = self.model_config.get("learning_rate", 0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # --- Training Loop ---
        num_epochs = self.model_config.get("num_epochs", 10)
        self.logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
            )

            for batch in progress_bar:
                # Move batch to device
                seq_values = batch["sequence_values"].to(self.device)
                seq_intervals = batch["sequence_intervals"].to(self.device)
                static_feats = batch["static_features"].to(self.device)
                lengths = batch["lengths"]  # Keep lengths on CPU for packing
                targets = batch["targets"].to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(seq_values, seq_intervals, static_feats, lengths)

                # Calculate loss (outputs are logits, targets are 0/1)
                loss = criterion(
                    outputs.squeeze(), targets.float()
                )  # Ensure targets are float

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.logger.info(
                f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}"
            )

        self.logger.info("Temporal model training complete.")
        return self.model

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for DataLoader to handle padding and packing.

        Sorts the batch by sequence length (descending), pads sequences to the
        maximum length within the batch, and stacks features and targets into tensors.

        Args:
            batch (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                          represents a sample from TemporalEHRDataset.

        Returns:
            Dict[str, Any]: A dictionary containing batched and padded tensors for
                            'sequence_values', 'sequence_intervals', 'static_features',
                            'targets', and a list of original 'lengths'.
        """
        # Sort batch by sequence length in descending order (required for pack_padded_sequence)
        batch.sort(key=lambda x: x["length"], reverse=True)

        # Separate components
        sequence_values = [item["sequence_values"] for item in batch]
        sequence_intervals = [item["sequence_intervals"] for item in batch]
        static_features = [item["static_features"] for item in batch]
        lengths = [item["length"] for item in batch]
        targets = [item["target"] for item in batch]

        # Pad sequences (assuming they are already numpy arrays or lists)
        # Find max length in this specific batch
        max_len_batch = lengths[0] if lengths else 0
        num_features = sequence_values[0].shape[1] if sequence_values else 0

        # Pad sequence values
        padded_values = torch.zeros((len(batch), max_len_batch, num_features))
        for i, seq in enumerate(sequence_values):
            end = lengths[i]
            if end > 0:  # Avoid indexing empty sequences
                padded_values[i, :end, :] = torch.tensor(seq[:end], dtype=torch.float32)

        # Pad sequence intervals
        padded_intervals = torch.zeros((len(batch), max_len_batch))
        for i, seq in enumerate(sequence_intervals):
            end = lengths[i]
            if end > 0:
                padded_intervals[i, :end] = torch.tensor(seq[:end], dtype=torch.float32)

        # Stack static features and targets
        static_features_tensor = torch.tensor(
            np.array(static_features), dtype=torch.float32
        )
        targets_tensor = torch.tensor(
            targets, dtype=torch.long
        )  # Assuming integer targets for classification

        return {
            "sequence_values": padded_values,
            "sequence_intervals": padded_intervals,
            "static_features": static_features_tensor,
            "lengths": lengths,  # Return lengths as a list of integers
            "targets": targets_tensor,
        }

    # Widen X_test type hint to match BaseModel for LSP compliance
    def evaluate(
        self, X_test: pd.DataFrame | Dict[Any, Any], y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the trained Temporal LSTM model on the test set.

        Sets the model to evaluation mode, iterates through the test data using
        a DataLoader, calculates predictions and probabilities, and computes
        standard classification metrics.

        Args:
            X_test (Dict[str, Any]): The preprocessed test data dictionary.
            y_test (pd.Series): The test target variable.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        self.logger.info(f"Evaluating {self.model_type} model...")
        self.model.eval()  # Set model to evaluation mode

        # Ensure y_test is converted to a numpy array
        test_dataset = TemporalEHRDataset(X_test, y_test.to_numpy())
        batch_size = self.model_config.get(
            "batch_size", 64
        )  # Use same batch size or a different one for eval
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        all_targets = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():  # Disable gradient calculations
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                seq_values = batch["sequence_values"].to(self.device)
                seq_intervals = batch["sequence_intervals"].to(self.device)
                static_feats = batch["static_features"].to(self.device)
                lengths = batch["lengths"]
                targets = batch["targets"].to(self.device)

                outputs = self.model(seq_values, seq_intervals, static_feats, lengths)
                # Apply sigmoid to get probabilities from logits
                probabilities = torch.sigmoid(outputs).squeeze()
                # Get binary predictions (e.g., threshold at 0.5)
                predictions = (probabilities >= 0.5).long()

                # Convert numpy arrays to lists before extending for type compatibility
                all_targets.extend(targets.cpu().numpy().tolist())
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_probabilities.extend(probabilities.cpu().numpy().tolist())

        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(all_targets, all_predictions),
            "precision": precision_score(all_targets, all_predictions, zero_division=0),
            "recall": recall_score(all_targets, all_predictions, zero_division=0),
            "f1": f1_score(all_targets, all_predictions, zero_division=0),
            "roc_auc": roc_auc_score(all_targets, all_probabilities),
            "pr_auc": average_precision_score(all_targets, all_probabilities),
        }

        # Log metrics
        for metric, value in metrics.items():
            self.logger.info(f"{metric.replace('_', ' ').title()}: {value:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_predictions)
        self.logger.info(f"Confusion Matrix:\n{cm}")

        return metrics

    # Specify dtype for returned ndarray (probabilities are floats)
    def predict(self, data: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Make predictions with the trained Temporal LSTM model.

        Handles preprocessing of the input DataFrame, creates a DataLoader,
        runs the model in evaluation mode, and returns the predicted probabilities.

        Args:
            data (pd.DataFrame): Input DataFrame for prediction.

        Returns:
            np.ndarray: NumPy array of predicted probabilities for the positive class.
                        Returns an empty array if preprocessing fails.

        Raises:
            ValueError: If the model or necessary scalers have not been fitted/loaded.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        if self.sequence_scaler is None or self.static_scaler is None:
            raise ValueError("Scalers have not been fitted yet")

        self.logger.info("Preprocessing data for temporal prediction...")
        X_processed, _ = self.preprocess(data, for_prediction=True)

        # Check if preprocessing returned valid data
        if (
            not X_processed
            or not X_processed.get("admission_ids", pd.DataFrame()).shape[0] > 0
        ):
            self.logger.error(
                "Preprocessing failed or resulted in empty data during prediction."
            )
            # Return empty array or raise error, depending on desired behavior
            return np.array([])

        # Create dataset and dataloader for prediction data
        # Need dummy targets for the dataset
        num_samples = len(X_processed["admission_ids"])
        dummy_targets = np.zeros(num_samples)
        predict_dataset = TemporalEHRDataset(X_processed, dummy_targets)
        # Use a batch size suitable for prediction (can be larger if memory allows)
        predict_batch_size = self.model_config.get("predict_batch_size", 128)
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=predict_batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        self.logger.info("Making predictions with temporal model...")
        self.model.eval()
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(predict_loader, desc="Predicting", leave=False):
                seq_values = batch["sequence_values"].to(self.device)
                seq_intervals = batch["sequence_intervals"].to(self.device)
                static_feats = batch["static_features"].to(self.device)
                lengths = batch["lengths"]
                # Targets are ignored here

                outputs = self.model(seq_values, seq_intervals, static_feats, lengths)
                probabilities = torch.sigmoid(outputs).squeeze()
                # Handle case where batch size is 1, squeeze might remove the dimension
                if probabilities.ndim == 0:
                    probabilities = probabilities.unsqueeze(0)
                all_probabilities.extend(probabilities.cpu().numpy())

        return np.array(all_probabilities)

    def save(self, path: str) -> None:
        """
        Save the Temporal LSTM model state_dict, scalers, feature names,
        architecture parameters, and configuration to a file using torch.save.

        Ensures the model is moved to CPU before saving for better compatibility.

        Args:
            path (str): The file path where the model checkpoint will be saved.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        if self.model_architecture_params is None:
            raise ValueError("Model architecture parameters not set. Cannot save.")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Ensure model is on CPU before saving state_dict for better compatibility
        original_device = self.device
        self.model.to("cpu")
        # Update device in saved params to reflect CPU saving state if needed, or keep original
        self.model_architecture_params["device"] = str("cpu")  # Save as CPU state

        model_data = {
            "model_state_dict": self.model.state_dict(),
            "model_architecture_params": self.model_architecture_params,  # Save architecture params
            "feature_names": self.feature_names,  # Combined list
            "sequence_feature_names": self.sequence_feature_names,
            "static_feature_names": self.static_feature_names,
            "sequence_scaler": self.sequence_scaler,
            "static_scaler": self.static_scaler,
            "model_type": self.model_type,
            "target": self.target,
            "config": self.config,
        }

        try:
            torch.save(model_data, path)
            self.logger.info(f"Saved temporal model state and metadata to {path}")
        except Exception as e:
            self.logger.error(
                f"Error saving temporal model to {path}: {e}", exc_info=True
            )
        finally:
            # Move model back to its original device
            self.model.to(original_device)
            # Optionally restore device in params if needed elsewhere, though load handles it
            self.model_architecture_params["device"] = str(original_device)

    @classmethod
    def load(cls, path: str) -> "TemporalReadmissionModel":
        """
        Load a Temporal LSTM model from a saved state_dict checkpoint file.

        Overrides BaseModel.load specifically for temporal models saved via torch.save.
        Reconstructs the model architecture using saved parameters before loading the state dict.

        Args:
            path (str): The file path of the saved model checkpoint.

        Returns:
            TemporalReadmissionModel: The loaded temporal model instance.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            ValueError: If the checkpoint format is invalid, missing required keys,
                        or the model type does not match.
            Exception: Re-raises other unexpected errors during loading.
        """
        logger.info(f"Attempting to load Temporal model from {path}")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Load onto CPU first as recommended practice
            checkpoint = torch.load(path, map_location="cpu")

            if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
                raise ValueError(f"Invalid checkpoint format in {path}")

            config = checkpoint["config"]
            model_type = checkpoint.get("model_type")

            if model_type != "temporal_readmission":
                raise ValueError(
                    f"Model type in file ({model_type}) does not match expected 'temporal_readmission'"
                )

            # Instantiate the model class
            model_instance = cls(
                config=config
            )  # cls refers to TemporalReadmissionModel

            # Recreate model architecture
            arch_params = checkpoint["model_architecture_params"]
            # Ensure device is set correctly for model instantiation
            arch_params["device"] = device
            model_instance.model = TimeAwarePatientLSTM(**arch_params)
            model_instance.model.load_state_dict(checkpoint["model_state_dict"])
            model_instance.model.to(device)  # Move model to the target device
            model_instance.device = device

            # Load metadata
            model_instance.feature_names = checkpoint.get("feature_names")
            model_instance.sequence_feature_names = checkpoint.get(
                "sequence_feature_names"
            )
            model_instance.static_feature_names = checkpoint.get("static_feature_names")
            model_instance.sequence_scaler = checkpoint.get("sequence_scaler")
            model_instance.static_scaler = checkpoint.get("static_scaler")
            model_instance.target = checkpoint.get("target")
            model_instance.model_architecture_params = (
                arch_params  # Store loaded params
            )

            # Basic checks for loaded components
            if model_instance.model is None:
                raise ValueError("Model state_dict loaded but model object is None.")
            if model_instance.sequence_scaler is None:
                logger.warning("Sequence scaler not found in loaded model data.")
            if model_instance.static_scaler is None:
                logger.warning("Static scaler not found in loaded model data.")

            logger.info(
                f"Loaded Temporal model '{model_instance.model_type}' from {path} onto {device}"
            )
            return model_instance

        except FileNotFoundError:
            logger.error(f"Temporal model file not found at {path}")
            raise
        except Exception as e:
            logger.error(
                f"Error loading temporal model from {path}: {e}", exc_info=True
            )
            raise

    # Override get_feature_importance and get_shap_values as they are not directly applicable
    # to LSTMs in the same way as tree/linear models.

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Feature importance is not directly applicable to LSTMs in this way."""
        self.logger.warning(
            "Standard feature importance is not directly applicable to LSTM models."
        )
        return pd.DataFrame(columns=["feature", "importance"])

    def get_shap_values(
        # Match superclass signature for LSP compliance
        self,
        X: pd.DataFrame,
        background_data: Optional[pd.DataFrame] = None,
        # Specify ndarray dtype for SHAP values (floats)
    ) -> Tuple[Optional[np.ndarray[Any, np.dtype[np.float64]]], Optional[Any]]:
        """SHAP values for LSTMs require specialized explainers (e.g., DeepExplainer) and are not implemented here."""
        self.logger.warning(
            "SHAP value calculation for TimeAwarePatientLSTM requires specific handling and is not implemented by default."
        )
        # SHAP for deep models often uses DeepExplainer or GradientExplainer,
        # requiring careful handling of tensor inputs and background data.
        return None, None
