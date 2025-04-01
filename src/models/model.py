"""
Model classes for the MIMIC project.
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import shap

from src.utils import get_logger, load_config, get_project_root


logger = get_logger(__name__)


def _reshape_data_to_admission_level(data: pd.DataFrame, id_cols: List[str], target_cols: List[str], logger_instance) -> pd.DataFrame:
    """
    Reshapes data from potentially multiple rows per admission to one row per admission.
    Handles pivoting based on feature structure.
    """
    logger_instance.info("Detected multiple rows per admission. Restructuring data...")

    # Keep only columns that exist in the dataframe
    existing_id_cols = [col for col in id_cols if col in data.columns]
    existing_target_cols = [col for col in target_cols if col in data.columns]

    # Extract identifiers and targets
    identifiers = data[existing_id_cols].drop_duplicates()

    # For each target column, ensure we have only one value per admission
    targets = {}
    for col in existing_target_cols:
        # Group by hadm_id and take the first value (they should all be the same)
        targets[col] = data.groupby('hadm_id')[col].first()

    # Combine targets into a dataframe
    targets_df = pd.DataFrame(targets)
    targets_df.index.name = 'hadm_id'
    targets_df = targets_df.reset_index()

    # Merge identifiers with targets
    admission_data = pd.merge(identifiers, targets_df, on='hadm_id', how='left')

    # Identify feature columns (excluding identifiers and targets)
    feature_cols = [col for col in data.columns
                   if col not in existing_id_cols + existing_target_cols]

    # Check if we have category and level_0 columns that indicate the feature type
    if 'category' in data.columns and 'level_0' in data.columns:
        logger_instance.info("Pivoting data based on category and level_0 columns...")

        # Create a unique identifier for each feature
        data['feature_id'] = data['category'] + '_' + data['level_0'].astype(str)

        # Pivot the data to wide format
        # Each row will be a unique hadm_id, and columns will be features
        pivoted_data = data.pivot_table(
            index='hadm_id',
            columns='feature_id',
            values='0',  # Assuming the value column is named '0'
            aggfunc='first'  # Take the first value if there are duplicates
        )

        # Reset index to get hadm_id as a column
        pivoted_data = pivoted_data.reset_index()

        # Merge with admission data to get all identifiers and targets
        restructured_data = pd.merge(
            admission_data,
            pivoted_data,
            on='hadm_id',
            how='left'
        )
    else:
        logger_instance.info("Pivoting data based on all non-identifier columns...")

        # For each admission, create a single row with all features
        restructured_data = pd.DataFrame()

        # Group by hadm_id
        for hadm_id, group in data.groupby('hadm_id'):
            # Start with the admission data for this hadm_id
            row = admission_data[admission_data['hadm_id'] == hadm_id].iloc[0].to_dict()

            # Add all feature columns
            for _, feature_row in group.iterrows():
                for col in feature_cols:
                    if col in feature_row:
                        # Create a unique column name if needed
                        col_name = col
                        if col in row and row[col] != feature_row[col]:
                            # Simple way to avoid collision, might need refinement
                            col_name = f"{col}_{len([c for c in row.keys() if c.startswith(col)])}"
                        row[col_name] = feature_row[col]

            # Add this row to the restructured data
            restructured_data = pd.concat([
                restructured_data,
                pd.DataFrame([row])
            ], ignore_index=True)

    logger_instance.info(f"Data restructured: {len(restructured_data)} unique admissions")
    return restructured_data

class BaseModel(ABC):
    """
    Base class for models.
    """
    
    def __init__(
        self, 
        model_type: str,
        config: Optional[Dict] = None,
        random_state: int = 42
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
        self.scaler = StandardScaler()
        
        # Set model-specific configuration
        self.model_config = self.config["models"][model_type]
        self.target = self.model_config["target"]
        self.algorithms = self.model_config["algorithms"]
        self.cv_folds = self.model_config["cv_folds"]
        self.hyperparameter_tuning = self.model_config["hyperparameter_tuning"]
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data for modelling.
        
        Args:
            data (pd.DataFrame): Input data
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Preprocessed features and target
        """
        pass
    
    @abstractmethod
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        algorithm: str = None
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
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
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
        self, 
        data: pd.DataFrame, 
        algorithm: str = None,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Fit the model to the data.
        
        Args:
            data (pd.DataFrame): Input data
            algorithm (str, optional): Algorithm to use. If None, uses the first
                algorithm in the configuration. Defaults to None.
            test_size (float, optional): Proportion of data to use for testing.
                Defaults to 0.2.
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Preprocess data
        X, y = self.preprocess(data)
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Train model
        self.model = self.train(X_train_scaled, y_train, algorithm)
        
        # Evaluate model
        metrics = self.evaluate(X_test_scaled, y_test)
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            data (pd.DataFrame): Input data
        
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess data
        X, _ = self.preprocess(data, for_prediction=True)
        
        # Ensure all feature columns are present
        missing_cols = set(self.feature_names) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        
        # Ensure columns are in the same order as during training
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model to
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
            "config": self.config
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Load a model from disk.
        
        Args:
            path (str): Path to load the model from
        
        Returns:
            BaseModel: Loaded model
        """
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        # Create model instance
        model_type = model_data["model_type"]
        config = model_data["config"]
        
        if model_type == "readmission":
            model_instance = ReadmissionModel(config=config)
        elif model_type == "mortality":
            model_instance = MortalityModel(config=config)
        elif model_type == "los":
            model_instance = LengthOfStayModel(config=config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load model and metadata
        model_instance.model = model_data["model"]
        model_instance.feature_names = model_data["feature_names"]
        model_instance.scaler = model_data["scaler"]
        
        return model_instance
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            top_n (int, optional): Number of top features to return. Defaults to 20.
        
        Returns:
            pd.DataFrame: Feature importance
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Get feature importance based on model type
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0]) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
        else:
            self.logger.warning("Model does not have feature importance")
            return pd.DataFrame()
        
        # Create dataframe
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)
        
        # Return top N features
        return importance_df.head(top_n)
    
    def get_shap_values(
        self, 
        data: pd.DataFrame, 
        sample_size: int = 100
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get SHAP values for the model.
        
        Args:
            data (pd.DataFrame): Input data
            sample_size (int, optional): Number of samples to use for SHAP values.
                Defaults to 100.
        
        Returns:
            Tuple[np.ndarray, List[str]]: SHAP values and feature names
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess data
        X, _ = self.preprocess(data, for_prediction=True)
        
        # Ensure all feature columns are present
        missing_cols = set(self.feature_names) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        
        # Ensure columns are in the same order as during training
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Sample data if needed
        if len(X_scaled) > sample_size:
            X_sample = X_scaled.sample(sample_size, random_state=self.random_state)
        else:
            X_sample = X_scaled
        
        # Create explainer
        if isinstance(self.model, (xgb.XGBModel, lgb.LGBMModel)):
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.KernelExplainer(self.model.predict, X_sample)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # For multi-class models, return values for the positive class
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        
        return shap_values, self.feature_names


class ReadmissionModel(BaseModel):
    """
    Model for predicting hospital readmission.
    """
    
    def __init__(
        self, 
        config: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize the readmission model.
        
        Args:
            config (Optional[Dict], optional): Configuration dictionary.
                If None, loads the default configuration. Defaults to None.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        super().__init__("readmission", config, random_state)
    
    def preprocess(
        self,
        data: pd.DataFrame,
        for_prediction: bool = False
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
        # Copy data to avoid modifying the original
        data = data.copy()
        
        # Check if we have multiple rows per admission (hadm_id)
        if 'hadm_id' in data.columns and data['hadm_id'].duplicated().any():
            id_cols = ['subject_id', 'hadm_id', 'stay_id']
            target_cols = ['readmission_30day', 'readmission_90day', 'los_days', 'hospital_death']
            data = _reshape_data_to_admission_level(data, id_cols, target_cols, self.logger)
        
        # Extract target if not for prediction
        if not for_prediction:
            y = data[self.target]
        else:
            y = None
        
        # Drop target and identifier columns
        X = data.drop(
            columns=[
                "subject_id", "hadm_id", "stay_id",
                "readmission_30day", "readmission_90day",
                "los_days", "hospital_death"
            ],
            errors="ignore"
        )
        
        # Handle categorical columns - drop any string columns that haven't been one-hot encoded
        # This includes 'gender', 'age_group', etc. which should already have one-hot encoded versions
        string_columns = X.select_dtypes(include=['object']).columns.tolist()
        X = X.drop(columns=string_columns, errors='ignore')
        
        # Handle NaN values - impute with median for numeric columns
        self.logger.info("Imputing missing values...")
        numeric_cols = X.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):  # If median is also NaN, use 0
                    median_val = 0
                X[col] = X[col].fillna(median_val)
                self.logger.debug(f"Imputed column {col} with median: {median_val}")
        
        # Check if there are still NaN values
        if X.isna().any().any():
            self.logger.warning("There are still NaN values in the data after imputation")
            # Drop columns with all NaN values
            cols_to_drop = X.columns[X.isna().all()].tolist()
            if cols_to_drop:
                self.logger.warning(f"Dropping columns with all NaN values: {cols_to_drop}")
                X = X.drop(columns=cols_to_drop)
            
            # For remaining NaN values, fill with 0
            X = X.fillna(0)
            self.logger.info("Filled remaining NaN values with 0")
        
        # Log the shape of the preprocessed data
        self.logger.info(f"Preprocessed data shape: X={X.shape}, y={len(y) if y is not None else 'None'}")
        
        return X, y
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        algorithm: str = None
    ) -> BaseEstimator:
        """
        Train the readmission model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            algorithm (str, optional): Algorithm to use. If None, uses the first
                algorithm in the configuration. Defaults to None.
        
        Returns:
            BaseEstimator: Trained model
        """
        # Use first algorithm if none specified
        if algorithm is None:
            algorithm = self.algorithms[0]
        
        # Check if algorithm is supported
        if algorithm not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm} not supported")
        
        # Initialize model based on algorithm
        if algorithm == "logistic_regression":
            model = LogisticRegression(
                random_state=self.random_state,
                class_weight=self.model_config.get("class_weight", None),
                max_iter=1000
            )
        elif algorithm == "random_forest":
            model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight=self.model_config.get("class_weight", None),
                n_estimators=100
            )
        elif algorithm == "xgboost":
            model = xgb.XGBClassifier(
                random_state=self.random_state,
                scale_pos_weight=1 if self.model_config.get("class_weight", None) == "balanced" else None,
                n_estimators=100
            )
        elif algorithm == "lightgbm":
            model = lgb.LGBMClassifier(
                random_state=self.random_state,
                class_weight=self.model_config.get("class_weight", None),
                n_estimators=100
            )
        else:
            raise ValueError(f"Algorithm {algorithm} not implemented")
        
        # Hyperparameter tuning if enabled
        if self.hyperparameter_tuning:
            self.logger.info(f"Performing hyperparameter tuning for {algorithm}")
            model = self._tune_hyperparameters(model, X_train, y_train, algorithm)
        
        # Train model
        self.logger.info(f"Training {algorithm} model for readmission prediction")
        model.fit(X_train, y_train)
        
        return model
    
    def _tune_hyperparameters(
        self, 
        model: BaseEstimator, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        algorithm: str
    ) -> BaseEstimator:
        """
        Tune hyperparameters for the model.
        
        Args:
            model (BaseEstimator): Model to tune
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            algorithm (str): Algorithm name
        
        Returns:
            BaseEstimator: Tuned model
        """
        # Define parameter grid based on algorithm
        if algorithm == "logistic_regression":
            param_grid = {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2", "elasticnet"],
                "solver": ["saga"]
            }
        elif algorithm == "random_forest":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        elif algorithm == "xgboost":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            }
        elif algorithm == "lightgbm":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            }
        else:
            self.logger.warning(f"No hyperparameter grid defined for {algorithm}")
            return model
        
        # Create grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Log best parameters
        self.logger.info(f"Best parameters for {algorithm}: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the readmission model.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["precision"] = precision_score(y_test, y_pred)
        metrics["recall"] = recall_score(y_test, y_pred)
        metrics["f1"] = f1_score(y_test, y_pred)
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
        metrics["pr_auc"] = average_precision_score(y_test, y_pred_proba)
        
        # Log metrics
        self.logger.info(f"Readmission model evaluation metrics: {metrics}")
        
        return metrics


class MortalityModel(BaseModel):
    """
    Model for predicting ICU mortality.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize the mortality model.
        
        Args:
            config (Optional[Dict], optional): Configuration dictionary.
                If None, loads the default configuration. Defaults to None.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        super().__init__("mortality", config, random_state)
        # Override the target to use hospital_death instead of icu_mortality
        self.target = "hospital_death"
    
    def preprocess(
        self,
        data: pd.DataFrame,
        for_prediction: bool = False
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
        # Copy data to avoid modifying the original
        data = data.copy()
        
        # Check if we have multiple rows per admission (hadm_id)
        if 'hadm_id' in data.columns and data['hadm_id'].duplicated().any():
            id_cols = ['subject_id', 'hadm_id', 'stay_id']
            target_cols = ['readmission_30day', 'readmission_90day', 'los_days', 'hospital_death']
            data = _reshape_data_to_admission_level(data, id_cols, target_cols, self.logger)
        
        # Extract target if not for prediction
        if not for_prediction:
            y = data[self.target]
        else:
            y = None
        
        # Drop target and identifier columns
        X = data.drop(
            columns=[
                "subject_id", "hadm_id", "stay_id",
                "readmission_30day", "readmission_90day",
                "los_days", "hospital_death"
            ],
            errors="ignore"
        )
        
        # Handle categorical columns - drop any string columns that haven't been one-hot encoded
        # This includes 'gender', 'age_group', etc. which should already have one-hot encoded versions
        string_columns = X.select_dtypes(include=['object']).columns.tolist()
        X = X.drop(columns=string_columns, errors='ignore')
        
        # Handle NaN values - impute with median for numeric columns
        self.logger.info("Imputing missing values...")
        numeric_cols = X.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):  # If median is also NaN, use 0
                    median_val = 0
                X[col] = X[col].fillna(median_val)
                self.logger.debug(f"Imputed column {col} with median: {median_val}")
        
        # Check if there are still NaN values
        if X.isna().any().any():
            self.logger.warning("There are still NaN values in the data after imputation")
            # Drop columns with all NaN values
            cols_to_drop = X.columns[X.isna().all()].tolist()
            if cols_to_drop:
                self.logger.warning(f"Dropping columns with all NaN values: {cols_to_drop}")
                X = X.drop(columns=cols_to_drop)
            
            # For remaining NaN values, fill with 0
            X = X.fillna(0)
            self.logger.info("Filled remaining NaN values with 0")
        
        # Log the shape of the preprocessed data
        self.logger.info(f"Preprocessed data shape: X={X.shape}, y={len(y) if y is not None else 'None'}")
        
        return X, y
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        algorithm: str = None
    ) -> BaseEstimator:
        """
        Train the mortality model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            algorithm (str, optional): Algorithm to use. If None, uses the first
                algorithm in the configuration. Defaults to None.
        
        Returns:
            BaseEstimator: Trained model
        """
        # Use first algorithm if none specified
        if algorithm is None:
            algorithm = self.algorithms[0]
        
        # Check if algorithm is supported
        if algorithm not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm} not supported")
        
        # Initialize model based on algorithm
        if algorithm == "logistic_regression":
            model = LogisticRegression(
                random_state=self.random_state,
                class_weight=self.model_config.get("class_weight", None),
                max_iter=1000
            )
        elif algorithm == "random_forest":
            model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight=self.model_config.get("class_weight", None),
                n_estimators=100
            )
        elif algorithm == "xgboost":
            model = xgb.XGBClassifier(
                random_state=self.random_state,
                scale_pos_weight=1 if self.model_config.get("class_weight", None) == "balanced" else None,
                n_estimators=100
            )
        elif algorithm == "lightgbm":
            model = lgb.LGBMClassifier(
                random_state=self.random_state,
                class_weight=self.model_config.get("class_weight", None),
                n_estimators=100
            )
        else:
            raise ValueError(f"Algorithm {algorithm} not implemented")
        
        # Hyperparameter tuning if enabled
        if self.hyperparameter_tuning:
            self.logger.info(f"Performing hyperparameter tuning for {algorithm}")
            model = self._tune_hyperparameters(model, X_train, y_train, algorithm)
        
        # Train model
        self.logger.info(f"Training {algorithm} model for mortality prediction")
        model.fit(X_train, y_train)
        
        return model
    
    def _tune_hyperparameters(
        self, 
        model: BaseEstimator, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        algorithm: str
    ) -> BaseEstimator:
        """
        Tune hyperparameters for the model.
        
        Args:
            model (BaseEstimator): Model to tune
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            algorithm (str): Algorithm name
        
        Returns:
            BaseEstimator: Tuned model
        """
        if algorithm == "logistic_regression":
            param_grid = {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2", "elasticnet"],
                "solver": ["saga"]
            }
        elif algorithm == "random_forest":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        elif algorithm == "xgboost":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            }
        elif algorithm == "lightgbm":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            }
        else:
            self.logger.warning(f"No hyperparameter grid defined for {algorithm}")
            return model
        
        # Create grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Log best parameters
        self.logger.info(f"Best parameters for {algorithm}: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the mortality model.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["precision"] = precision_score(y_test, y_pred)
        metrics["recall"] = recall_score(y_test, y_pred)
        metrics["f1"] = f1_score(y_test, y_pred)
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
        metrics["pr_auc"] = average_precision_score(y_test, y_pred_proba)
        
        # Log metrics
        self.logger.info(f"Mortality model evaluation metrics: {metrics}")
        
        return metrics


class LengthOfStayModel(BaseModel):
    """
    Model for predicting length of stay.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize the length of stay model.
        
        Args:
            config (Optional[Dict], optional): Configuration dictionary.
                If None, loads the default configuration. Defaults to None.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        super().__init__("los", config, random_state)
        # Override the target to use los_days instead of length_of_stay
        self.target = "los_days"
    
    def preprocess(
        self, 
        data: pd.DataFrame, 
        for_prediction: bool = False
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
        # Copy data to avoid modifying the original
        data = data.copy()
        
        # Check if we have multiple rows per admission (hadm_id)
        if 'hadm_id' in data.columns and data['hadm_id'].duplicated().any():
            id_cols = ['subject_id', 'hadm_id', 'stay_id']
            target_cols = ['readmission_30day', 'readmission_90day', 'los_days', 'hospital_death']
            data = _reshape_data_to_admission_level(data, id_cols, target_cols, self.logger)
        
        # Extract target if not for prediction
        if not for_prediction:
            y = data[self.target]
        else:
            y = None
        
        # Drop target and identifier columns
        X = data.drop(
            columns=[
                "subject_id", "hadm_id", "stay_id",
                "readmission_30day", "readmission_90day",
                "los_days", "hospital_death"
            ],
            errors="ignore"
        )
        
        # Handle categorical columns - drop any string columns that haven't been one-hot encoded
        # This includes 'gender', 'age_group', etc. which should already have one-hot encoded versions
        string_columns = X.select_dtypes(include=['object']).columns.tolist()
        X = X.drop(columns=string_columns, errors='ignore')
        
        # Handle NaN values - impute with median for numeric columns
        self.logger.info("Imputing missing values...")
        numeric_cols = X.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):  # If median is also NaN, use 0
                    median_val = 0
                X[col] = X[col].fillna(median_val)
                self.logger.debug(f"Imputed column {col} with median: {median_val}")
        
        # Check if there are still NaN values
        if X.isna().any().any():
            self.logger.warning("There are still NaN values in the data after imputation")
            # Drop columns with all NaN values
            cols_to_drop = X.columns[X.isna().all()].tolist()
            if cols_to_drop:
                self.logger.warning(f"Dropping columns with all NaN values: {cols_to_drop}")
                X = X.drop(columns=cols_to_drop)
            
            # For remaining NaN values, fill with 0
            X = X.fillna(0)
            self.logger.info("Filled remaining NaN values with 0")
        
        # Log the shape of the preprocessed data
        self.logger.info(f"Preprocessed data shape: X={X.shape}, y={len(y) if y is not None else 'None'}")
        
        return X, y
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        algorithm: str = None
    ) -> BaseEstimator:
        """
        Train the length of stay model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            algorithm (str, optional): Algorithm to use. If None, uses the first
                algorithm in the configuration. Defaults to None.
        
        Returns:
            BaseEstimator: Trained model
        """
        # Use first algorithm if none specified
        if algorithm is None:
            algorithm = self.algorithms[0]
        
        # Check if algorithm is supported
        if algorithm not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm} not supported")
        
        # Initialize model based on algorithm
        if algorithm == "linear_regression":
            model = LinearRegression()
        elif algorithm == "random_forest":
            model = RandomForestRegressor(
                random_state=self.random_state,
                n_estimators=100
            )
        elif algorithm == "xgboost":
            model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_estimators=100
            )
        # Define parameter grid based on algorithm
        if algorithm == "logistic_regression":
            param_grid = {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2", "elasticnet"],
                "solver": ["saga"]
            }
        elif algorithm == "random_forest":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        elif algorithm == "xgboost":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            }
        elif algorithm == "lightgbm":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            }
        else:
            self.logger.warning(f"No hyperparameter grid defined for {algorithm}")
            return model
        
        # Create grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Log best parameters
        self.logger.info(f"Best parameters for {algorithm}: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the mortality model.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["precision"] = precision_score(y_test, y_pred)
        metrics["recall"] = recall_score(y_test, y_pred)
        metrics["f1"] = f1_score(y_test, y_pred)
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
        metrics["pr_auc"] = average_precision_score(y_test, y_pred_proba)
        
        # Log metrics
        self.logger.info(f"Mortality model evaluation metrics: {metrics}")
        
        return metrics


# (Duplicate LengthOfStayModel class removed)