"""
Enhanced Temporal Model with Time-Aware Embeddings

This script implements a time-aware LSTM model for predicting hospital readmissions
using time-series EHR data from the MIMIC dataset.
"""

import os
import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns # Unused
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# Create mock utility functions if imports fail
try:
    # Try to import from src.utils
    from src.utils import get_data_path, get_logger, load_config
except ModuleNotFoundError:
    # Create mock versions of the utility functions
    print("Could not import from src.utils, using mock utility functions")

    def get_logger(name):
        """Mock logger function"""
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(name)

    def load_config():
        """Mock config function"""
        return {"data_paths": {}}

    def get_data_path(dir_type, file_name, config):
        """Mock data path function"""
        return os.path.join(dir_type, file_name + ".csv")


# Configure matplotlib
try:
    # For newer matplotlib versions
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    # Fallback for older versions or if the style is not available
    plt.style.use("default")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# Initialize logger
logger = get_logger("temporal_model_enhanced")


def create_temporal_dataset(data, vital_features, lab_features, seq_length=24):
    """
    Create a temporal dataset from the processed data with explicit time intervals.

    For this POC, we'll simulate temporal data by:
    1. Using the existing features as the final values
    2. Generating synthetic time series leading up to these values
    3. Adding explicit time intervals between measurements

    Args:
        data: DataFrame with processed features
        vital_features: List of vital sign feature names
        lab_features: List of lab value feature names
        seq_length: Number of time steps in each sequence

    Returns:
        X_temporal: Dictionary mapping hadm_id to sequence data
        time_intervals: Dictionary mapping hadm_id to time interval data
        y: Series with readmission labels
    """
    # Extract target and patient IDs
    y = data["readmission_30day"].copy()
    hadm_ids = data["hadm_id"].values

    # Combine vital and lab features
    temporal_features = [
        f
        for f in data.columns
        if any(vf in f for vf in vital_features) or any(lf in f for lf in lab_features)
    ]

    # Create dictionaries to store sequences and time intervals for each admission
    X_temporal = {}
    time_intervals = {}

    # For each admission, create a synthetic time series with time intervals
    for i, hadm_id in enumerate(hadm_ids):
        # Get final values for this admission
        final_values = data.loc[i, temporal_features].values.astype(float)

        # Create a sequence leading up to these values
        sequence = np.zeros((seq_length, len(temporal_features)))

        # Generate time intervals (hours between measurements)
        # For simplicity, we'll use irregular intervals to simulate real-world scenarios
        intervals = np.zeros(seq_length)
        total_hours = 0

        for j in range(seq_length):
            # Generate random interval (1-6 hours between measurements)
            if j > 0:
                interval = np.random.uniform(1, 6)
                total_hours += interval
                intervals[j] = total_hours
            else:
                intervals[j] = 0  # First measurement at time 0

        for j, final_val in enumerate(final_values):
            # Start with a value in the healthy range
            start_val = final_val * 0.8 + np.random.normal(0, 0.1)

            # Generate a trajectory from start to final value
            # Use non-linear progression to make it more realistic
            progress = np.linspace(0, 1, seq_length) ** 1.5  # Non-linear progression
            trajectory = start_val + (final_val - start_val) * progress

            # Add some noise to make it realistic
            noise = np.random.normal(0, abs(final_val) * 0.05, seq_length)
            trajectory += noise

            # Store in sequence
            sequence[:, j] = trajectory

        # Store sequence and time intervals for this admission
        X_temporal[hadm_id] = sequence
        time_intervals[hadm_id] = intervals.reshape(-1, 1)  # Reshape for model input

    return X_temporal, time_intervals, y


class TemporalEHRDataset(Dataset):
    def __init__(self, sequences, time_intervals, labels, hadm_ids):
        self.sequences = sequences
        self.time_intervals = time_intervals
        self.labels = labels
        self.hadm_ids = hadm_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        hadm_id = self.hadm_ids[idx]
        sequence = self.sequences[hadm_id]
        interval = self.time_intervals[hadm_id]
        label = self.labels[idx]
        return (
            torch.FloatTensor(sequence),
            torch.FloatTensor(interval),
            torch.FloatTensor([label]),
        )


class TimeEncoder(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_embed = nn.Linear(1, embed_dim)

    def forward(self, time_intervals):
        # time_intervals shape: [batch_size, seq_len, 1]
        # Apply sinusoidal encoding for better representation of time
        time_encoding = self.time_embed(time_intervals)
        return time_encoding


class TimeAwarePatientLSTM(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, time_embed_dim=16, num_layers=2, dropout=0.2
    ):
        super().__init__()
        self.time_encoder = TimeEncoder(time_embed_dim)

        # LSTM takes concatenated feature and time embeddings
        self.lstm = nn.LSTM(
            input_dim + time_embed_dim,  # Concatenated input
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, time_intervals):
        # Encode time intervals
        time_encoding = self.time_encoder(time_intervals)

        # Concatenate features with time encoding
        x_with_time = torch.cat([x, time_encoding], dim=2)

        # Process sequence with LSTM
        lstm_out, _ = self.lstm(
            x_with_time
        )  # lstm_out shape: [batch_size, seq_len, hidden_dim]

        # Apply attention to focus on important time steps
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # Weighted sum

        # Classify
        return torch.sigmoid(self.classifier(context))


def get_attention_weights(model, sequence, time_interval, device):
    model.eval()
    with torch.no_grad():
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        interval_tensor = torch.FloatTensor(time_interval).unsqueeze(0).to(device)

        # Get time encoding
        time_encoding = model.time_encoder(interval_tensor)

        # Concatenate features with time encoding
        x_with_time = torch.cat([sequence_tensor, time_encoding], dim=2)

        # Get LSTM outputs
        lstm_out, _ = model.lstm(x_with_time)

        # Get attention weights
        attention_weights = torch.softmax(model.attention(lstm_out), dim=1)

    return attention_weights.cpu().numpy().squeeze()


def train_model(
    X_temporal,
    time_intervals,
    y,
    train_ids,
    test_ids,
    train_labels,
    test_labels,
    input_dim,
    hidden_dim=64,
    time_embed_dim=16,
    num_epochs=15,
    batch_size=32,
):
    # Create datasets
    train_dataset = TemporalEHRDataset(
        X_temporal, time_intervals, train_labels, train_ids
    )
    test_dataset = TemporalEHRDataset(X_temporal, time_intervals, test_labels, test_ids)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = TimeAwarePatientLSTM(input_dim, hidden_dim, time_embed_dim)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Lists to store metrics
    train_losses = []
    test_losses = []
    test_aucs = []
    test_pr_aucs = []  # Added PR AUC tracking

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for sequences, intervals, labels in train_loader:
            sequences, intervals, labels = (
                sequences.to(device),
                intervals.to(device),
                labels.to(device),
            )

            # Forward pass
            outputs = model(sequences, intervals)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sequences.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluation
        model.eval()
        test_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for sequences, intervals, labels in test_loader:
                sequences, intervals, labels = (
                    sequences.to(device),
                    intervals.to(device),
                    labels.to(device),
                )

                # Forward pass
                outputs = model(sequences, intervals)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * sequences.size(0)

                # Store predictions and labels
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        # Calculate metrics
        all_labels = np.array(all_labels).flatten()
        all_preds = np.array(all_preds).flatten()
        test_auc = roc_auc_score(all_labels, all_preds)
        test_pr_auc = average_precision_score(all_labels, all_preds)
        test_aucs.append(test_auc)
        test_pr_aucs.append(test_pr_auc)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, "
            f"Test ROC AUC: {test_auc:.4f}, "
            f"Test PR AUC: {test_pr_auc:.4f}"
        )

    return (
        model,
        train_losses,
        test_losses,
        test_aucs,
        test_pr_aucs,
        all_labels,
        all_preds,
        device,
    )


def train_baseline_model(data, train_ids, test_ids):
    # Extract features (excluding identifiers and targets)
    feature_cols = [
        col
        for col in data.columns
        if col
        not in [
            "subject_id",
            "hadm_id",
            "stay_id",
            "readmission_30day",
            "readmission_90day",
            "los_days",
            "hospital_death",
        ]
    ]

    X = data[feature_cols]
    y_all = data["readmission_30day"]

    # Get the same train/test indices used for the LSTM
    train_indices = [
        i for i, hadm_id in enumerate(data["hadm_id"]) if hadm_id in train_ids
    ]
    test_indices = [
        i for i, hadm_id in enumerate(data["hadm_id"]) if hadm_id in test_ids
    ]

    # Split data using these indices
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y_all.iloc[train_indices]
    y_test = y_all.iloc[test_indices]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # Train LightGBM model
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        random_state=42,
    )

    lgb_model.fit(X_train_scaled, y_train)

    # Make predictions
    lgb_preds_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
    lgb_preds = lgb_model.predict(X_test_scaled)

    # Calculate metrics
    lgb_roc_auc = roc_auc_score(y_test, lgb_preds_proba)
    lgb_pr_auc = average_precision_score(y_test, lgb_preds_proba)

    print(f"LightGBM ROC AUC: {lgb_roc_auc:.4f}")
    print(f"LightGBM PR AUC: {lgb_pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, lgb_preds))

    return lgb_model, lgb_roc_auc, lgb_pr_auc, y_test, lgb_preds, lgb_preds_proba


def visualize_results(
    train_losses,
    test_losses,
    test_aucs,
    test_pr_aucs,
    all_labels,
    all_preds,
    y_test,
    lgb_preds_proba,
    lgb_roc_auc,
    lgb_pr_auc,
    num_epochs,
):
    # Convert LSTM predictions to binary for classification report
    lstm_preds_binary = (all_preds > 0.5).astype(int)

    print("Time-Aware LSTM Classification Report:")
    print(classification_report(all_labels, lstm_preds_binary))

    # Plot training curves
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, "b-", label="Train Loss")
    plt.plot(range(1, num_epochs + 1), test_losses, "r-", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs + 1), test_aucs, "g-", label="LSTM")
    plt.axhline(y=lgb_roc_auc, color="r", linestyle="--", label="LightGBM")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC Comparison")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_epochs + 1), test_pr_aucs, "g-", label="LSTM")
    plt.axhline(y=lgb_pr_auc, color="r", linestyle="--", label="LightGBM")
    plt.xlabel("Epoch")
    plt.ylabel("PR AUC")
    plt.title("PR AUC Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/time_aware_lstm_training_curves.png")
    plt.show()

    # Plot ROC curves
    plt.figure(figsize=(12, 10))

    # ROC Curve
    plt.subplot(2, 1, 1)
    fpr_lstm, tpr_lstm, _ = roc_curve(all_labels, all_preds)
    fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_preds_proba)
    plt.plot(fpr_lstm, tpr_lstm, label=f"LSTM (AUC = {test_aucs[-1]:.3f})")
    plt.plot(fpr_lgb, tpr_lgb, label=f"LightGBM (AUC = {lgb_roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()

    # Precision-Recall Curve
    plt.subplot(2, 1, 2)
    precision_lstm, recall_lstm, _ = precision_recall_curve(all_labels, all_preds)
    precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, lgb_preds_proba)
    plt.plot(
        recall_lstm, precision_lstm, label=f"LSTM (PR AUC = {test_pr_aucs[-1]:.3f})"
    )
    plt.plot(recall_lgb, precision_lgb, label=f"LightGBM (PR AUC = {lgb_pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/time_aware_lstm_roc_pr_curves.png")
    plt.show()


def analyze_attention(model, test_loader, device, feature_names, num_samples=5):
    model.eval()
    attention_weights_list = []
    sequences_list = []
    intervals_list = []
    labels_list = []
    count = 0

    with torch.no_grad():
        for sequences, intervals, labels in test_loader:
            if count >= num_samples:
                break
            sequences, intervals = sequences.to(device), intervals.to(device)

            # Get attention weights
            weights = get_attention_weights(model, sequences[0], intervals[0], device)
            attention_weights_list.append(weights)
            sequences_list.append(sequences[0].cpu().numpy())
            intervals_list.append(intervals[0].cpu().numpy().flatten())
            labels_list.append(labels[0].item())
            count += 1

    # Visualize attention for a few samples
    for i in range(num_samples):
        plt.figure(figsize=(15, 5))
        plt.plot(intervals_list[i], attention_weights_list[i], marker="o")
        plt.xlabel("Time from Admission (Hours)")
        plt.ylabel("Attention Weight")
        plt.title(
            f"Attention Weights over Time (Sample {i+1}, Label: {labels_list[i]})"
        )
        plt.grid(True)
        plt.savefig(f"results/attention_sample_{i+1}.png")
        plt.show()

        # Optional: Show feature values alongside attention
        # fig, ax1 = plt.subplots(figsize=(15, 7))
        # ax2 = ax1.twinx()
        # ax1.plot(intervals_list[i], attention_weights_list[i], 'g-', marker='o', label='Attention')
        # # Plot a few key features
        # for feat_idx, feat_name in enumerate(feature_names[:3]): # Example: first 3 features
        #     ax2.plot(intervals_list[i], sequences_list[i][:, feat_idx], '--', label=feat_name)
        # ax1.set_xlabel("Time from Admission (Hours)")
        # ax1.set_ylabel("Attention Weight", color='g')
        # ax2.set_ylabel("Feature Value")
        # plt.title(f"Attention & Features (Sample {i+1}, Label: {labels_list[i]})")
        # fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        # plt.grid(True)
        # plt.show()


def main():
    # Load configuration
    config = load_config()

    # Load processed data
    data_path = get_data_path("processed", "combined_features", config)
    data = pd.read_csv(data_path)

    # Define feature sets (example, adjust based on actual columns)
    vital_features = [
        col
        for col in data.columns
        if any(vf in col for vf in ["heart", "bp", "resp", "temp", "spo2", "gcs"])
    ]
    lab_features = [
        col
        for col in data.columns
        if any(
            lf in col
            for lf in [
                "glucose",
                "potassium",
                "sodium",
                "bun",
                "creatinine",
                "wbc",
                "hgb",
                "hct",
                "platelet",
            ]
        )
    ]

    # Create temporal dataset
    X_temporal, time_intervals, y = create_temporal_dataset(
        data, vital_features, lab_features, seq_length=24
    )

    # Split data into train and test based on hadm_id
    hadm_ids = data["hadm_id"].unique()
    train_ids, test_ids = train_test_split(hadm_ids, test_size=0.2, random_state=42)

    # Get corresponding labels for train/test sets
    train_labels = y[data["hadm_id"].isin(train_ids)].values
    test_labels = y[data["hadm_id"].isin(test_ids)].values

    # Train Time-Aware LSTM
    input_dim = list(X_temporal.values())[0].shape[1]  # Get feature dimension
    (
        lstm_model,
        train_losses,
        test_losses,
        test_aucs,
        test_pr_aucs,
        all_labels,
        all_preds,
        device,
    ) = train_model(
        X_temporal,
        time_intervals,
        y,
        train_ids,
        test_ids,
        train_labels,
        test_labels,
        input_dim,
    )

    # Train Baseline LightGBM
    (
        lgb_model,
        lgb_roc_auc,
        lgb_pr_auc,
        y_test_lgb,
        lgb_preds,
        lgb_preds_proba,
    ) = train_baseline_model(data, train_ids, test_ids)

    # Visualize Results
    visualize_results(
        train_losses,
        test_losses,
        test_aucs,
        test_pr_aucs,
        all_labels,
        all_preds,
        y_test_lgb,
        lgb_preds_proba,
        lgb_roc_auc,
        lgb_pr_auc,
        num_epochs=15,  # Match training epochs
    )

    # Analyze Attention (Optional)
    # Recreate test loader for attention analysis
    test_dataset = TemporalEHRDataset(X_temporal, time_intervals, test_labels, test_ids)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Batch size 1
    feature_names_temporal = [
        f
        for f in data.columns
        if any(vf in f for vf in vital_features) or any(lf in f for lf in lab_features)
    ]
    analyze_attention(lstm_model, test_loader, device, feature_names_temporal)


if __name__ == "__main__":
    main()
