"""
Core PyTorch components for temporal modeling (e.g., Time-Aware LSTM).
"""

from typing import Dict, List, Optional, Tuple  # Import necessary types

import numpy as np  # Import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TemporalEHRDataset(Dataset):
    """
    PyTorch Dataset for handling temporal EHR sequences.
    Assumes input data structures are dictionaries keyed by hadm_id,
    where sequences and intervals are numpy arrays (already padded).
    """

    def __init__(
        self,
        sequences: Dict[str, np.ndarray],
        time_intervals: Dict[str, np.ndarray],
        static_features: Dict[str, np.ndarray],
        labels: Dict[str, int],
        hadm_ids: List[str],
    ):
        """
        Initializes the dataset.

        Args:
            sequences (Dict[str, np.ndarray]): Dict mapping hadm_id to sequence array [max_len, num_seq_feats].
            time_intervals (Dict[str, np.ndarray]): Dict mapping hadm_id to interval array [max_len, 1].
            static_features (Dict[str, np.ndarray]): Dict mapping hadm_id to static feature array [num_static_feats].
            labels (Dict[str, int]): Dict mapping hadm_id to label (0 or 1).
            hadm_ids (List[str]): List of hadm_ids to include in the dataset.
        """
        # Store only the hadm_ids relevant for this dataset split (train/val/test)
        self.hadm_ids = hadm_ids
        self.sequences = sequences
        self.time_intervals = time_intervals
        self.static_features = static_features
        self.labels = labels

        # Pre-filter data for efficiency (optional, but good practice)
        # self.sequences = {hid: sequences[hid] for hid in self.hadm_ids if hid in sequences}
        # self.time_intervals = {hid: time_intervals[hid] for hid in self.hadm_ids if hid in time_intervals}
        # self.static_features = {hid: static_features[hid] for hid in self.hadm_ids if hid in static_features}
        # self.labels = {hid: labels[hid] for hid in self.hadm_ids if hid in labels}
        # self.hadm_ids = [hid for hid in self.hadm_ids if hid in self.labels] # Ensure only IDs with labels remain

    def __len__(self) -> int:  # Add return type hint
        """Returns the number of samples in the dataset."""
        return len(self.hadm_ids)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # Add type hints
        """
        Retrieves a single sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - sequence_tensor: Shape [max_len, num_seq_feats]
                - interval_tensor: Shape [max_len, 1]
                - static_tensor: Shape [num_static_feats]
                - label_tensor: Shape [1]
        """
        hadm_id = self.hadm_ids[idx]

        # Retrieve pre-processed numpy arrays
        # Add error handling or default values if an ID is somehow missing data
        sequence = self.sequences.get(hadm_id)
        interval = self.time_intervals.get(hadm_id)
        static = self.static_features.get(hadm_id)
        label = self.labels.get(hadm_id)

        # Basic validation
        if sequence is None or interval is None or static is None or label is None:
            # This shouldn't happen if data is pre-filtered correctly in __init__ or preprocess
            # Handle this case, e.g., raise error or return None/dummy data
            raise IndexError(f"Data missing for hadm_id {hadm_id} at index {idx}")

        # Convert numpy arrays to PyTorch tensors
        sequence_tensor = torch.from_numpy(sequence).float()
        interval_tensor = torch.from_numpy(interval).float()
        static_tensor = torch.from_numpy(static).float()
        label_tensor = torch.tensor(
            [label], dtype=torch.float32
        )  # Label as float for BCE loss

        return sequence_tensor, interval_tensor, static_tensor, label_tensor


class TimeEncoder(nn.Module):
    """
    Encodes time intervals using a linear layer.
    Could be enhanced with sinusoidal embeddings or other techniques.
    """

    def __init__(self, embed_dim: int = 16):  # Add type hint
        super().__init__()
        self.embed_dim = embed_dim
        self.time_embed = nn.Linear(
            1, embed_dim
        )  # Assumes time interval is a single value

    def forward(self, time_intervals: torch.Tensor) -> torch.Tensor:  # Add type hints
        # time_intervals shape: [batch_size, seq_len, 1]
        time_encoding = self.time_embed(time_intervals)
        return time_encoding


class TimeAwarePatientLSTM(nn.Module):
    """
    LSTM model incorporating time embeddings and potentially static features.
    Includes an attention mechanism.
    """

    def __init__(
        self,
        input_dim: int,  # Dimension of sequence features
        hidden_dim: int,
        static_dim: int,  # Dimension of static features
        time_embed_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,  # Typically 1 for binary classification
    ):  # Add type hints
        super().__init__()
        self.time_encoder = TimeEncoder(time_embed_dim)

        # LSTM input dimension includes sequence features + time embedding
        lstm_input_dim = input_dim + time_embed_dim

        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout if num_layers > 1 else 0
            ),  # Dropout only between LSTM layers
        )

        # Attention mechanism (applied to LSTM outputs)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        # Classifier takes the context vector from attention
        # Potentially concatenate static features here or integrate differently
        classifier_input_dim = (
            hidden_dim + static_dim
        )  # Example: Concatenate static features
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        # Sigmoid applied outside the model during loss calculation (BCEWithLogitsLoss)

    def forward(
        self, x_seq: torch.Tensor, time_intervals: torch.Tensor, x_static: torch.Tensor
    ) -> torch.Tensor:  # Add type hints
        # x_seq shape: [batch_size, seq_len, input_dim]
        # time_intervals shape: [batch_size, seq_len, 1]
        # x_static shape: [batch_size, static_dim]

        # Encode time intervals
        time_encoding = self.time_encoder(
            time_intervals
        )  # [batch_size, seq_len, time_embed_dim]

        # Concatenate sequence features with time encoding
        x_seq_with_time = torch.cat(
            [x_seq, time_encoding], dim=2
        )  # [batch_size, seq_len, input_dim + time_embed_dim]

        # Process sequence with LSTM
        # Initialize hidden and cell states if needed (defaults to zeros)
        lstm_out, (h_n, c_n) = self.lstm(x_seq_with_time)
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        # h_n shape: [num_layers, batch_size, hidden_dim]
        # c_n shape: [num_layers, batch_size, hidden_dim]

        # Apply attention mechanism to LSTM outputs
        # attention input shape: [batch_size, seq_len, hidden_dim]
        attention_logits = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(
            attention_logits, dim=1
        )  # [batch_size, seq_len, 1]

        # Calculate context vector (weighted sum of LSTM outputs)
        context = torch.sum(
            attention_weights * lstm_out, dim=1
        )  # [batch_size, hidden_dim]

        # Concatenate context vector with static features
        combined_features = torch.cat(
            [context, x_static], dim=1
        )  # [batch_size, hidden_dim + static_dim]

        # Classify using the combined features
        logits = self.classifier(combined_features)  # [batch_size, output_dim]

        return logits  # Return logits, apply sigmoid in loss function


# --- Helper functions (potentially move to utils or keep here if specific) ---


def get_attention_weights(
    model: nn.Module,
    sequence: np.ndarray,
    time_interval: np.ndarray,
    static_features: np.ndarray,
    device: torch.device,
) -> np.ndarray:  # Add type hints
    """
    Extracts attention weights for a single sample.
    Requires adaptation based on final model input structure.
    """
    model.eval()
    with torch.no_grad():
        # Unsqueeze to add batch dimension
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        interval_tensor = torch.FloatTensor(time_interval).unsqueeze(0).to(device)
        static_tensor = torch.FloatTensor(static_features).unsqueeze(0).to(device)

        # --- Replicate relevant parts of the forward pass ---
        time_encoding = model.time_encoder(interval_tensor)
        x_seq_with_time = torch.cat([sequence_tensor, time_encoding], dim=2)
        lstm_out, _ = model.lstm(x_seq_with_time)
        attention_logits = model.attention(lstm_out)
        attention_weights = torch.softmax(attention_logits, dim=1)
        # --- End replication ---

    return (
        attention_weights.cpu().numpy().squeeze()
    )  # Return weights for the single sample
