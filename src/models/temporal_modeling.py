"""
Core PyTorch components for temporal EHR modeling.

This module defines:
- TemporalEHRDataset: A PyTorch Dataset class to handle sequences, static features, and labels.
- TimeEncoder: A simple module to encode time intervals.
- TimeAwarePatientLSTM: An LSTM-based model incorporating time awareness and static features,
  potentially using an attention mechanism.
- Helper functions related to temporal data processing (padding, intervals, attention extraction).
"""

from typing import Any, Dict, List, Optional, Tuple  # Added Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Note: Consider adding logger if more complex logging is needed within these components.
# from src.utils import get_logger
# logger = get_logger(__name__)


class TemporalEHRDataset(Dataset):
    """
    PyTorch Dataset for handling temporal EHR data.

    Expects pre-processed data structured as dictionaries mapping admission IDs
    to their corresponding sequences, time intervals, static features, and labels.
    It retrieves individual samples as tensors.

    Attributes:
        hadm_ids (List[str]): List of admission IDs included in this dataset split.
        sequences (Dict[str, np.ndarray]): Sequences of clinical events/measurements.
        time_intervals (Dict[str, np.ndarray]): Time intervals between sequence events.
        static_features (Dict[str, np.ndarray]): Static features per admission.
        labels (Dict[str, int]): Binary labels for each admission.
    """

    def __init__(
        self,
        processed_data: Dict[
            str, Any
        ],  # Expects the dict from TemporalReadmissionModel.preprocess
        targets: np.ndarray,  # Expects targets as a numpy array aligned with admission_ids
    ):
        """
        Initializes the TemporalEHRDataset.

        Args:
            processed_data (Dict[str, Any]): A dictionary containing the preprocessed data.
                Expected keys: 'sequences' (Dict[str, np.ndarray]),
                               'static_features' (pd.DataFrame),
                               'sequence_lengths' (List[int]),
                               'admission_ids' (pd.DataFrame).
                               The 'sequences' dict itself contains numpy arrays like
                               'lab_X_values_scaled', 'lab_X_timestamps', 'lab_X_intervals', etc.
            targets (np.ndarray): A NumPy array of target labels (0 or 1), aligned
                                  with the admission IDs in processed_data['admission_ids'].
        """
        self.admission_ids_df = processed_data["admission_ids"]
        self.hadm_ids = self.admission_ids_df[
            "hadm_id"
        ].tolist()  # Assuming hadm_id is the key identifier
        self.sequences = processed_data["sequences"]
        self.static_features = processed_data["static_features"]
        self.lengths = processed_data["sequence_lengths"]
        self.labels = targets  # Targets are passed directly

        # Extract sequence feature names (assuming structure like 'lab_X_values_scaled')
        self.sequence_feature_names = sorted(
            [
                k.replace("_values_scaled", "")
                for k in self.sequences
                if k.endswith("_values_scaled")
            ]
        )

        # Validate alignment
        if len(self.hadm_ids) != len(self.labels):
            raise ValueError("Mismatch between number of admission IDs and labels.")
        if len(self.hadm_ids) != len(self.lengths):
            raise ValueError(
                "Mismatch between number of admission IDs and sequence lengths."
            )
        if len(self.hadm_ids) != len(self.static_features):
            raise ValueError(
                "Mismatch between number of admission IDs and static features."
            )

    def __len__(self) -> int:
        """Returns the number of samples (admissions) in the dataset."""
        return len(self.hadm_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a single sample (admission data) from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the data for one admission:
                - 'sequence_values' (np.ndarray): Shape [seq_len, num_seq_features] - Scaled values.
                - 'sequence_intervals' (np.ndarray): Shape [seq_len] - Time intervals.
                - 'static_features' (np.ndarray): Shape [num_static_features].
                - 'length' (int): The original (unpadded) length of the sequence.
                - 'target' (int): The label (0 or 1).

        Raises:
            IndexError: If data for the requested index/hadm_id is missing.
        """
        # Use the index to get the corresponding row from admission_ids_df
        admission_info = self.admission_ids_df.iloc[idx]
        hadm_id = admission_info["hadm_id"]  # Or however hadm_id is stored if different
        subject_id = admission_info["subject_id"]  # Keep subject_id if needed

        seq_len = self.lengths[idx]

        # --- Assemble Sequence Features ---
        # Stack the scaled values, timestamps, and intervals for this admission
        # Order must be consistent (e.g., alphabetical by feature name)
        num_seq_features = len(self.sequence_feature_names)
        sequence_values_list = []
        sequence_intervals_list = []  # Only need intervals for TimeAwareLSTM

        for base_name in self.sequence_feature_names:
            # Get scaled values
            values_key = f"{base_name}_values_scaled"
            if values_key in self.sequences:
                # Select the sequence for the current index
                seq_vals = self.sequences[values_key][
                    idx, :seq_len
                ]  # Get unpadded sequence
                sequence_values_list.append(seq_vals)
            else:
                # Handle missing feature sequence (e.g., fill with zeros or raise error)
                # logger.warning(f"Missing sequence values for {base_name} at index {idx}")
                sequence_values_list.append(
                    np.zeros(seq_len)
                )  # Example: fill with zeros

            # Get intervals (only need one set, e.g., from the first feature)
            if not sequence_intervals_list and seq_len > 0:  # Only grab intervals once
                intervals_key = f"{base_name}_intervals"
                if intervals_key in self.sequences:
                    seq_intervals = self.sequences[intervals_key][idx, :seq_len]
                    sequence_intervals_list = seq_intervals  # Assign directly
                else:
                    # logger.warning(f"Missing sequence intervals for {base_name} at index {idx}")
                    sequence_intervals_list = np.zeros(
                        seq_len
                    )  # Example: fill with zeros

        # Stack features along the feature dimension
        sequence_values_array = (
            np.stack(sequence_values_list, axis=-1)
            if sequence_values_list
            else np.zeros((seq_len, 0))
        )
        # Ensure intervals is a 1D array
        sequence_intervals_array = (
            np.array(sequence_intervals_list)
            if sequence_intervals_list is not None
            else np.zeros(seq_len)
        )

        # --- Get Static Features ---
        # Assuming static_features is a DataFrame indexed correctly
        static_features_array = self.static_features.iloc[idx].values

        # --- Get Label ---
        label = self.labels[idx]

        return {
            "sequence_values": sequence_values_array,
            "sequence_intervals": sequence_intervals_array,
            "static_features": static_features_array,
            "length": seq_len,
            "target": label,
        }


class TimeEncoder(nn.Module):
    """
    Encodes time intervals using a simple linear layer.

    Can be extended to use more sophisticated time encoding methods like
    sinusoidal positional embeddings if needed.

    Args:
        embed_dim (int): The desired output dimension for the time embedding.
    """

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Linear layer expects input shape [*, H_in], where H_in is the last dimension size.
        # Our time interval is a single feature.
        self.time_embed = nn.Linear(1, embed_dim)

    def forward(self, time_intervals: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time encoding.

        Args:
            time_intervals (torch.Tensor): Tensor of time intervals, expected shape
                                           [batch_size, seq_len] or [batch_size, seq_len, 1].

        Returns:
            torch.Tensor: Encoded time tensor, shape [batch_size, seq_len, embed_dim].
        """
        # Ensure input has a feature dimension of 1 if it's [batch_size, seq_len]
        if time_intervals.ndim == 2:
            time_intervals = time_intervals.unsqueeze(
                -1
            )  # Add feature dim: [batch_size, seq_len, 1]

        time_encoding = self.time_embed(
            time_intervals
        )  # Output: [batch_size, seq_len, embed_dim]
        return time_encoding


class TimeAwarePatientLSTM(nn.Module):
    """
    Time-Aware LSTM model for EHR data.

    Combines sequence features with learned time interval embeddings before
    passing them to an LSTM layer. Optionally incorporates static features
    and uses an attention mechanism over the LSTM outputs before final classification.

    Args:
        input_dim (int): Dimensionality of the input sequence features at each time step.
        hidden_dim (int): Number of features in the hidden state of the LSTM.
        num_static_features (int): Dimensionality of the static features.
        num_layers (int, optional): Number of recurrent layers in the LSTM. Defaults to 1.
        dropout (float, optional): Dropout probability for LSTM layers (if num_layers > 1)
                                   and the final classifier. Defaults to 0.2.
        time_embed_dim (int, optional): Embedding dimension for time intervals. Defaults to 16.
        use_time_features (bool, optional): Whether to include time interval embeddings.
                                            Defaults to True.
        output_dim (int, optional): Dimension of the output layer (typically 1 for binary
                                    classification). Defaults to 1.
        device (torch.device, optional): Device to run the model on ('cpu' or 'cuda').
                                         If None, determined automatically. Defaults to None.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_static_features: int,
        num_layers: int = 1,  # Changed default to 1
        dropout: float = 0.2,
        time_embed_dim: int = 16,
        use_time_features: bool = True,
        output_dim: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.use_time_features = use_time_features
        self.time_embed_dim = time_embed_dim if use_time_features else 0
        self.num_static_features = num_static_features
        self.hidden_dim = hidden_dim
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self.use_time_features:
            self.time_encoder = TimeEncoder(self.time_embed_dim)
            lstm_input_dim = input_dim + self.time_embed_dim
        else:
            self.time_encoder = None
            lstm_input_dim = input_dim

        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0),
        )

        # Attention mechanism (applied to LSTM outputs)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        # Classifier takes the context vector from attention + static features
        classifier_input_dim = hidden_dim + num_static_features
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        # Sigmoid applied outside the model during loss calculation (BCEWithLogitsLoss)

    def forward(
        self,
        x_seq: torch.Tensor,
        time_intervals: torch.Tensor,
        x_static: torch.Tensor,
        lengths: List[int],  # Original lengths needed for packing
    ) -> torch.Tensor:
        """
        Forward pass of the TimeAwarePatientLSTM.

        Args:
            x_seq (torch.Tensor): Padded sequence features tensor.
                                  Shape: [batch_size, max_seq_len, input_dim].
            time_intervals (torch.Tensor): Padded time intervals tensor.
                                           Shape: [batch_size, max_seq_len].
            x_static (torch.Tensor): Static features tensor.
                                     Shape: [batch_size, num_static_features].
            lengths (List[int]): List of original sequence lengths for each sample in the batch.

        Returns:
            torch.Tensor: Output logits from the classifier. Shape: [batch_size, output_dim].
        """
        batch_size = x_seq.size(0)

        # Encode time intervals if used
        if self.use_time_features and self.time_encoder is not None:
            time_encoding = self.time_encoder(
                time_intervals
            )  # [batch_size, max_seq_len, time_embed_dim]
            # Concatenate sequence features with time encoding
            x_seq_combined = torch.cat([x_seq, time_encoding], dim=2)
        else:
            x_seq_combined = x_seq

        # Pack padded sequence (requires lengths on CPU)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x_seq_combined,
            lengths,
            batch_first=True,
            enforce_sorted=False,  # Already sorted in collate_fn
        )

        # Process sequence with LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        # Unpack sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # lstm_out shape: [batch_size, max_actual_seq_len_in_batch, hidden_dim]

        # Apply attention mechanism to LSTM outputs
        attention_logits = self.attention(
            lstm_out
        )  # [batch_size, max_actual_seq_len, 1]
        # Create mask based on actual lengths to ignore padding in softmax
        mask = torch.arange(lstm_out.size(1))[None, :].to(self.device) < torch.tensor(
            lengths
        )[:, None].to(self.device)
        attention_logits.masked_fill_(
            ~mask.unsqueeze(-1), -float("inf")
        )  # Mask padding tokens

        attention_weights = torch.softmax(
            attention_logits, dim=1
        )  # [batch_size, max_actual_seq_len, 1]

        # Calculate context vector (weighted sum of LSTM outputs)
        context = torch.sum(
            attention_weights * lstm_out, dim=1
        )  # [batch_size, hidden_dim]

        # Concatenate context vector with static features
        # Ensure static features have the correct batch size dimension
        if x_static.ndim == 1:  # Handle case where batch size might be 1
            x_static = x_static.unsqueeze(0)
        if x_static.size(0) != batch_size:
            # This might happen if static features weren't processed correctly
            raise ValueError(
                f"Batch size mismatch between context ({context.size(0)}) and static features ({x_static.size(0)})"
            )

        combined_features = torch.cat(
            [context, x_static], dim=1
        )  # [batch_size, hidden_dim + num_static_features]

        # Classify using the combined features
        logits = self.classifier(combined_features)  # [batch_size, output_dim]

        return logits  # Return logits, apply sigmoid in loss function


# --- Helper functions (potentially move to utils or keep here if specific) ---


def get_attention_weights(
    model: TimeAwarePatientLSTM,  # Use specific model type hint
    sequence_values: np.ndarray,
    sequence_intervals: np.ndarray,
    static_features: np.ndarray,
    length: int,  # Need original length
    device: torch.device,
) -> Optional[np.ndarray]:
    """
    Extracts attention weights for a single sample from the TimeAwarePatientLSTM.

    Performs a forward pass for the single sample and extracts the calculated
    attention weights before the final classification layer.

    Args:
        model (TimeAwarePatientLSTM): The trained TimeAwarePatientLSTM model instance.
        sequence_values (np.ndarray): The scaled sequence values for the sample. Shape [seq_len, num_features].
        sequence_intervals (np.ndarray): The time intervals for the sample. Shape [seq_len].
        static_features (np.ndarray): The static features for the sample. Shape [num_static_features].
        length (int): The original (unpadded) length of the sequence.
        device (torch.device): The device the model is on.

    Returns:
        Optional[np.ndarray]: NumPy array of attention weights for the sequence time steps
                              (shape [seq_len]), or None if an error occurs.
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension and move to device
        seq_val_tensor = torch.FloatTensor(sequence_values).unsqueeze(0).to(device)
        seq_int_tensor = (
            torch.FloatTensor(sequence_intervals).unsqueeze(0).to(device)
        )  # Add feature dim later if needed by TimeEncoder
        static_tensor = torch.FloatTensor(static_features).unsqueeze(0).to(device)
        lengths_list = [length]  # Length as a list for packing

        try:
            # --- Replicate relevant parts of the forward pass ---
            if model.use_time_features and model.time_encoder is not None:
                time_encoding = model.time_encoder(seq_int_tensor)
                x_seq_combined = torch.cat([seq_val_tensor, time_encoding], dim=2)
            else:
                x_seq_combined = seq_val_tensor

            packed_input = nn.utils.rnn.pack_padded_sequence(
                x_seq_combined, lengths_list, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = model.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )

            attention_logits = model.attention(lstm_out)
            # Create mask based on actual length
            mask = torch.arange(lstm_out.size(1))[None, :].to(device) < torch.tensor(
                lengths_list
            )[:, None].to(device)
            attention_logits.masked_fill_(~mask.unsqueeze(-1), -float("inf"))

            attention_weights = torch.softmax(attention_logits, dim=1)
            # --- End replication ---

            # Return weights for the actual length of the sequence
            return attention_weights.cpu().numpy().squeeze()[:length]

        except Exception as e:
            logger.error(f"Error extracting attention weights: {e}", exc_info=True)
            return None
