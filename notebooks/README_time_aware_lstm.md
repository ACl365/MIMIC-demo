# Time-Aware LSTM for Readmission Prediction

This implementation enhances the original LSTM proof-of-concept by adding time-aware embeddings and conducting a more rigorous comparison with baseline models. It demonstrates how incorporating temporal information can improve readmission prediction performance.

## Overview

The `time_aware_lstm.py` script implements a time-aware LSTM model that explicitly incorporates time intervals between measurements. This allows the model to understand not just the sequence of values, but also when they occurred, which is crucial for clinical time-series data where the timing of measurements can be as important as the values themselves.

## Key Features

1. **Time-Aware Embeddings**: The model encodes time intervals between measurements, allowing it to understand the clinical significance of when measurements were taken.

2. **Attention Mechanism**: The model includes an attention mechanism that considers both the values and their temporal context, focusing on clinically significant patterns.

3. **Rigorous Comparison**: The implementation includes a fair comparison with a LightGBM baseline model, using the same train/test split and evaluating on both ROC AUC and PR AUC metrics.

4. **Interpretability**: The attention weights provide interpretable insights into which time points were most important for the prediction.

## How to Run

1. Ensure you have all required dependencies installed:
   ```
   pip install torch pandas numpy matplotlib seaborn scikit-learn lightgbm
   ```

2. Run the script:
   ```
   python time_aware_lstm.py
   ```

3. The script will:
   - Load the MIMIC data
   - Create synthetic temporal sequences with time intervals
   - Train the time-aware LSTM model
   - Train a baseline LightGBM model for comparison
   - Visualise the results and save plots to the results directory
   - Analyse the attention weights to identify important time points

## Results

The script generates several visualisations:

1. **Training Curves**: Shows the training and test loss, as well as ROC AUC and PR AUC comparisons between the LSTM and LightGBM models.

2. **ROC and PR Curves**: Compares the ROC and PR curves of the LSTM and LightGBM models.

3. **Attention Analysis**: Visualises the attention weights over time, along with the corresponding vital signs and lab values, to identify which time points were most important for the prediction.

## Advantages of Time-Aware Temporal Modeling

1. **Explicit Time Encoding**: By incorporating time intervals between measurements, the model can better understand the clinical significance of when measurements were taken. Rapid changes in vital signs or lab values often indicate deterioration, while stable measurements over time may indicate recovery.

2. **Attention Mechanism with Time Context**: The attention mechanism now considers not just the values themselves but their temporal context, allowing it to focus on clinically significant patterns that develop over specific timeframes.

3. **Improved Performance**: The time-aware LSTM model demonstrates competitive or superior performance compared to traditional ML models like LightGBM, particularly in terms of PR AUC, which is important for imbalanced classification problems like readmission prediction.

4. **Interpretability with Temporal Context**: The attention weights provide interpretable insights into which time points were most important for the prediction, which could help clinicians understand when critical changes occurred and potentially intervene earlier.

## Limitations and Future Work

1. **Simulated Data**: This implementation uses simulated temporal data. In a real implementation, actual time-stamped measurements from the MIMIC database would provide more realistic patterns and potentially better performance.

2. **Advanced Time Encodings**: More sophisticated time encoding methods could be explored, such as sinusoidal positional encoding (as used in Transformer models) or learnable time embeddings that capture domain-specific temporal patterns in clinical data.

3. **Multi-modal Integration**: Future work could integrate this temporal approach with other data modalities, such as clinical notes, medications, and procedures, to create a more comprehensive patient representation.

4. **Causal Analysis**: Combining temporal modeling with causal inference techniques could help identify not just when important changes occur, but what interventions might have caused those changes and how they affect readmission risk.

## Clinical Implications

The ability to identify specific time points that are predictive of readmission could help clinicians:

1. Develop more targeted intervention strategies at critical time points
2. Better understand the temporal progression of risk factors
3. Create more personalised discharge planning based on individual temporal patterns

This enhanced temporal modeling approach represents a significant step forward from traditional static models, moving us closer to truly understanding the dynamic nature of patient health trajectories.
