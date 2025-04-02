#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Run the time-aware LSTM implementation
python time_aware_lstm.py

echo "Time-aware LSTM analysis complete. Results saved to the 'results' directory."
