# Imbalance Handling Techniques Analysis

This document describes the implementation and analysis of different techniques for handling class imbalance in the readmission prediction model.

## Overview

Class imbalance is a common challenge in healthcare prediction tasks, where the positive class (e.g., patients who get readmitted) is often much less frequent than the negative class. This imbalance can lead to models that are biased towards the majority class, resulting in poor performance on the minority class.

This analysis implements and compares the following techniques:

1. **Baseline** (no imbalance handling)
2. **Class weights** (`class_weight='balanced'`)
3. **Random oversampling**
4. **SMOTE** (Synthetic Minority Over-sampling Technique)
5. **Random undersampling**

## Implementation Details

### Techniques

#### Class Weights

Class weights adjust the importance of each class during training, making the model pay more attention to the minority class without changing the data. In scikit-learn, this is implemented using the `class_weight='balanced'` parameter.

```python
LogisticRegression(class_weight='balanced')
```

#### Random Oversampling

Random oversampling duplicates existing minority samples to balance the classes. This is implemented using the `RandomOverSampler` from the `imbalanced-learn` library.

```python
from imblearn.over_sampling import RandomOverSampler
sampler = RandomOverSampler()
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

#### SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic examples by interpolating between existing minority samples. This is implemented using the `SMOTE` class from the `imbalanced-learn` library.

```python
from imblearn.over_sampling import SMOTE
sampler = SMOTE()
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

#### Random Undersampling

Random undersampling removes samples from the majority class to balance the classes. This is implemented using the `RandomUnderSampler` from the `imbalanced-learn` library.

```python
from imblearn.under_sampling import RandomUnderSampler
sampler = RandomUnderSampler()
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

### Evaluation Metrics

The techniques are evaluated using the following metrics:

- **Precision**: The ability of the model to avoid false positives
- **Recall**: The ability of the model to find all positive samples
- **F1 Score**: The harmonic mean of precision and recall
- **PR AUC**: The area under the precision-recall curve

All metrics are computed using cross-validation to ensure robust evaluation.

## Running the Analysis

### Command Line

You can run the analysis from the command line using the `run_imbalance_analysis.py` script:

```bash
python run_imbalance_analysis.py --output-dir results/imbalance_analysis --cv-folds 5
```

Arguments:
- `--output-dir`: Directory to save outputs (default: results/imbalance_analysis)
- `--cv-folds`: Number of cross-validation folds (default: 5)
- `--random-state`: Random state for reproducibility (default: 42)

### Jupyter Notebook

Alternatively, you can run the analysis interactively using the Jupyter notebook:

```bash
jupyter notebook notebooks/imbalance_analysis.ipynb
```

The notebook provides a more interactive experience with visualisations and detailed explanations.

## Expected Outputs

The analysis generates the following outputs:

1. **PR Curves**: Precision-recall curves for each technique
2. **Metrics Comparison**: Bar chart comparing precision, recall, F1, and PR AUC across techniques
3. **Results CSV**: CSV file with all evaluation metrics
4. **Summary Report**: Text summary of the results and discussion of trade-offs

## Understanding the Trade-offs

### Baseline vs. Class Weights

Class weights adjust the importance of each class during training, which can help the model pay more attention to the minority class without changing the data. This typically improves recall at the expense of precision.

### Random Oversampling vs. SMOTE

Random oversampling duplicates existing minority samples, which can lead to overfitting as the model sees the exact same minority samples multiple times.

SMOTE creates synthetic examples by interpolating between existing minority samples, which can help the model generalise better by learning from a more diverse set of minority class examples.

SMOTE may perform differently than random oversampling because it creates new, synthetic samples rather than just duplicating existing ones, potentially leading to better generalisation but possibly introducing noise if the synthetic samples are not representative of the true data distribution.

### Oversampling vs. Undersampling

Oversampling techniques (Random Oversampling, SMOTE) increase the number of minority class samples to balance the classes, preserving all available information but potentially leading to longer training times and overfitting.

Undersampling reduces the number of majority class samples, which can lead to information loss but may help prevent the model from being biased towards the majority class and can reduce training time.

## Limitations

It's important to note the limitations of this analysis:

1. **Small Dataset Size**: With only ~200 demo patients, the absolute performance metrics may be unstable and not generalisable. The relative differences between techniques are more informative than the absolute values.

2. **Cross-validation Stability**: Even with cross-validation, the small dataset size means that the results may vary significantly depending on the random splits.

3. **Model Simplicity**: We used logistic regression for all techniques to focus on the imbalance handling methods, but more complex models might interact differently with these techniques.

## Next Steps

For a more comprehensive analysis, consider:

1. Testing these techniques on the full MIMIC dataset when available
2. Exploring combinations of techniques (e.g., SMOTE + class weights)
3. Trying different base classifiers (e.g., random forest, XGBoost)
4. Implementing more advanced techniques like ADASYN, SMOTETomek, or SMOTEENN
5. Exploring threshold optimisation for each technique
