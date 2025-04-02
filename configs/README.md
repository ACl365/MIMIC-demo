# MIMIC Demo Project Configuration

This directory contains configuration files for the MIMIC demo project.

## Configuration Files

- `config.yaml`: Main configuration file for the project
- `mappings.yaml`: Configuration file for clinical feature mappings

## Main Configuration (config.yaml)

The main configuration file (`config.yaml`) contains settings for:

- Logging configuration
  - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - File and console output options

- Data paths
  - Raw data paths (MIMIC-III, MIMIC-IV)
  - Processed data paths

- Feature engineering parameters
  - Demographic features
  - Vital sign features
  - Lab value features
  - Medication features
  - Procedure features
  - Diagnosis features
  - Temporal features

- Model parameters
  - Readmission prediction
  - Mortality prediction
  - Length of stay prediction

- Evaluation metrics
  - Classification metrics
  - Regression metrics

- Interpretability settings
  - SHAP
  - LIME
  - Feature importance

- Visualisation settings
  - Plot types
  - Interactive options

- API configuration
  - Host
  - Port
  - Debug mode

- Dashboard configuration
  - Host
  - Port
  - Debug mode

## Mappings Configuration (mappings.yaml)

The mappings configuration file (`mappings.yaml`) contains mappings for:

- Lab tests
  - Common lab tests to include in feature extraction
  - Mapping of standardised lab names to their variations in the dataset

- Vital signs
  - Mapping of vital sign categories to their itemids in MIMIC

- ICD-9 diagnosis codes
  - Mapping of ICD-9 code ranges to clinical categories
  - Specific ICD-9 codes for common conditions

## Usage

The configuration files are loaded using the `load_config` and `load_mappings` functions in the `src/utils/config.py` module:

```python
from src.utils import load_config, load_mappings

# Load main configuration
config = load_config()

# Access configuration values
log_level = config["logging"]["level"]
mimic_iii_path = config["data"]["raw"]["mimic_iii"]
include_demographics = config["features"]["demographic"]["include"]

# Load mappings configuration
mappings = load_mappings()

# Access mappings values
common_labs = mappings["lab_tests"]["common_labs"]
vital_categories = mappings["vital_signs"]["categories"]
icd9_ranges = mappings["icd9_categories"]["ranges"]
```

## Modifying Configuration

When modifying the configuration files, follow these guidelines:

1. Maintain the YAML format
2. Use consistent indentation (2 spaces)
3. Add comments to explain non-obvious settings
4. Keep related settings grouped together
5. Use appropriate data types (strings, numbers, booleans, lists, dictionaries)
6. Update the code that uses the configuration if you change the structure

## Configuration Impact on Pipeline

Key configuration choices directly impact the behaviour of the pipeline. Here are some important examples:

### Feature Engineering

- **`features.vitals.window_hours`**: Controls the time window (in hours) used for aggregating vital sign measurements in `feature_extractors.py`. Increasing this value will capture longer-term trends but may reduce the temporal resolution of the data.

- **`features.labs.include_trends`**: When set to `true`, the pipeline will extract trend features (e.g., slope, variance) from lab values over time, which can capture deterioration or improvement patterns.

- **`features.demographic.encoding`**: Determines how categorical demographic variables are encoded (e.g., one-hot, target, frequency). This affects model interpretability and performance.

### Model Training

- **`models.readmission.algorithms`**: The list of algorithms to try for readmission prediction. The first algorithm in the list is used by default if not specified.

- **`models.readmission.hyperparameter_tuning.enabled`**: When `true`, enables grid search for hyperparameter optimization, which can significantly improve model performance but increases training time.

- **`models.readmission.class_weight`**: Setting this to `balanced` helps address class imbalance in the readmission prediction task, improving recall for the minority class.

### Data Processing

- **`data.processed.imputation_strategy`**: Determines how missing values are handled during preprocessing. Options like `median`, `mean`, or `zero` have different effects on feature distributions.

- **`data.raw.mimic_version`**: Specifies which version of MIMIC to use (III or IV), affecting the available features and data structure.

### API and Dashboard

- **`api.batch_size`**: Controls the maximum number of samples that can be processed in a single API request, affecting memory usage and response time.

- **`dashboard.cache_predictions`**: When enabled, caches model predictions to improve dashboard responsiveness at the cost of memory usage.

### Logging and Debugging

- **`logging.level`**: Setting this to `DEBUG` provides more detailed logs for troubleshooting but increases log file size.

- **`logging.console_output`**: When `true`, logs are printed to the console in addition to being written to log files.
