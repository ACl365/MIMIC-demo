# MIMIC Demo Project Architecture

This document provides an overview of the project architecture and data flow.

## Architecture Diagram

```mermaid
graph TD
    subgraph Data_Sources
        MIMIC3[MIMIC-III Dataset]
        MIMIC4[MIMIC-IV Dataset]
    end

    subgraph Data_Processing
        Ingest[Data Ingestion]
        Clean[Data Cleaning]
        Process[Data Processing]
    end

    subgraph Feature_Engineering
        Demographics[Demographic Features]
        Vitals[Vital Sign Features]
        Labs[Lab Value Features]
        Meds[Medication Features]
        Procedures[Procedure Features]
        Diagnoses[Diagnosis Features]
        Temporal[Temporal Features]
        Combine[Feature Combination]
    end

    subgraph Model_Development
        Train[Model Training]
        Tune[Hyperparameter Tuning]
        Evaluate[Model Evaluation]
        Interpret[Model Interpretation]
    end

    subgraph Deployment
        API[REST API]
        Dashboard[Web Dashboard]
        Monitor[Model Monitoring]
    end

    %% Data flow
    MIMIC3 --> Ingest
    MIMIC4 --> Ingest
    Ingest --> Clean
    Clean --> Process
    Process --> Demographics
    Process --> Vitals
    Process --> Labs
    Process --> Meds
    Process --> Procedures
    Process --> Diagnoses
    Process --> Temporal

    Demographics --> Combine
    Vitals --> Combine
    Labs --> Combine
    Meds --> Combine
    Procedures --> Combine
    Diagnoses --> Combine
    Temporal --> Combine

    Combine --> Train
    Train --> Tune
    Tune --> Evaluate
    Evaluate --> Interpret

    Interpret --> API
    Interpret --> Dashboard
    API --> Monitor
    Dashboard --> Monitor

    %% Styling
    classDef sourceNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333;
    classDef processNode fill:#bbf,stroke:#33f,stroke-width:2px,color:#333;
    classDef featureNode fill:#bfb,stroke:#3f3,stroke-width:2px,color:#333;
    classDef modelNode fill:#fbb,stroke:#f33,stroke-width:2px,color:#333;
    classDef deployNode fill:#ffb,stroke:#ff3,stroke-width:2px,color:#333;

    class MIMIC3,MIMIC4 sourceNode;
    class Ingest,Clean,Process processNode;
    class Demographics,Vitals,Labs,Meds,Procedures,Diagnoses,Temporal,Combine featureNode;
    class Train,Tune,Evaluate,Interpret modelNode;
    class API,Dashboard,Monitor deployNode;
```

## Component Descriptions

### Data Sources
- **MIMIC-III Dataset**: Medical Information Mart for Intensive Care III - a large, freely-available database comprising de-identified health-related data.
- **MIMIC-IV Dataset**: The updated version of MIMIC-III with additional data types and improved structure.

### Data Processing
- **Data Ingestion**: Scripts to load and parse raw MIMIC data files.
- **Data Cleaning**: Handling missing values, outliers, and inconsistencies in the raw data.
- **Data Processing**: Transforming and structuring the data for feature extraction.

### Feature Engineering
- **Demographic Features**: Patient age, gender, ethnicity, etc.
- **Vital Sign Features**: Heart rate, blood pressure, temperature, etc.
- **Lab Value Features**: Blood tests, chemistry panels, etc.
- **Medication Features**: Drugs administered, dosages, etc.
- **Procedure Features**: Medical procedures performed during the stay.
- **Diagnosis Features**: ICD-9/10 codes and clinical categories.
- **Temporal Features**: Time-based patterns and sequences.
- **Feature Combination**: Integration of all feature types into a unified dataset.

### Model Development
- **Model Training**: Training machine learning models on the prepared features.
- **Hyperparameter Tuning**: Optimising model parameters for best performance.
- **Model Evaluation**: Assessing model performance using appropriate metrics.
- **Model Interpretation**: Explaining model predictions using SHAP values and other techniques.

### Deployment
- **REST API**: FastAPI service for real-time predictions.
- **Web Dashboard**: Interactive visualisation and exploration of model predictions.
- **Model Monitoring**: Tracking model performance and data drift over time.

## Key Files and Directories

- `src/data/`: Data processing scripts
- `src/features/`: Feature engineering code
- `src/models/`: Model implementation and training
- `src/utils/`: Utility functions and configuration
- `api/`: API implementation
- `dashboard/`: Dashboard implementation
- `configs/`: Configuration files
- `tests/`: Unit and integration tests

## Configuration Impact

The project's behaviour can be significantly modified through configuration settings:

- **Feature Engineering**: Changing `window_hours` in `config.yaml` under `features.vitals` directly affects the time window used for vital sign aggregation in `feature_extractors.py`.
- **Model Selection**: The `algorithms` setting under each model type in `config.yaml` determines which ML algorithms are used.
- **Evaluation Metrics**: The metrics used for model evaluation can be configured in the `evaluation` section.
- **API Settings**: Host, port, and debug settings for the API can be modified in the configuration.
