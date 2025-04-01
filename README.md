# MIMIC Critical Care Dataset Project

## Predicting Hospital Readmissions and Intensive Care Outcomes

This project uses the MIMIC-III and MIMIC-IV critical care datasets to develop machine learning models for predicting:

1. 30-day hospital readmission risk
2. ICU mortality prediction
3. Length of stay estimation

## Project Overview

This project combines advanced analytics with clinical insights to predict critical care outcomes in hospitalized patients. It incorporates state-of-the-art techniques in data preprocessing, feature engineering, and model development with a focus on interpretable machine learning to provide actionable insights for clinical decision-making.

### Architecture Overview

The project follows a structured MLOps pipeline from data ingestion to deployment:

![Architecture Diagram](docs/architecture.md)

For a detailed view of the architecture, including component descriptions and data flow, see the [Architecture Documentation](docs/architecture.md).

## Important Note: Using the MIMIC Demo Dataset

This project currently utilizes the **MIMIC-III Clinical Database Demo (v1.4)**, a publicly available subset of the full MIMIC-III dataset.

**Limitations:** Please be aware that this demo dataset is significantly smaller than the full MIMIC dataset. Consequently, the statistical power of the analyses and the generalizability of the model performance metrics presented here are limited.

**Project Goal:** The primary objective of this project is **not** to achieve state-of-the-art predictive performance on the demo data, but rather to **demonstrate a robust, reproducible, and end-to-end MLOps pipeline** for a complex clinical prediction task. This includes data processing, feature engineering, model training, evaluation (including imbalance handling), interpretability (SHAP), experiment tracking (MLflow), data versioning (DVC), containerization (Docker), CI/CD (GitHub Actions), and deployment via an API and dashboard. The methodology and pipeline developed here are designed to be scalable and applicable to the full MIMIC dataset or similar real-world healthcare data challenges.

## Advanced AI/ML Techniques

This project goes beyond standard machine learning approaches to implement cutting-edge techniques specifically suited for healthcare time-series data:

### Temporal Modelling

We've implemented advanced temporal models that leverage the sequential nature of EHR data:

- **LSTM Networks with Attention**: Captures temporal patterns in patient vitals and lab values while highlighting clinically significant events
- **Transformer-Based Approaches**: Inspired by BEHRT (BERT for Electronic Health Records) to model complex temporal dependencies
- **Temporal Convolutional Networks**: For multi-scale temporal pattern recognition

See our [advanced temporal model proof-of-concept](notebooks/advanced_temporal_model_poc.ipynb) for implementation details.

### Causal Inference

Moving beyond correlation to understand causal drivers of readmission:

- **Causal Forests**: For estimating heterogeneous treatment effects
- **Doubly Robust Estimation**: Combining outcome modelling and propensity scoring
- **Targeted Maximum Likelihood Estimation (TMLE)**: For robust causal effect estimation

### Modern MLOps Beyond Basics

Our MLOps implementation includes advanced monitoring and deployment strategies:

- **Comprehensive Drift Detection**: For data, concept, and prediction drift
- **Automated Model Retraining**: With performance-based promotion
- **Fairness Monitoring**: Continuous evaluation of model fairness across demographic groups

For details, see our [Advanced MLOps documentation](docs/advanced_mlops.md).

### Ethical Considerations & Bias Mitigation

We've implemented a comprehensive framework for addressing ethical considerations:

- **Bias Detection**: Pre-processing, in-processing, and post-processing techniques
- **Fairness-Aware Algorithms**: Including adversarial debiasing and fairness constraints
- **Explainability**: Natural language explanations from SHAP values

For details, see our [Ethical Considerations documentation](docs/ethical_considerations.md).

### Strategic Business Impact

We've quantified the potential impact of our models on healthcare outcomes:

- **Cost Savings Estimation**: Detailed ROI analysis for hospital readmission reduction
- **Clinical Workflow Integration**: Seamless integration with existing EHR systems
- **Stakeholder Communication**: Tailored communication strategies for different stakeholders

For details, see our [Strategic Impact documentation](docs/strategic_impact.md).

### Future Directions

For a comprehensive overview of advanced techniques that could further enhance this project, see our [Future Work documentation](FUTURE_WORK.md).

## Technical Components

### Data Pipeline and Preprocessing

- ETL pipeline for the MIMIC-III and MIMIC-IV datasets
- Data cleaning and preprocessing strategies for handling missing values, outliers, and temporality
- Standardised feature extraction methods for lab values, vital signs, medications, and procedures

### Feature Engineering and Domain Knowledge Integration

- Clinical severity scores based on vitals and laboratory values
- Temporal patterns in physiological measurements
- Medical knowledge through custom feature transformations
- NLP techniques for diagnostic codes and medical procedures

### Model Development

- Ensemble machine learning approaches (gradient boosting, random forests)
- Specialized models for different patient subgroups
- Transfer learning techniques across different ICU departments
- Advanced temporal models (LSTM, Transformers) for sequential EHR data
- Causal inference techniques for understanding intervention effects
- Fairness-aware algorithms with bias mitigation strategies

### Interpretability and Clinical Relevance

- SHAP values and LIME techniques for model explanation
- Risk stratification tools for clinical use
- Interactive visualisations for understanding model predictions
- Patient-specific risk profiles and intervention suggestions

### Deployment and Evaluation

- API for model inference
- Web dashboard for visualisation and interaction
- Monitoring and model performance tracking
- Documentation for reproducibility

## Project Structure

### Directory Structure

```
mimic-readmission-predictor/
├── configs/                   # Configuration files
├── data/
│   ├── raw/                   # Raw MIMIC data
│   ├── processed/             # Processed datasets
│   └── external/              # External datasets
├── src/
│   ├── data/                  # Data processing scripts
│   ├── features/              # Feature engineering
│   ├── models/                # Model implementations
│   ├── visualisation/         # Visualisation functions
│   └── utils/                 # Utility functions
├── tests/                     # Unit and integration tests
├── dashboard/                 # Dashboard implementation
├── api/                       # API implementation
├── docs/                      # Documentation
│   ├── architecture.md        # System architecture
│   ├── ethical_considerations.md # Ethical framework and bias mitigation
│   ├── strategic_impact.md    # Business impact analysis
│   └── advanced_mlops.md      # Advanced MLOps practices
├── .github/                   # CI/CD workflows
├── .gitignore                 # Git ignore file
├── LICENCE                    # Licence file
├── README.md                  # Project overview
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── Makefile                   # Project commands
```

## Getting Started

### Prerequisites

- Python 3.8+
- Access to MIMIC-III and/or MIMIC-IV datasets
- Required Python packages (see requirements.txt)

### Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Configure data paths in `configs/config.yaml`
4. Run the data processing pipeline:
   ```
   python src/data/make_dataset.py
   ```

### Usage

1. Process the raw data:
   ```
   python src/data/make_dataset.py
   ```

2. Generate features:
   ```
   python src/features/build_features.py
   ```

3. Train models:
   ```
   python src/models/train_model.py --model [readmission|mortality|los]
   ```

4. Generate predictions:
   ```
   python src/models/predict_model.py --model [readmission|mortality|los] --input [path_to_input_data]
   ```

5. Run the dashboard:
   ```
   python dashboard/app.py
   ```

6. Run the API:
   ```
   python api/app.py
   ```

## Technologies and Tools

- **Languages**: Python, SQL
- **Libraries**:
  - Data processing: pandas, numpy, scikit-learn, polars
  - Visualisation: matplotlib, seaborn, plotly
  - Machine Learning: XGBoost, LightGBM, PyTorch
  - Deep Learning: LSTM, Transformer models for temporal data
  - NLP: NLTK, spaCy, Hugging Face Transformers
  - Causal Inference: EconML, CausalML
  - Interpretability: SHAP, LIME, ELI5
- **Infrastructure**:
  - Git/GitHub (version control and project management)
  - Docker (containerization)
  - MLflow (comprehensive experiment tracking and model registry)
  - DVC (data version control)
  - FastAPI (API development)
  - Streamlit/Dash (interactive dashboards)
  - Evidently AI & Prometheus/Grafana (model monitoring)
  - GitHub Actions (CI/CD pipelines with automated testing and deployment)

## Licence

This project is licensed under the MIT Licence - see the LICENCE file for details.

## Acknowledgments

- MIMIC-III and MIMIC-IV dataset creators and maintainers
- PhysioNet for providing access to the datasets