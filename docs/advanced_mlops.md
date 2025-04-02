# Advanced MLOps for MIMIC Readmission Prediction

This document outlines advanced MLOps practices for the MIMIC readmission prediction project, focusing on monitoring, experiment tracking, and CI/CD pipelines that go beyond basic implementations.

## Model Monitoring and Observability

### Data Drift Detection

Data drift occurs when the statistical properties of the input data change over time, potentially degrading model performance. For healthcare data, this is particularly critical as clinical practices evolve.

#### Implementation with Evidently AI

```python
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, NumTargetDriftTab

def monitor_data_drift(reference_data, current_data, column_mapping=None):
    """Monitor data drift between reference and current datasets."""
    # Create dashboard with data drift and target drift tabs
    dashboard = Dashboard(tabs=[DataDriftTab(), NumTargetDriftTab()])

    # Calculate drift metrics
    dashboard.calculate(reference_data, current_data, column_mapping=column_mapping)

    # Get drift metrics
    drift_metrics = dashboard.get_metrics()

    # Extract key metrics
    data_drift_score = drift_metrics['data_drift']['data_drift_score']
    drifted_features = [
        feature for feature, metrics in drift_metrics['data_drift']['feature_metrics'].items()
        if metrics['drift_detected']
    ]

    # Log drift metrics
    logger.info(f"Data Drift Score: {data_drift_score:.4f}")
    logger.info(f"Number of Drifted Features: {len(drifted_features)}")

    return dashboard, drift_metrics
```

#### Custom Statistical Drift Detection

For more specialized healthcare metrics, we implement custom drift detection:

```python
def detect_clinical_drift(reference_data, current_data, clinical_features, threshold=0.05):
    """Detect drift in clinical features using appropriate statistical tests."""
    from scipy.stats import ks_2samp, chi2_contingency

    results = []

    for feature, feature_type in clinical_features.items():
        # Extract feature values
        ref_values = reference_data[feature].dropna()
        cur_values = current_data[feature].dropna()

        # Detect drift based on feature type
        if feature_type == 'numerical':
            # Kolmogorov-Smirnov test for numerical features
            statistic, p_value = ks_2samp(ref_values, cur_values)
            test_name = 'Kolmogorov-Smirnov'

        elif feature_type == 'categorical':
            # Chi-squared test for categorical features
            ref_counts = pd.Series(ref_values).value_counts()
            cur_counts = pd.Series(cur_values).value_counts()

            # Get all categories
            all_categories = sorted(set(ref_counts.index) | set(cur_counts.index))

            # Create contingency table
            contingency = np.zeros((2, len(all_categories)))
            for i, category in enumerate(all_categories):
                contingency[0, i] = ref_counts.get(category, 0)
                contingency[1, i] = cur_counts.get(category, 0)

            # Chi-squared test
            statistic, p_value, _, _ = chi2_contingency(contingency)
            test_name = 'Chi-squared'

        else:
            # Skip unknown feature types
            continue

        # Determine if drift detected
        drift_detected = p_value < threshold

        # Add to results
        results.append({
            'feature': feature,
            'feature_type': feature_type,
            'test': test_name,
            'p_value': p_value,
            'drift_detected': drift_detected
        })

    return pd.DataFrame(results)
```

### Concept Drift Detection

Concept drift occurs when the relationship between input features and the target variable changes. In healthcare, this might happen due to changes in clinical practice or patient populations.

```python
def detect_concept_drift(model, reference_X, reference_y, current_X, current_y):
    """Detect concept drift by comparing model performance on reference and current datasets."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    # Get predictions on both datasets
    ref_probs = model.predict_proba(reference_X)[:, 1]
    cur_probs = model.predict_proba(current_X)[:, 1]

    # Calculate performance metrics on reference data
    ref_metrics = {
        'auc': roc_auc_score(reference_y, ref_probs),
        'avg_precision': average_precision_score(reference_y, ref_probs),
        'brier': brier_score_loss(reference_y, ref_probs)
    }

    # Calculate performance metrics on current data
    cur_metrics = {
        'auc': roc_auc_score(current_y, cur_probs),
        'avg_precision': average_precision_score(current_y, cur_probs),
        'brier': brier_score_loss(current_y, cur_probs)
    }

    # Calculate relative performance changes
    performance_changes = {
        metric: (cur_metrics[metric] - ref_metrics[metric]) / ref_metrics[metric]
        for metric in ref_metrics
    }

    # Determine if concept drift detected
    concept_drift_detected = (
        abs(performance_changes['auc']) > 0.05 or
        abs(performance_changes['avg_precision']) > 0.05
    )

    return {
        'reference_metrics': ref_metrics,
        'current_metrics': cur_metrics,
        'performance_changes': performance_changes,
        'concept_drift_detected': concept_drift_detected
    }
```

### Comprehensive Monitoring Dashboard

We'll implement a comprehensive monitoring dashboard using Grafana and Prometheus:

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.30.3
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:8.2.2
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    restart: unless-stopped
    depends_on:
      - prometheus

  model-metrics-exporter:
    build:
      context: ./monitoring
      dockerfile: Dockerfile.metrics-exporter
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    ports:
      - "8000:8000"
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

## Experiment Tracking with MLflow

While the project already mentions MLflow, we'll implement a more comprehensive tracking system that goes beyond basic metrics logging.

### Structured Experiment Organisation

```python
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def setup_mlflow_experiment(experiment_name, artifact_location=None):
    """Set up MLflow experiment with proper organisation."""
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location
        )
    else:
        experiment_id = experiment.experiment_id

    # Set experiment as active
    mlflow.set_experiment(experiment_name)

    return experiment_id

def log_dataset_profile(data, name="dataset"):
    """Log dataset profile as artifact."""
    from pandas_profiling import ProfileReport

    # Generate profile report
    profile = ProfileReport(data, title=f"{name} Profile", minimal=True)

    # Save report
    report_path = f"{name}_profile.html"
    profile.to_file(report_path)

    # Log as artifact
    mlflow.log_artifact(report_path)

    # Log dataset shape
    mlflow.log_param(f"{name}_rows", data.shape[0])
    mlflow.log_param(f"{name}_columns", data.shape[1])
```

### Comprehensive Run Tracking

```python
def log_model_training_run(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name,
    params,
    feature_names,
    target_name,
    sensitive_attributes=None
):
    """Log comprehensive model training run to MLflow."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        roc_curve, precision_recall_curve, confusion_matrix,
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score, accuracy_score
    )
    import shap

    # Start run
    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        mlflow.log_params(params)

        # Log dataset information
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("positive_class_ratio_train", y_train.mean())

        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Log basic metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "pr_auc": average_precision_score(y_test, y_pred_proba)
        }
        mlflow.log_metrics(metrics)

        # Log ROC curve
        fig, ax = plt.subplots(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        mlflow.log_figure(fig, "roc_curve.png")

        # Log confusion matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        mlflow.log_figure(fig, "confusion_matrix.png")

        # Log feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]

            fig, ax = plt.subplots(figsize=(12, 10))
            ax.barh(range(len(indices[:20])), importance[indices[:20]])
            ax.set_yticks(range(len(indices[:20])))
            ax.set_yticklabels([feature_names[i] for i in indices[:20]])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 20 Feature Importances')
            mlflow.log_figure(fig, "feature_importance.png")

        # Log SHAP values
        try:
            # Sample data for SHAP analysis if dataset is large
            if len(X_test) > 500:
                X_sample = X_test.sample(500, random_state=42)
            else:
                X_sample = X_test

            # Create explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model)
                shap_values = explainer(X_sample)

                # Summary plot
                fig, ax = plt.subplots(figsize=(12, 10))
                shap.summary_plot(shap_values, X_sample, show=False)
                mlflow.log_figure(fig, "shap_summary.png")
        except Exception as e:
            mlflow.log_param("shap_error", str(e))

        # Log model with signature
        signature = infer_signature(X_train, y_pred_proba)
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train.iloc[0:5]
        )

        return run.info.run_id
```

## Robust CI/CD Pipeline

We'll implement a comprehensive CI/CD pipeline using GitHub Actions:

```yaml
# .github/workflows/mlops-pipeline.yml
name: MLOps Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly run on Sundays

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    - name: Lint with flake8
      run: |
        flake8 src tests
    - name: Type check with mypy
      run: |
        mypy src
    - name: Run tests
      run: |
        pytest tests/
    - name: Check code coverage
      run: |
        pytest --cov=src tests/

  build-and-push:
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1
    - name: Login to DockerHub
      uses: docker/login-action@v1
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    - name: Build and push API image
      uses: docker/build-push-action@v2
      with:
        context: ./api
        push: true
        tags: mimic-readmission-api:latest
    - name: Build and push Dashboard image
      uses: docker/build-push-action@v2
      with:
        context: ./dashboard
        push: true
        tags: mimic-readmission-dashboard:latest

  retrain-model:
    needs: test
    if: github.event_name == 'schedule' || (github.event_name == 'push' && contains(github.event.head_commit.message, '[retrain]'))
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Download latest data
      run: |
        python src/data/make_dataset.py
    - name: Retrain model
      run: |
        python src/models/train_model.py --log-to-mlflow
    - name: Evaluate model
      run: |
        python src/models/evaluate_model.py --latest
    - name: Deploy if performance improved
      run: |
        python scripts/deploy_if_improved.py
```

### Automated Model Deployment Script

```python
# scripts/deploy_if_improved.py
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

def deploy_model_if_better(
    model_name,
    new_version,
    metric='roc_auc',
    min_improvement=0.01
):
    """Deploy a model to production if it performs better than the current production model."""
    # Get client
    client = MlflowClient()

    # Get current production model if any
    production_models = client.get_latest_versions(model_name, stages=["Production"])

    # Transition new model to staging
    client.transition_model_version_stage(
        name=model_name,
        version=new_version,
        stage="Staging"
    )

    # If no production model, promote new model directly
    if len(production_models) == 0:
        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production"
        )
        print(f"Model {model_name} version {new_version} promoted to Production (no previous production model)")
        return True

    # Get current production model
    production_model = production_models[0]

    # Get run IDs
    production_run_id = client.get_model_version(
        name=model_name,
        version=production_model.version
    ).run_id

    new_run_id = client.get_model_version(
        name=model_name,
        version=new_version
    ).run_id

    # Get metrics
    production_metrics = client.get_run(production_run_id).data.metrics
    new_metrics = client.get_run(new_run_id).data.metrics

    # Compare metrics
    if metric in new_metrics and metric in production_metrics:
        improvement = new_metrics[metric] - production_metrics[metric]

        if improvement > min_improvement:
            # Promote new model to production
            client.transition_model_version_stage(
                name=model_name,
                version=new_version,
                stage="Production"
            )

            # Archive old production model
            client.transition_model_version_stage(
                name=model_name,
                version=production_model.version,
                stage="Archived"
            )

            print(f"Model {model_name} version {new_version} promoted to Production")
            print(f"Previous model version {production_model.version} archived")
            print(f"Improvement in {metric}: {improvement:.4f}")
            return True
        else:
            print(f"Model {model_name} version {new_version} not promoted to Production")
            print(f"Improvement in {metric} ({improvement:.4f}) below threshold ({min_improvement})")
            return False
    else:
        print(f"Cannot compare models: metric {metric} not found in both runs")
        return False

if __name__ == "__main__":
    # Get latest model version from MLflow
    client = MlflowClient()
    model_name = "readmission_predictor"

    latest_versions = client.get_latest_versions(model_name, stages=["None"])
    if not latest_versions:
        print(f"No versions found for model {model_name}")
        sys.exit(1)

    latest_version = latest_versions[0].version

    # Deploy if better
    deploy_model_if_better(model_name, latest_version)
```


## Data and Model Versioning Strategy

Versioning both the datasets used for training and the resulting model artifacts is crucial for reproducibility, traceability, and managing the evolution of the ML system. Without proper versioning, it becomes difficult to:

*   Reproduce past experiments or model results.
*   Track which dataset was used to train a specific model version.
*   Roll back to a previous model version if issues arise in production.
*   Understand the impact of data changes on model performance over time.

### Proposed Strategy

1.  **Data Versioning (using DVC - Data Version Control):**
    *   **Tool:** DVC integrates seamlessly with Git to version large data files and directories without bloating the Git repository.
    *   **Process:**
        *   Initialize DVC in the project (`dvc init`).
        *   Add large data files/directories (e.g., `data/raw`, `data/processed`) to DVC tracking (`dvc add data/processed/combined_features.csv`). This creates small `.dvc` metafiles that are committed to Git.
        *   Configure remote storage (like S3, GCS, Azure Blob, or even a shared network drive) where the actual data files will be stored (`dvc remote add ...`).
        *   Push the data to remote storage (`dvc push`).
        *   Commit the `.dvc` files and changes to `.dvc/config` to Git.
    *   **Benefits:** Git tracks the *versions* of the data (via the metafiles), while DVC handles the storage and retrieval of the large files. Checking out a specific Git commit allows you to retrieve the corresponding data version using `dvc pull`.

2.  **Model Versioning (using MLflow Model Registry or Git LFS):**
    *   **Option A: MLflow Model Registry (Recommended):**
        *   **Tool:** MLflow provides a centralized model store with versioning, stage management (Staging, Production, Archived), and metadata tracking.
        *   **Process:** When logging models during training (as implemented in `train_model.py`), use the `registered_model_name` argument in `mlflow.sklearn.log_model` or `mlflow.log_artifact`. This automatically versions the model within the registry.
        *   Use the MLflow UI or API to manage model stages (e.g., promote a validated model from "Staging" to "Production").
        *   The API or deployment process can then fetch the "Production" version of the model by name.
        *   **Benefits:** Centralized management, stage transitions, lineage tracking (links model version to the run that produced it), API for programmatic access.
    *   **Option B: Git LFS (Large File Storage) + Git Tags:**
        *   **Tool:** Git LFS is a Git extension for versioning large files directly in Git repositories.
        *   **Process:**
            *   Install and initialize Git LFS (`git lfs install`).
            *   Track model file types (e.g., `*.pkl`) using `git lfs track "*.pkl"`. Commit the `.gitattributes` file.
            *   Commit the model files (`models/readmission_model.pkl`) as usual. Git LFS replaces them with text pointers in Git, storing the actual files on an LFS server.
            *   Use Git tags (e.g., `git tag v1.0 <commit_hash>`) to mark specific commits corresponding to model releases.
        *   **Benefits:** Keeps everything within the Git workflow. Simpler setup if not using MLflow extensively.
        *   **Drawbacks:** Less metadata tracking compared to MLflow Model Registry, no built-in stage management. Requires LFS server setup or compatible hosting (like GitHub, GitLab).

### Importance Summary

Implementing a clear versioning strategy for both data and models ensures that the entire ML workflow is reproducible and auditable. It allows teams to confidently manage changes, debug issues, and deploy reliable models, which is paramount in sensitive domains like healthcare.


## Conclusion

By implementing these advanced MLOps practices, the MIMIC readmission prediction project will benefit from:

1. **Robust Monitoring**: Early detection of data and concept drift ensures the model remains accurate as clinical practices evolve.

2. **Comprehensive Experiment Tracking**: Detailed logging of model training runs facilitates comparison and selection of the best models.

3. **Automated CI/CD Pipeline**: Regular testing, building, and deployment reduces manual effort and ensures consistency.

4. **Automated Retraining**: Scheduled retraining with performance-based promotion ensures the model improves over time.

These practices go beyond basic MLOps implementations to provide a production-grade system suitable for healthcare applications where reliability and performance are critical.
