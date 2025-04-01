"""
Setup script for the MIMIC project.
"""

from setuptools import setup, find_packages

setup(
    name="mimic-readmission-predictor",
    version="0.1.0",
    description="MIMIC Critical Care Dataset Project for predicting hospital readmissions and ICU outcomes",
    author="MIMIC Project Team",
    author_email="example@example.com",
    url="https://github.com/example/mimic-readmission-predictor",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "polars>=0.15.0",
        "pyarrow>=7.0.0",
        "tqdm>=4.62.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "shap>=0.40.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
        "click>=8.0.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.1.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mimic-process=src.data.make_dataset:main",
            "mimic-features=src.features.build_features:main",
            "mimic-train=src.models.train_model:main",
            "mimic-predict=src.models.predict_model:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Licence :: OSI Approved :: MIT Licence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)