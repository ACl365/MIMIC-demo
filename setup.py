"""
Setup script for the MIMIC project.
"""

from setuptools import find_packages, setup

setup(
    name="mimic-readmission-predictor",
    version="0.1.0",
    description="MIMIC Critical Care Dataset Project for predicting hospital readmissions and ICU outcomes",
    author="MIMIC Project Team",
    author_email="alexanderclarke365@gmail.com",
    url="https://github.com/ACl365/mimic-readmission-predictor",
    package_dir={"": "src"},  # Tell setuptools packages are under src
    packages=find_packages(where="src"),  # Find packages in src
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
        # Add API/Dashboard specific deps here if not already covered
        "fastapi>=0.75.0,<0.76.0",
        "uvicorn>=0.17.0,<0.18.0",
        "streamlit>=1.5.0,<1.6.0",
        "requests>=2.27.0,<2.28.0",
        "pydantic>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.10.0",  # Ensure mock is here
            "black>=22.1.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.17.0",
            "mypy>=1.0",  # Updated version, ensure compatibility
            "types-PyYAML",  # Stubs for PyYAML
            "types-requests",  # Stubs for requests
            "pandas-stubs",  # Stubs for pandas
            "pip-tools>=6.0.0",  # Added for pinning dependencies
        ],
        "docs": [
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "full": [  # Optional: Combine base + dev + docs
            # You might need to list base requirements explicitly or handle this differently
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.10.0",
            "black>=22.1.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.17.0",
            "mypy>=1.0",
            "types-PyYAML",
            "types-requests",
            "pandas-stubs",  # Added mypy deps here too
            "pip-tools>=6.0.0",
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Note: Entry points need to reference the installed package structure
            "mimic-process=data.make_dataset:main",
            "mimic-features=features.build_features:main",
            "mimic-train=models.train_model:main",
            "mimic-predict=models.predict_model:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  # Corrected spelling
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
