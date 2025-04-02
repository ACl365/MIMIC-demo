# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# We'll create this file next
COPY ./requirements.txt /app/requirements.txt

# Install system dependencies required for building some Python packages (like gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# --no-cache-dir: Disables the cache to keep the image size smaller
# --upgrade pip: Ensures pip is up-to-date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (API and source code) into the container
# Copy the 'api' directory content to /app/api
COPY ./ /app/api/
# Copy the 'src' directory content to /app/src
# This assumes the model loading/prediction logic is in 'src'
COPY ../src /app/src
# Copy the config files
COPY ../configs /app/configs
# Copy the trained model(s) - Adjust path if necessary based on config.yaml
# Example: Assuming model is saved in 'models/' directory at the root
COPY ../models /app/models
# Copy the processed data if needed for preprocessing steps during inference
# Example: Assuming some lookup tables or scalers are needed from 'data/processed'
# COPY ../data/processed /app/data/processed

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable (optional, can be useful)
ENV MODULE_NAME="api.main"
ENV VARIABLE_NAME="app"

# Run main.py when the container launches
# Use uvicorn to run the FastAPI application defined in api/main.py
# --host 0.0.0.0 makes it accessible from outside the container
# --port 8000 matches the EXPOSE instruction
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
