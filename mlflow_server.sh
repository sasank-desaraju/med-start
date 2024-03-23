#!/bin/bash

env_path=/blue/banks/sasank.desaraju/conda_envs/envs/mlflow/bin
export PATH=$env_path:$PATH
export MLFLOW_TRACKING_URI=file:///home/sasank.desaraju/mlflow/

# This starts an MLFlow server
mlflow server --backend-store-uri file:///home/sasank.desaraju/mlflow --default-artifact-root file:///home/sasank.desaraju/mlflow/artifacts --host 0.0.0.0 --port 5000
# mlflow server --backend-store-uri file:///home/sasank.desaraju/mlflow --default-artifact-root file:///home/sasank.desaraju/mlflow/artifacts --host 0.0.0.0 --port 5000
# mlflow server --host 0.0.0.0 --port 5000
