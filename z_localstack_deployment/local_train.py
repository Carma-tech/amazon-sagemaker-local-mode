#!/usr/bin/env python3
"""
Local Model Training for SageMaker Integration

This script:
1. Trains a real ML model using scikit-learn
2. Saves the model in a format compatible with SageMaker
3. Creates tarball artifacts that can be used for deployment
"""

import os
import json
import time
import logging
import argparse
import tarfile
import tempfile
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('local-train')

# Model output file
MODEL_OUTPUT_DIR = 'model'
MODEL_ARTIFACT_PATH = os.path.join(MODEL_OUTPUT_DIR, 'model.tar.gz')
MODEL_OUTPUTS_FILE = 'model_outputs.json'


def load_california_housing_data():
    """Load and prepare the California Housing dataset."""
    logger.info("Loading California Housing dataset")
    dataset = fetch_california_housing()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.DataFrame(dataset.target)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type='random_forest'):
    """Train a machine learning model."""
    logger.info(f"Training {model_type} model")
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    else:
        # Default to RandomForest if the model type is not recognized
        logger.warning(f"Unknown model type '{model_type}', defaulting to random forest")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and compute metrics."""
    logger.info("Evaluating model")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Compute metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    logger.info(f"Mean Squared Error: {mse:.4f}")
    logger.info(f"R-squared: {r2:.4f}")
    
    # Show sample predictions
    logger.info("\nSample predictions vs. actual values:")
    logger.info("Prediction      Actual          Diff           ")
    logger.info("---------------------------------------------")
    for i in range(min(5, len(predictions))):
        diff = abs(float(predictions[i]) - float(y_test.iloc[i, 0]))
        logger.info(f"{float(predictions[i]):<15.4f} {float(y_test.iloc[i, 0]):<15.4f} {diff:<15.4f}")
    
    metrics = {
        'mse': float(mse),
        'r2': float(r2),
        'sample_predictions': predictions[:10].tolist(),
        'sample_actuals': y_test.iloc[:10, 0].tolist()
    }
    
    return predictions, metrics


def create_model_artifact(model, output_dir=MODEL_OUTPUT_DIR):
    """Create a model artifact for deployment."""
    logger.info("Creating model artifact")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Temporary directory for model files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save the model to a pickle file
        model_file = os.path.join(tmp_dir, 'model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {model_file}")
        
        # Create a tarball of the model
        tarball_path = os.path.join(output_dir, 'model.tar.gz')
        with tarfile.open(tarball_path, 'w:gz') as tar:
            tar.add(model_file, arcname='model.pkl')
        logger.info(f"Created model tarball at {tarball_path}")
        
        # Full path with file:// prefix for SageMaker compatibility
        full_path = f"file://{os.path.abspath(tarball_path)}"
        return full_path


def save_model_outputs(model_outputs, file_path=None):
    """Save model training outputs to a JSON file."""
    if not file_path:
        file_path = MODEL_OUTPUTS_FILE
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(model_outputs, f, indent=2)
    
    logger.info(f"Saved model outputs to {file_path}")
    return model_outputs


def main(args=None):
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train a local model for SageMaker integration')
    parser.add_argument('--model-type', type=str, default='random_forest', 
                        help='Type of model to train (random_forest)')
    parser.add_argument('--output-dir', type=str, default=MODEL_OUTPUT_DIR,
                        help='Directory to save model artifacts')
    
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training workflow
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        # Train model
        model = train_model(X_train, y_train, model_type=args.model_type)
        
        # Evaluate model
        predictions, metrics = evaluate_model(model, X_test, y_test)
        
        # Create model artifact
        model_artifact = create_model_artifact(model, output_dir=args.output_dir)
        
        # Generate unique names for this training run
        timestamp = int(time.time())
        training_job_name = f'local-training-job-{timestamp}'
        
        # Store outputs
        model_outputs = {
            'training_job_name': training_job_name,
            'model_artifact': model_artifact,
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        # Save outputs
        save_model_outputs(model_outputs)
        
        logger.info("Training completed successfully")
        return model_outputs
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def lambda_handler(event, context=None):
    """AWS Lambda handler for training."""
    try:
        logger.info(f"Received Lambda event: {event}")
        
        # Extract parameters from the event
        model_type = event.get('model_type', 'random_forest')
        hyperparameters = event.get('hyperparameters', {})
        dataset = event.get('dataset', 'california_housing')
        output_dir = event.get('output_dir', MODEL_OUTPUT_DIR)
        
        # Prepare args for the main function
        args = []
        if model_type:
            args.extend(['--model-type', model_type])
        if output_dir:
            args.extend(['--output-dir', output_dir])
        
        # Run the training
        result = main(args)
        
        return {
            'statusCode': 200,
            'body': {
                'training_job_name': result['training_job_name'],
                'model_artifact': result['model_artifact'],
                'metrics': result['metrics'],
                'endpoint_name': f"california-housing-{int(time.time())}"
            }
        }
    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': {
                'error': str(e)
            }
        }

if __name__ == "__main__":
    main()
