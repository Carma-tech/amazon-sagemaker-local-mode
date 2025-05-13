#!/usr/bin/env python3
"""
SageMaker Local Mode - Real Model Inference

This script:
1. Performs inference using models deployed to SageMaker local endpoints
2. Can also load model artifacts directly for inference without an endpoint
3. Works with Step Functions orchestration via LocalStack
4. Supports custom test cases and real data
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
import numpy as np
import pandas as pd
import boto3
import pickle
import tempfile
import tarfile
import requests
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.tensorflow import TensorFlowPredictor
from sagemaker.tensorflow import TensorFlowModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sagemaker-local-inference')

# Configuration
LOCALSTACK_ENDPOINT = 'http://localhost:4567'
REGION = 'us-east-1'
DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'
MODEL_OUTPUTS_FILE = 'model_outputs.json'


def load_model_outputs(file_path=None):
    """Load model outputs from a JSON file."""
    if not file_path:
        file_path = MODEL_OUTPUTS_FILE
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Model outputs file not found: {file_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading model outputs: {str(e)}")
        return {}


def get_boto3_client(service_name):
    """Create a boto3 client for LocalStack with S3 acceleration disabled."""
    return boto3.client(
        service_name,
        endpoint_url=LOCALSTACK_ENDPOINT,
        region_name=REGION,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        config=boto3.session.Config(
            s3={'addressing_style': 'path', 'use_accelerate_endpoint': False},
            signature_version='s3v4'
        )
    )


def get_test_data():
    """Get test data from the California Housing dataset."""
    logger.info("Loading California Housing test data")
    dataset = fetch_california_housing()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.DataFrame(dataset.target)
    
    # Split the dataset to get test data
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Test data shape: {X_test.shape}")
    return X_test, y_test


def check_endpoint_status(endpoint_name):
    """Check if the SageMaker local endpoint is running."""
    try:
        # Check using docker
        cmd = f"docker ps --filter name=sagemaker-{endpoint_name}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        if endpoint_name in result.stdout:
            logger.info(f"Endpoint {endpoint_name} is running")
            return True
        else:
            logger.warning(f"Endpoint {endpoint_name} is not running")
            # List running containers
            logger.info("Running containers:")
            subprocess.run(["docker", "ps"], capture_output=False)
            return False
    except Exception as e:
        logger.error(f"Error checking endpoint status: {str(e)}")
        return False


def get_endpoint_predictor(endpoint_name):
    """Get a predictor for a SageMaker endpoint."""
    try:
        if not check_endpoint_status(endpoint_name):
            logger.warning("Endpoint is not running, predictions may not work")
        
        # Create a predictor for the endpoint
        predictor = TensorFlowPredictor(
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        logger.info(f"Created predictor for endpoint: {endpoint_name}")
        return predictor
    except Exception as e:
        logger.error(f"Error creating predictor: {str(e)}")
        return None


def invoke_endpoint_via_boto3(endpoint_name, test_data):
    """Invoke a SageMaker endpoint using boto3."""
    try:
        # Create a SageMaker runtime client
        runtime = boto3.client('sagemaker-runtime', endpoint_url=LOCALSTACK_ENDPOINT, region_name=REGION)
        
        # Format the payload
        payload = {"instances": test_data.values.tolist()}
        
        # Invoke the endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse the response
        result = json.loads(response['Body'].read().decode())
        logger.info(f"Got predictions via boto3: {result[:5]}")
        return result
    except Exception as e:
        logger.error(f"Error invoking endpoint via boto3: {str(e)}")
        return None


def invoke_endpoint_direct(endpoint_name, test_data):
    """Invoke the endpoint directly using HTTP."""
    try:
        # Get container port using docker inspect
        cmd = f"docker ps -q --filter name=sagemaker-{endpoint_name} | xargs -I {{}} docker port {{}} 8080/tcp"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        port_mapping = result.stdout.strip().split(':')[-1] if result.stdout else '8080'
        
        logger.info(f"Using local endpoint port: {port_mapping}")
        
        # Direct HTTP request
        url = f"http://localhost:{port_mapping}/invocations"
        headers = {"Content-Type": "application/json"}
        
        # Format the payload
        payload = {"instances": test_data.values.tolist()}
        
        # Send the request
        logger.info(f"Sending direct HTTP request to {url}")
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Got predictions via HTTP: {result[:5] if isinstance(result, list) else 'non-list result'}")
            
            # If the result is a dictionary with 'predictions', extract it
            if isinstance(result, dict) and 'predictions' in result:
                result = result['predictions']
                
            return result
        else:
            logger.error(f"HTTP error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error invoking endpoint directly: {str(e)}")
        return None


def load_model_from_artifact(model_artifact):
    """Load a model directly from a model artifact."""
    try:
        logger.info(f"Loading model from artifact: {model_artifact}")
        
        # Handle S3 paths
        if model_artifact.startswith('s3://'):
            # Parse bucket and key
            s3_path = model_artifact.replace('s3://', '')
            bucket_name = s3_path.split('/')[0]
            key = '/'.join(s3_path.split('/')[1:])
            
            # Download the model artifact
            s3_client = get_boto3_client('s3')
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                s3_client.download_file(bucket_name, key, tmp.name)
                model_artifact = tmp.name
                logger.info(f"Downloaded model from S3 to: {model_artifact}")
        
        # Handle file:// prefix
        if model_artifact.startswith('file://'):
            model_artifact = model_artifact[7:]
        
        # Create a temp directory for extraction
        with tempfile.TemporaryDirectory() as extract_dir:
            logger.info(f"Extracting model to: {extract_dir}")
            
            # Extract the model
            with tarfile.open(model_artifact) as tar:
                tar.extractall(path=extract_dir)
            
            # Check for TensorFlow saved model
            saved_model_dir = os.path.join(extract_dir, 'model', '1')
            if os.path.exists(os.path.join(saved_model_dir, 'saved_model.pb')):
                try:
                    import tensorflow as tf
                    model = tf.saved_model.load(saved_model_dir)
                    logger.info("Loaded TensorFlow SavedModel")
                    return model
                except Exception as e:
                    logger.error(f"Error loading TensorFlow model: {str(e)}")
            
            # Check for pickle files
            pickle_files = list(Path(extract_dir).glob('**/*.pkl'))
            if pickle_files:
                try:
                    with open(pickle_files[0], 'rb') as f:
                        model = pickle.load(f)
                    logger.info(f"Loaded pickled model: {type(model)}")
                    return model
                except Exception as e:
                    logger.error(f"Error loading pickle model: {str(e)}")
            
            logger.error("No suitable model found in artifact")
            return None
    except Exception as e:
        logger.error(f"Error loading model from artifact: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def predict_with_model(model, test_data):
    """Make predictions using a loaded model."""
    try:
        logger.info(f"Making predictions with model of type: {type(model)}")
        
        # Try different prediction methods depending on the model type
        if hasattr(model, 'predict'):
            # Standard sklearn-like interface
            predictions = model.predict(test_data)
            logger.info(f"Made predictions using model.predict: {predictions[:5]}")
            return predictions
        elif hasattr(model, 'signatures'):
            # TensorFlow SavedModel
            import tensorflow as tf
            predict_fn = model.signatures["serving_default"]
            
            # Convert to tensor
            tensor_data = tf.convert_to_tensor(test_data.values, dtype=tf.float32)
            
            # Get the input name
            input_name = list(predict_fn.structured_input_signature[1].keys())[0]
            
            # Make prediction
            result = predict_fn(**{input_name: tensor_data})
            
            # Get the output tensor
            output_name = list(result.keys())[0]
            predictions = result[output_name].numpy()
            
            logger.info(f"Made predictions using TensorFlow model: {predictions[:5]}")
            return predictions
        else:
            logger.error(f"Model doesn't have a known prediction interface: {dir(model)}")
            return None
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return None


def compare_predictions(predictions, actuals):
    """Compare predictions with actual values and compute metrics."""
    try:
        if predictions is None or actuals is None:
            logger.warning("Cannot compare predictions: predictions or actuals are None")
            return None
        
        # Convert to numpy arrays
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        if not isinstance(actuals, np.ndarray):
            actuals = np.array(actuals)
        
        # Flatten if needed
        predictions = predictions.flatten()
        actuals = actuals.flatten()
        
        # Ensure same length for comparison
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]
        
        # Compute metrics
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Display comparison
        logger.info("\n=== Model Predictions vs. Actual Values ===")
        logger.info("Prediction      Actual          Diff           ")
        logger.info("---------------------------------------------")
        for i in range(min(10, min_len)):
            diff = abs(float(predictions[i]) - float(actuals[i]))
            logger.info(f"{float(predictions[i]):<15.4f} {float(actuals[i]):<15.4f} {diff:<15.4f}")
        
        logger.info("\nModel Performance Metrics:")
        logger.info(f"Mean Squared Error: {mse:.6f}")
        logger.info(f"R-squared: {r2:.6f}")
        
        return {
            'mse': float(mse),
            'r2': float(r2),
            'sample_predictions': predictions[:10].tolist(),
            'sample_actuals': actuals[:10].tolist()
        }
    except Exception as e:
        logger.error(f"Error comparing predictions: {str(e)}")
        return None


def run_inference(endpoint_name=None, model_artifact=None, test_data=None, method='direct'):
    """Run inference using a deployed model or model artifact."""
    # Load model outputs if needed
    model_outputs = load_model_outputs()
    
    # Determine endpoint name
    if not endpoint_name and 'endpoint_name' in model_outputs:
        endpoint_name = model_outputs['endpoint_name']
        logger.info(f"Using endpoint from model outputs: {endpoint_name}")
    
    # Determine model artifact
    if not model_artifact and 'model_artifact' in model_outputs:
        model_artifact = model_outputs['model_artifact']
        logger.info(f"Using model artifact from model outputs: {model_artifact}")
    
    # Load test data if not provided
    if test_data is None:
        X_test, y_test = get_test_data()
        test_data = X_test
        actuals = y_test
    else:
        # Assume test_data is a tuple (X_test, y_test)
        test_data, actuals = test_data
    
    predictions = None
    
    # Try endpoint inference first if endpoint is provided
    if endpoint_name:
        logger.info(f"Running inference with endpoint: {endpoint_name}")
        
        if method == 'boto3':
            # Use boto3 client
            predictions = invoke_endpoint_via_boto3(endpoint_name, test_data)
        elif method == 'predictor':
            # Use SageMaker predictor
            predictor = get_endpoint_predictor(endpoint_name)
            if predictor:
                try:
                    payload = {"instances": test_data.values.tolist()}
                    predictions = predictor.predict(payload)
                    logger.info(f"Got predictions via predictor: {predictions[:5] if isinstance(predictions, list) else 'non-list result'}")
                except Exception as e:
                    logger.error(f"Error using predictor: {str(e)}")
        else:
            # Use direct HTTP
            predictions = invoke_endpoint_direct(endpoint_name, test_data)
    
    # If endpoint inference failed or no endpoint, try direct model inference
    if predictions is None and model_artifact:
        logger.info("Endpoint inference failed or not available, trying direct model inference")
        model = load_model_from_artifact(model_artifact)
        if model:
            predictions = predict_with_model(model, test_data)
    
    # Compare predictions with actual values
    if predictions is not None:
        metrics = compare_predictions(predictions, actuals)
        return predictions, metrics
    else:
        logger.error("Failed to get predictions from any method")
        return None, None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run inference with SageMaker local models')
    parser.add_argument('--endpoint-name', type=str, help='Name of the SageMaker endpoint')
    parser.add_argument('--model-artifact', type=str, help='Path to model artifact')
    parser.add_argument('--method', choices=['direct', 'boto3', 'predictor'], default='direct',
                      help='Method to use for endpoint inference')
    parser.add_argument('--output-file', type=str, help='File to save inference results')
    
    args = parser.parse_args()
    
    # Run inference
    predictions, metrics = run_inference(
        endpoint_name=args.endpoint_name,
        model_artifact=args.model_artifact,
        method=args.method
    )
    
    # Save results if requested
    if args.output_file and metrics:
        try:
            with open(args.output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved inference results to {args.output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    logger.info("Inference completed")
    
    # Return predictions for potential further processing
    return predictions


def lambda_handler(event, context=None):
    """AWS Lambda handler for inference."""
    try:
        logger.info(f"Received Lambda event: {event}")
        
        # Extract parameters from the event
        endpoint_name = event.get('endpoint_name')
        endpoint_url = event.get('endpoint_url')
        model_path = event.get('model_path')
        data = event.get('data')
        
        # If data is a dictionary with a specific format, load the data accordingly
        if isinstance(data, dict) and 'type' in data:
            if data['type'] == 'california_housing':
                from sklearn.datasets import fetch_california_housing
                dataset = fetch_california_housing()
                X = dataset.data
                sample_size = data.get('sample_size', 10)
                X_sample = X[:sample_size]
                data = X_sample.tolist()
        
        # Configure inference options
        inference_options = {}
        if endpoint_name:
            inference_options['endpoint_name'] = endpoint_name
        if endpoint_url:
            inference_options['endpoint_url'] = endpoint_url
        if model_path:
            inference_options['model_path'] = model_path
        if data:
            inference_options['input_data'] = json.dumps({'instances': data})
            inference_options['content_type'] = 'application/json'
        
        # Run inference
        result = run_inference(**inference_options)
        
        return {
            'statusCode': 200,
            'body': {
                'predictions': result.tolist() if isinstance(result, np.ndarray) else result,
                'result_type': str(type(result))
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
