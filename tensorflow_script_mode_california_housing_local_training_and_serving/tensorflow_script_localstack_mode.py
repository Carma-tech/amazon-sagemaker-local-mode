import os
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing # Corrected from your script's import *
import json
import time

import boto3 # Added for LocalStack configuration
import sagemaker # Main SageMaker SDK
from sagemaker.tensorflow import TensorFlow
from sagemaker.local import LocalSession # Required for local mode

# This should be defined globally or passed as in your original script
DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

# Configuration for LocalStack
LOCALSTACK_ENDPOINT_URL = os.environ.get('LOCALSTACK_ENDPOINT', 'http://localhost:4566') # Default LocalStack edge port
LOCALSTACK_REGION = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1') # Or your LocalStack region

# Output file for capturing model results
OUTPUT_FILE = os.environ.get('SAGEMAKER_OUTPUT_FILE', None)

# Dictionary to store pipeline results for Step Functions
model_results = {
    'training_job_name': f'sagemaker-training-job-{int(time.time())}',
    'endpoint_name': f'sagemaker-endpoint-{int(time.time())}',
    'model_artifact': '',
    'predictions': [],
    'target_values': [],
    'data_prep_metadata': {},
    'training_metadata': {},
    'deployment_metadata': {},
    'inference_metadata': {}
}


# --- Make sure these helper functions from your original script are available ---
def download_training_and_eval_data():
    if os.path.isfile('./data/train/x_train.npy') and \
            os.path.isfile('./data/test/x_test.npy') and \
            os.path.isfile('./data/train/y_train.npy') and \
            os.path.isfile('./data/test/y_test.npy'):
        print('Training and evaluation datasets exist. Skipping Download')
    else:
        print('Downloading training and evaluation dataset')
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)

        train_dir = os.path.join(os.getcwd(), 'data/train')
        os.makedirs(train_dir, exist_ok=True)

        test_dir = os.path.join(os.getcwd(), 'data/test')
        os.makedirs(test_dir, exist_ok=True)

        data_set = fetch_california_housing()

        X = pd.DataFrame(data_set.data, columns=data_set.feature_names)
        Y = pd.DataFrame(data_set.target)

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.33, random_state=42) # Added random_state for reproducibility

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        np.save(os.path.join(train_dir, 'x_train.npy'), x_train)
        np.save(os.path.join(test_dir, 'x_test.npy'), x_test)
        np.save(os.path.join(train_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(test_dir, 'y_test.npy'), y_test)

        print('Downloading completed')
        
    # Update model results with data preparation metadata
    model_results['data_prep_metadata'] = {
        "training_data": "s3://{bucket}/processed/training/",
        "test_data": "s3://{bucket}/processed/test/"
    }


def do_inference_on_local_endpoint(predictor):
    """Run inference using the deployed model."""
    # Load test data
    test_dir = os.path.join(os.getcwd(), 'data/test')
    x_test = np.load(os.path.join(test_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(test_dir, 'y_test.npy'))
    
    print("Running inference on local endpoint...")
    
    # Take a sample of test data for inference
    num_samples = min(10, len(x_test))
    sample_indices = np.random.choice(len(x_test), num_samples, replace=False)
    x_test_sample = x_test[sample_indices]
    y_test_sample = y_test[sample_indices]
    
    # Run inference on the sample data
    predictions = predictor.predict(x_test_sample)
    
    print(f"Received predictions for {num_samples} samples.")
    print("Sample predictions:", predictions[:5])
    print("Sample actuals:", y_test_sample[:5])
    
    # Store the predictions and target values in our results
    model_results['predictions'] = predictions
    model_results['target_values'] = y_test_sample
    model_results['inference_metadata'] = {
        "endpoint_name": predictor.endpoint_name,
        "status": "Complete",
        "num_samples": num_samples,
        "mean_prediction": float(np.mean(predictions)),
        "timestamp": time.time()
    }


def train_model():
    """Train a TensorFlow model using SageMaker LocalSession."""
    # Download the training data if needed
    download_training_and_eval_data()
    
    # Configure SageMaker for local mode
    sagemaker_session = LocalSession(
        boto_session=boto3.Session(region_name=LOCALSTACK_REGION),
    )
    
    # This can be any local directory for model output as LocalMode doesn't need S3
    # but ensures compatibility with your original script
    output_path = 'file://' + os.path.join(os.getcwd(), 'model')
    
    # Set up the estimator for a TensorFlow model
    training_job_name = f'sagemaker-training-job-{int(time.time())}'
    estimator = TensorFlow(
        entry_point='california_housing_train.py',
        source_dir='source_dir',
        role=DUMMY_IAM_ROLE,  # Doesn't matter for local mode but must be provided
        instance_count=1,
        instance_type='local',
        framework_version='2.3.1',  # Using TF 2.3 for this example
        py_version='py37',  # For TF 2.3
        output_path=output_path,
        hyperparameters={
            'batch-size': 128,
            'epochs': 10,
            # Add any other hyperparameters your model requires
        },
        sagemaker_session=sagemaker_session,
        base_job_name=training_job_name
    )
    
    # Store the training job name in our results
    model_results['training_job_name'] = training_job_name
    model_results['model_artifact'] = output_path + '/model.tar.gz'

    inputs = {'train': 'file://./data/train', 'test': 'file://./data/test'}
    estimator.fit(inputs)
    print('Completed model training.')
    
    return estimator


def deploy_model(estimator=None):
    """Deploy the trained model to a local endpoint."""
    if estimator is None:
        print("No estimator provided. Deployment cannot proceed.")
        return None
    
    print("Deploying model to local endpoint...")
    
    # Configure LocalSession with LocalStack endpoint if needed
    sagemaker_session = LocalSession(
        boto_session=boto3.Session(region_name=LOCALSTACK_REGION),
    )
    
    # Enable verbose logging to debug issues
    os.environ['SAGEMAKER_CONTAINER_LOG_LEVEL'] = '20'  # INFO
    
    # Create a unique endpoint name
    endpoint_name = f"tf-california-local-{int(time.time())}"
    
    # Deploy the model to a local endpoint
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type='local',
        endpoint_name=endpoint_name,
    )
    
    print(f"Endpoint {endpoint_name} deployed locally.")
    
    # Store the endpoint name in our results
    model_results['endpoint_name'] = endpoint_name
    model_results['deployment_metadata'] = {
        "endpoint_name": endpoint_name,
        "status": "InService",
        "timestamp": time.time()
    }
    
    return predictor


def run_inference(predictor=None):
    """Run inference on the deployed model."""
    if predictor is None:
        print("No predictor provided. Inference cannot be performed.")
        return
    
    do_inference_on_local_endpoint(predictor)


def cleanup_resources(predictor=None):
    """Clean up resources."""
    if predictor is None:
        print("No predictor provided. Nothing to clean up.")
        return
    
    print('About to delete the endpoint.')
    try:
        predictor.delete_endpoint(predictor.endpoint_name)
        print(f"Endpoint {predictor.endpoint_name} deleted successfully.")
    except Exception as e:
        print(f"Error deleting endpoint {predictor.endpoint_name}: {e}")


def save_model_results():
    """Save the model results to a JSON file if OUTPUT_FILE is specified."""
    if OUTPUT_FILE:
        try:
            print(f"Saving model results to {OUTPUT_FILE}")
            # Ensure any placeholder values with {bucket} are properly formatted
            bucket_name = os.environ.get('S3_BUCKET_NAME', 'default-bucket')
            for key, value in model_results.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, str) and '{bucket}' in v:
                            model_results[key][k] = v.format(bucket=bucket_name)
                        
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(model_results, f, indent=2)
            print(f"Successfully saved model results to {OUTPUT_FILE}")
        except Exception as e:
            print(f"Error saving model results to {OUTPUT_FILE}: {e}")

def main_with_localstack(args):
    """Main function to handle different execution modes."""
    # Configure boto3 for LocalStack if needed
    print(f"Configuring LocalStack for auxiliary AWS services: {LOCALSTACK_ENDPOINT_URL}")
    
    # Parse command-line arguments
    if args.train_only:
        print("Running training only mode")
        train_model()
    elif args.deploy_only:
        print("Running deployment only mode")
        predictor = deploy_model()
        print(f"Model deployed to endpoint: {predictor.endpoint_name}")
    elif args.inference_only:
        print("Running inference only mode")
        # In a real scenario, we would get the endpoint name from somewhere
        endpoint_name = os.environ.get('ENDPOINT_NAME', 'dummy-endpoint')
        print(f"Would run inference on endpoint: {endpoint_name}")
        print("Skipping actual inference as no real endpoint exists in this mode")
    elif args.cleanup_only:
        print("Running cleanup only mode")
        # In a real scenario, we would get the endpoint name from somewhere
        endpoint_name = os.environ.get('ENDPOINT_NAME', 'dummy-endpoint')
        print(f"Would clean up endpoint: {endpoint_name}")
        print("Skipping actual cleanup as no real endpoint exists in this mode")
    else:
        # Full pipeline
        print("Running full pipeline: train -> deploy -> inference -> cleanup")
        estimator = train_model()
        predictor = deploy_model(estimator)
        run_inference(predictor)
        # cleanup_resources(predictor)
    
    # Save the model results to a file if specified
    save_model_results()


if __name__ == "__main__":
    # Set up command-line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='TensorFlow SageMaker Local Mode with LocalStack')
    parser.add_argument('--train-only', action='store_true', help='Run only the training phase')
    parser.add_argument('--deploy-only', action='store_true', help='Run only the deployment phase')
    parser.add_argument('--inference-only', action='store_true', help='Run only the inference phase')
    parser.add_argument('--cleanup-only', action='store_true', help='Run only the cleanup phase')
    parser.add_argument('--output-file', help='Path to save model results as JSON')
    parser.add_argument('--localstack-mode', action='store_true', help='Run in LocalStack mode')
    
    args = parser.parse_args()
    
    # Set output file from args if provided
    if args.output_file:
        OUTPUT_FILE = args.output_file
        print(f"Will save model results to {OUTPUT_FILE}")
    
    main_with_localstack(args)