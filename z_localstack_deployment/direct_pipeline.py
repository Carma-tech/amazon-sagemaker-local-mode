#!/usr/bin/env python3
"""
Direct Integration Pipeline for SageMaker Local Mode with LocalStack

This script demonstrates a simplified approach to integrate SageMaker local mode
with LocalStack for storage and state management, bypassing the limitations
of Lambda functions in LocalStack's free tier.

This provides a conceptual model of how you would use AWS Step Functions
to orchestrate SageMaker jobs in the real AWS environment.
"""

import boto3
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

# LocalStack endpoint
LOCALSTACK_ENDPOINT = os.environ.get(
    'LOCALSTACK_ENDPOINT_URL', 'http://localhost:4567')

# Your SageMaker local script path
SAGEMAKER_SCRIPT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "tensorflow_script_mode_california_housing_local_training_and_serving",
    "tensorflow_script_localstack_mode.py"
))
print(SAGEMAKER_SCRIPT)

# Default region
REGION = 'us-east-1'

# Create a unique identifier for this run
RUN_ID = str(uuid.uuid4())[:8]

# Resource naming
BUCKET_NAME = f"ml-pipeline-data-{RUN_ID}"

# Create a boto3 session with our config


def get_boto3_session():
    """Create a boto3 session configured for LocalStack."""
    return boto3.Session(
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name=REGION
    )


def get_boto3_resource(service_name):
    """Create a boto3 resource for LocalStack."""
    session = get_boto3_session()
    return session.resource(
        service_name,
        endpoint_url=LOCALSTACK_ENDPOINT,
        config=boto3.session.Config(
            s3={'addressing_style': 'path', 'use_accelerate_endpoint': False},
            signature_version='s3v4'
        )
    )


def get_boto3_client(service_name):
    """Create a boto3 client for LocalStack."""
    session = get_boto3_session()
    return session.client(
        service_name,
        endpoint_url=LOCALSTACK_ENDPOINT,
        config=boto3.session.Config(
            s3={'addressing_style': 'path', 'use_accelerate_endpoint': False},
            signature_version='s3v4'
        )
    )


def create_s3_bucket():
    """Create an S3 bucket for storing ML data and models."""
    print(f"Creating S3 bucket: {BUCKET_NAME}")

    try:
        s3 = get_boto3_resource('s3')

        # Check if bucket exists first
        buckets = [bucket.name for bucket in s3.buckets.all()]
        if BUCKET_NAME in buckets:
            print(f"Bucket {BUCKET_NAME} already exists, continuing...")
            return

        # Create bucket
        bucket = s3.create_bucket(Bucket=BUCKET_NAME)
        print(f"S3 bucket created: {BUCKET_NAME}")

        # Create directory structure in the bucket
        paths = ['raw', 'processed/training',
                 'processed/test', 'models', 'metrics']
        for path in paths:
            s3.Object(BUCKET_NAME, f"{path}/.placeholder").put(Body="")
            print(f"Created path s3://{BUCKET_NAME}/{path}/")

    except Exception as e:
        print(f"Error creating S3 bucket: {str(e)}")
        # Try alternative method if S3 Accelerate error
        if "S3 Accelerate" in str(e):
            print("S3 Accelerate error detected - using alternative method")
            try:
                s3_client = get_boto3_client('s3')
                s3_client.create_bucket(Bucket=BUCKET_NAME)
                print(f"S3 bucket created (alternative method): {BUCKET_NAME}")

                # Create directory structure
                for path in ['raw', 'processed/training', 'processed/test', 'models', 'metrics']:
                    s3_client.put_object(
                        Bucket=BUCKET_NAME, Key=f"{path}/.placeholder", Body="")
                    print(f"Created path s3://{BUCKET_NAME}/{path}/")
            except Exception as inner_e:
                print(f"Still failed to create bucket: {str(inner_e)}")
                raise inner_e
        else:
            raise e


def prepare_data():
    """Prepare data for model training.

    In a real AWS environment, this would be an AWS Step Functions Task state
    that calls a Lambda function or an AWS Batch job.
    """
    print("\n==== STEP 1: DATA PREPARATION ====")

    # In this simplified direct approach, we're just simulating the data preparation
    # Normally this would extract data from S3, process it, and upload it back to S3

    print("Preparing data for model training...")
    time.sleep(1)  # Simulate some processing time

    # Upload some metadata to S3
    s3 = get_boto3_client('s3')
    metadata = {
        "dataset": "California Housing",
        "timestamp": time.time(),
        "preprocessing": "StandardScaler",
        "split_ratio": 0.67
    }

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key="processed/metadata.json",
        Body=json.dumps(metadata),
        ContentType="application/json"
    )

    print("Data preparation complete! Metadata saved to S3.")
    return metadata


def train_model(data_metadata):
    """Train the model using SageMaker local mode.

    In a real AWS environment, this would be a Step Functions Task state
    that starts a SageMaker training job.
    """
    print("\n==== STEP 2: MODEL TRAINING ====\n")

    # Configure environment variables for the SageMaker local script
    env = os.environ.copy()
    env['TF_EPOCHS'] = '10'
    env['TF_BATCH_SIZE'] = '32'
    env['TF_LEARNING_RATE'] = '0.01'
    env['S3_BUCKET_NAME'] = BUCKET_NAME
    env['S3_ENDPOINT_URL'] = LOCALSTACK_ENDPOINT

    # Set the correct directory for the script
    script_dir = os.path.dirname(SAGEMAKER_SCRIPT)

    print(f"Running SageMaker local mode script: {SAGEMAKER_SCRIPT}")
    try:
        # Run the SageMaker local script (just the training part)
        # Normally this would be a SageMaker training job started via the AWS SDK
        result = subprocess.run(
            ["python", os.path.basename(SAGEMAKER_SCRIPT), "--train-only"],
            capture_output=True,
            text=True,
            env=env,
            cwd=script_dir
        )

        if result.returncode != 0:
            print(f"Error running SageMaker script: {result.stderr}")
            raise Exception("Training failed")

        # Log some of the output
        print("\nTraining output:")
        output_lines = result.stdout.split('\n')
        for line in output_lines[-20:]:  # Show the last 20 lines
            if line.strip():
                print(f"  {line}")

        # Upload training results to S3
        model_metadata = {
            "timestamp": time.time(),
            "data_metadata": data_metadata,
            "hyperparameters": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.01
            },
            "status": "complete"
        }

        s3 = get_boto3_client('s3')
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key="models/model_metadata.json",
            Body=json.dumps(model_metadata),
            ContentType="application/json"
        )

        print("Model training complete! Metadata saved to S3.")
        return model_metadata

    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise


def deploy_model(model_metadata):
    """Deploy the model using SageMaker local mode.

    In a real AWS environment, this would be a Step Functions Task state
    that deploys a SageMaker model to an endpoint.
    """
    print("\n==== STEP 3: MODEL DEPLOYMENT ====\n")

    # Configure environment variables for the SageMaker local script
    env = os.environ.copy()
    env['S3_BUCKET_NAME'] = BUCKET_NAME
    env['S3_ENDPOINT_URL'] = LOCALSTACK_ENDPOINT

    print(
        f"Running SageMaker local mode script to deploy model: {SAGEMAKER_SCRIPT}")
    try:
        # Run the SageMaker local script (just the deployment part)
        # Normally this would be a SageMaker deployment via the AWS SDK
        script_dir = os.path.dirname(SAGEMAKER_SCRIPT)
        result = subprocess.run(
            ["python", os.path.basename(SAGEMAKER_SCRIPT), "--deploy-only"],
            capture_output=True,
            text=True,
            env=env,
            cwd=script_dir
        )

        if result.returncode != 0:
            print(f"Error deploying model: {result.stderr}")
            raise Exception("Deployment failed")

        # Log some of the output
        print("\nDeployment output:")
        output_lines = result.stdout.split('\n')
        for line in output_lines[-20:]:  # Show the last 20 lines
            if line.strip():
                print(f"  {line}")

        # Upload deployment results to S3
        deployment_metadata = {
            "timestamp": time.time(),
            "model_metadata": model_metadata,
            "endpoint_name": f"tensorflow-local-endpoint-{RUN_ID}",
            "status": "in-service"
        }

        s3 = get_boto3_client('s3')
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key="models/deployment_metadata.json",
            Body=json.dumps(deployment_metadata),
            ContentType="application/json"
        )

        print("Model deployment complete! Metadata saved to S3.")
        return deployment_metadata

    except Exception as e:
        print(f"Error in model deployment: {str(e)}")
        raise


def run_inference(deployment_metadata):
    """Run inference on the deployed model.

    In a real AWS environment, this would be a Step Functions Task state
    that invokes a SageMaker endpoint.
    """
    print("\n==== STEP 4: MODEL INFERENCE ====\n")

    # Configure environment variables for the SageMaker local script
    env = os.environ.copy()
    env['S3_BUCKET_NAME'] = BUCKET_NAME
    env['S3_ENDPOINT_URL'] = LOCALSTACK_ENDPOINT
    env['ENDPOINT_NAME'] = deployment_metadata.get(
        'endpoint_name', f"tensorflow-local-endpoint-{RUN_ID}")

    print(f"Running inference on deployed model...")
    try:
        # Run the SageMaker local script (just the inference part)
        # Normally this would be a SageMaker inference via the AWS SDK
        script_dir = os.path.dirname(SAGEMAKER_SCRIPT)
        result = subprocess.run(
            ["python", os.path.basename(SAGEMAKER_SCRIPT), "--inference-only"],
            capture_output=True,
            text=True,
            env=env,
            cwd=script_dir
        )

        if result.returncode != 0:
            print(f"Error running inference: {result.stderr}")
            raise Exception("Inference failed")

        # Log the output
        print("\nInference output:")
        output_lines = result.stdout.split('\n')
        for line in output_lines[-20:]:  # Show the last 20 lines
            if line.strip():
                print(f"  {line}")

        # Upload inference results to S3
        inference_metadata = {
            "timestamp": time.time(),
            "deployment_metadata": deployment_metadata,
            "samples_processed": 10,
            "status": "complete"
        }

        s3 = get_boto3_client('s3')
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key="metrics/inference_results.json",
            Body=json.dumps(inference_metadata),
            ContentType="application/json"
        )

        print("Inference complete! Results saved to S3.")
        return inference_metadata

    except Exception as e:
        print(f"Error in inference: {str(e)}")
        raise


def cleanup(deployment_metadata):
    """Clean up resources.

    In a real AWS environment, this would be a Step Functions Task state
    that deletes a SageMaker endpoint.
    """
    print("\n==== STEP 5: CLEANUP ====\n")

    # Configure environment variables for the SageMaker local script
    env = os.environ.copy()
    env['S3_BUCKET_NAME'] = BUCKET_NAME
    env['S3_ENDPOINT_URL'] = LOCALSTACK_ENDPOINT
    env['ENDPOINT_NAME'] = deployment_metadata.get(
        'endpoint_name', f"tensorflow-local-endpoint-{RUN_ID}")

    print(f"Cleaning up resources...")
    try:
        # Run the SageMaker local script (just the cleanup part)
        # Normally this would be a SageMaker deletion via the AWS SDK
        script_dir = os.path.dirname(SAGEMAKER_SCRIPT)
        result = subprocess.run(
            ["python", os.path.basename(SAGEMAKER_SCRIPT), "--cleanup-only"],
            capture_output=True,
            text=True,
            env=env,
            cwd=script_dir
        )

        if result.returncode != 0:
            print(f"Error during cleanup: {result.stderr}")
            raise Exception("Cleanup failed")

        # Log the output
        print("\nCleanup output:")
        output_lines = result.stdout.split('\n')
        for line in output_lines[-10:]:  # Show the last 10 lines
            if line.strip():
                print(f"  {line}")

        print("Cleanup complete!")

    except Exception as e:
        print(f"Error in cleanup: {str(e)}")
        print("You may need to manually clean up resources.")


def run_unified_workflow(data_metadata):
    """Run the entire workflow in one unified process to ensure model artifacts persist between steps.

    This is a compromise approach that still demonstrates the Step Functions concept
    but bypasses the technical limitations of LocalStack's free tier and SageMaker local mode.
    """
    print("\n==== EXECUTING UNIFIED TRAINING AND DEPLOYMENT ====\n")

    # Configure environment variables for the SageMaker local script
    env = os.environ.copy()
    env['TF_EPOCHS'] = '10'
    env['TF_BATCH_SIZE'] = '32'
    env['TF_LEARNING_RATE'] = '0.01'
    env['S3_BUCKET_NAME'] = BUCKET_NAME
    env['S3_ENDPOINT_URL'] = LOCALSTACK_ENDPOINT

    # Set the correct directory for the script
    script_dir = os.path.dirname(SAGEMAKER_SCRIPT)

    # Run the full SageMaker pipeline (train and deploy in one step)
    print(f"Running full SageMaker local mode workflow: {SAGEMAKER_SCRIPT}")
    print("DEBUG: Starting subprocess at " + time.strftime("%H:%M:%S"))

    # Run the command without capturing output so we can see it in real time
    try:
        # Run the script directly without capturing output
        print("DEBUG: Running with real-time output (this may take several minutes)...")
        # Don't use capture_output so we can see progress in real-time
        result = subprocess.run(
            ["python", os.path.basename(SAGEMAKER_SCRIPT)],  # Run the script without extra arguments
            env=env,
            cwd=script_dir,
            check=False,  # Don't raise an exception on non-zero exit
            stdout=None,  # Allow output to show in terminal
            stderr=None,  # Allow errors to show in terminal
            timeout=600   # Set a 10-minute timeout
        )
        print(f"DEBUG: Subprocess completed with exit code: {result.returncode}")
        if result.returncode != 0:
            raise Exception(f"SageMaker script failed with exit code {result.returncode}")
            
        # Since we're not capturing output, we need a different way to know what happened
        print("Execution complete. Checking for endpoint in SageMaker...")
        # Store dummy output for later processing
        dummy_output = "Execution complete. Check Docker logs for details."
    except subprocess.TimeoutExpired:
        print("DEBUG: Subprocess timed out after 5 minutes!")
        print("\nWARNING: The SageMaker script is taking too long to execute.")
        print("This could be due to Docker pulling the TensorFlow container image,")
        print("which can take several minutes the first time.")
        print("\nPlease check if Docker is running and has enough resources allocated.")
        print("\nTrying a direct docker pull to see if that's the issue...")

        # Try to pull the TensorFlow container directly to see if that's the issue
        try:
            pull_result = subprocess.run(
                ["docker", "pull", "tensorflow/serving:latest"],
                check=False,
                timeout=60
            )
            if pull_result.returncode == 0:
                print(
                    "Successfully pulled TensorFlow container. Docker seems to be working.")
            else:
                print(
                    "Failed to pull TensorFlow container. Docker might have connectivity issues.")
        except Exception as e:
            print(f"Error running docker pull: {str(e)}")

        raise Exception("SageMaker script execution timed out")

    # Since we're not capturing output anymore, we don't need to log it here
    # The output is already shown in real-time in the terminal
    print("\nSageMaker execution completed.")
    
    # Check if Docker is running properly by listing containers
    print("Checking Docker status...")
    try:
        docker_result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if docker_result.returncode == 0:
            print("Docker is running properly.")
            # Check if any SageMaker containers are running
            if "sagemaker" in docker_result.stdout.lower():
                print("SageMaker containers are active.")
        else:
            print("Docker might be having issues:")
            print(docker_result.stderr)
    except Exception as e:
        print(f"Error checking Docker status: {str(e)}")
    
    # We don't need to check result.returncode here as we did it above

    # Record the results in S3
    s3 = get_boto3_client('s3')
    execution_metadata = {
        "timestamp": time.time(),
        "data_metadata": data_metadata,
        "hyperparameters": {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.01
        },
        "execution_id": RUN_ID,
        "status": "complete"
    }

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key="metrics/execution_results.json",
        Body=json.dumps(execution_metadata),
        ContentType="application/json"
    )

    return execution_metadata


def main():
    """Execute the entire pipeline, simulating a Step Functions workflow."""
    print(f"\n===== STARTING DIRECT PIPELINE (RUN ID: {RUN_ID}) =====\n")

    try:
        # Create S3 bucket for data and model storage
        create_s3_bucket()

        # Execute the pipeline steps
        data_metadata = prepare_data()

        # For demonstration purposes, we present both options:
        # 1. The conceptual Step Functions approach (which won't work due to SageMaker local mode limitations)
        # 2. The unified workflow approach that actually works

        # OPTION 1: Conceptual Step Functions approach (commented out as it won't work in practice)
        # This would mimic how Step Functions would work in real AWS
        # model_metadata = train_model(data_metadata)  # Step 1: Train model
        # deployment_metadata = deploy_model(model_metadata)  # Step 2: Deploy model
        # inference_metadata = run_inference(deployment_metadata)  # Step 3: Run inference
        # cleanup(deployment_metadata)  # Step 4: Cleanup

        # OPTION 2: Unified workflow (functional approach)
        # This bypasses the limitations of SageMaker local mode by running everything in one process
        execution_metadata = run_unified_workflow(data_metadata)

        print(f"\n===== PIPELINE COMPLETED SUCCESSFULLY =====")
        print(f"S3 Bucket: {BUCKET_NAME}")
        print("Use the LocalStack Web UI to explore the data and metadata.")
        print(f"LocalStack Web UI: http://localhost:8081")

    except Exception as e:
        print(f"\n===== PIPELINE FAILED =====")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
