#!/usr/bin/env python3
"""
Asynchronous ML Workflow Runner with LocalStack Integration

This script:
1. Runs ML training and inference jobs in the background
2. Uses S3 in LocalStack for data storage
3. Provides status tracking without blocking the terminal
"""

import os
import sys
import json
import time
import uuid
import boto3
import logging
import argparse
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('async-ml-workflow')

# Configuration
LOCALSTACK_ENDPOINT = os.environ.get('LOCALSTACK_ENDPOINT', 'http://localhost:4567')
REGION = 'us-east-1'
BUCKET_PREFIX = 'sagemaker-local'
RUN_ID = str(uuid.uuid4())[:8]
BUCKET_NAME = f"{BUCKET_PREFIX}-{RUN_ID}"
STATUS_FILE = 'ml_workflow_status.json'

# Disable S3 acceleration for LocalStack compatibility
os.environ['AWS_S3_DISABLE_ACCELERATE_ENDPOINT'] = 'true'
os.environ['AWS_S3_FORCE_PATH_STYLE'] = 'true'
os.environ['S3_USE_ACCELERATE_ENDPOINT'] = 'false'
os.environ['S3_ADDRESSING_STYLE'] = 'path'


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


def create_bucket(bucket_name=None):
    """Create S3 bucket for ML data and artifacts."""
    if bucket_name is None:
        bucket_name = BUCKET_NAME
        
    logger.info(f"Creating S3 bucket: {bucket_name}")
    s3_client = get_boto3_client('s3')

    try:
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} already exists")
        except Exception:
            # Create the bucket
            s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"S3 bucket created: {bucket_name}")

        # Create prefixes for organization
        prefixes = ['data/', 'models/', 'output/', 'logs/']
        for prefix in prefixes:
            s3_client.put_object(Bucket=bucket_name, Key=prefix, Body='')
            logger.info(f"Created S3 prefix: {prefix}")

    except Exception as e:
        logger.error(f"Error creating bucket: {str(e)}")
        return None

    return bucket_name


def upload_to_s3(bucket_name, local_path, s3_key):
    """Upload a file to S3."""
    logger.info(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
    s3_client = get_boto3_client('s3')
    
    try:
        with open(local_path, 'rb') as f:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=f.read()
            )
        logger.info(f"Upload successful: s3://{bucket_name}/{s3_key}")
        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return None


def download_from_s3(bucket_name, s3_key, local_path):
    """Download a file from S3."""
    logger.info(f"Downloading s3://{bucket_name}/{s3_key} to {local_path}")
    s3_client = get_boto3_client('s3')
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download the file
        s3_client.download_file(
            Bucket=bucket_name,
            Key=s3_key,
            Filename=local_path
        )
        logger.info(f"Download successful: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Error downloading from S3: {str(e)}")
        return None


def check_localstack():
    """Check if LocalStack is running."""
    import requests
    try:
        # Try LocalStack Desktop API endpoint
        response = requests.get(f"{LOCALSTACK_ENDPOINT}/_localstack/info")
        if response.status_code == 200:
            logger.info("LocalStack Desktop is running")
            return True
        
        # Try standard LocalStack health endpoint
        response = requests.get(f"{LOCALSTACK_ENDPOINT}/health")
        if response.status_code == 200:
            logger.info("LocalStack is running")
            return True
        
        logger.error(f"LocalStack returned status code: {response.status_code}")
        return False
    except Exception as e:
        logger.error(f"Error connecting to LocalStack: {str(e)}")
        logger.error(f"Make sure LocalStack is running at {LOCALSTACK_ENDPOINT}")
        return False


def run_in_background(cmd, log_file=None):
    """Run a command in the background and return the process."""
    if log_file is None:
        log_file = f"workflow_{int(time.time())}.log"
    
    logger.info(f"Running command in background: {' '.join(cmd)}")
    logger.info(f"Logs will be written to: {log_file}")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=False
        )
    
    # Save process info to status file
    process_info = {
        'pid': process.pid,
        'command': ' '.join(cmd),
        'log_file': log_file,
        'start_time': datetime.now().isoformat(),
        'status': 'RUNNING'
    }
    
    # Add to status file
    update_status(process_info)
    
    return process, process_info


def update_status(process_info):
    """Update the workflow status file with process information."""
    try:
        # Load existing status if available
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
        else:
            status = {'processes': []}
        
        # Check if this process is already in the status file
        for i, p in enumerate(status['processes']):
            if p.get('pid') == process_info.get('pid'):
                # Update existing process info
                status['processes'][i] = process_info
                break
        else:
            # Add new process info
            status['processes'].append(process_info)
        
        # Write updated status
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
        
        logger.info(f"Updated status file: {STATUS_FILE}")
    except Exception as e:
        logger.error(f"Error updating status file: {str(e)}")


def check_process_status(pid):
    """Check if a process is still running."""
    try:
        # For Unix/Linux/macOS
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def setup():
    """Set up resources for the workflow."""
    logger.info("Setting up resources for ML workflow")
    
    # Check if LocalStack is running
    if not check_localstack():
        logger.error("LocalStack is not running, setup failed")
        return None
    
    # Create S3 bucket
    bucket_name = create_bucket()
    if not bucket_name:
        logger.error("Failed to create S3 bucket")
        return None
    
    logger.info(f"Setup complete with bucket: {bucket_name}")
    return bucket_name


def start_training(model_type='random_forest', hyperparameters=None, s3_bucket=None):
    """Start a training job in the background."""
    logger.info(f"Starting training job for model type: {model_type}")
    
    if s3_bucket is None:
        s3_bucket = setup()
        if not s3_bucket:
            return None
    
    # Prepare command
    cmd = [sys.executable, 'local_train.py']
    
    # Add arguments
    if model_type:
        cmd.extend(['--model-type', model_type])
    
    # Add hyperparameters if provided
    if hyperparameters:
        if isinstance(hyperparameters, str):
            hyperparameters_dict = json.loads(hyperparameters)
        else:
            hyperparameters_dict = hyperparameters
        
        for key, value in hyperparameters_dict.items():
            cmd.extend([f'--{key}', str(value)])
    
    # Configure log file
    timestamp = int(time.time())
    log_file = f"training_{timestamp}.log"
    
    # Run the command in the background
    process, process_info = run_in_background(cmd, log_file)
    
    # Add training-specific information
    process_info['job_type'] = 'training'
    process_info['model_type'] = model_type
    process_info['s3_bucket'] = s3_bucket
    
    # Update status with training info
    update_status(process_info)
    
    return process_info


def start_deployment(model_artifact=None, endpoint_name=None, port=8080):
    """Start a model deployment in the background."""
    logger.info(f"Starting deployment for model: {model_artifact}")
    
    # Prepare command
    cmd = [sys.executable, 'local_serve.py']
    
    # Add arguments
    if model_artifact:
        cmd.extend(['--model-artifact', model_artifact])
    
    if endpoint_name:
        cmd.extend(['--endpoint-name', endpoint_name])
    
    cmd.extend(['--port', str(port)])
    
    # Configure log file
    timestamp = int(time.time())
    log_file = f"deployment_{timestamp}.log"
    
    # Run the command in the background
    process, process_info = run_in_background(cmd, log_file)
    
    # Add deployment-specific information
    process_info['job_type'] = 'deployment'
    process_info['model_artifact'] = model_artifact
    process_info['endpoint_name'] = endpoint_name
    process_info['port'] = port
    
    # Update status with deployment info
    update_status(process_info)
    
    return process_info


def start_inference(endpoint_name=None, model_path=None, input_data=None, content_type=None):
    """Start an inference job in the background."""
    logger.info("Starting inference job")
    
    # Prepare command
    cmd = [sys.executable, 'inference.py']
    
    # Add arguments
    if endpoint_name:
        cmd.extend(['--endpoint-name', endpoint_name])
    
    if model_path:
        cmd.extend(['--model-path', model_path])
    
    if input_data:
        cmd.extend(['--input-data', input_data])
    
    if content_type:
        cmd.extend(['--content-type', content_type])
    
    # Configure log file
    timestamp = int(time.time())
    log_file = f"inference_{timestamp}.log"
    
    # Run the command in the background
    process, process_info = run_in_background(cmd, log_file)
    
    # Add inference-specific information
    process_info['job_type'] = 'inference'
    process_info['endpoint_name'] = endpoint_name
    process_info['model_path'] = model_path
    
    # Update status with inference info
    update_status(process_info)
    
    return process_info


def check_status(pid=None, job_type=None):
    """Check the status of running jobs."""
    logger.info("Checking job status")
    
    if not os.path.exists(STATUS_FILE):
        logger.error(f"Status file not found: {STATUS_FILE}")
        return None
    
    try:
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
        
        processes = status.get('processes', [])
        
        if not processes:
            logger.info("No processes found in the status file")
            return None
        
        # Filter processes if pid or job_type is specified
        if pid is not None:
            processes = [p for p in processes if p.get('pid') == pid]
        
        if job_type is not None:
            processes = [p for p in processes if p.get('job_type') == job_type]
        
        # Check status of each process
        for p in processes:
            pid = p.get('pid')
            if pid is not None:
                p['running'] = check_process_status(pid)
                if not p['running'] and p.get('status') == 'RUNNING':
                    p['status'] = 'COMPLETED'
                    update_status(p)
            
            # Get the latest log content
            log_file = p.get('log_file')
            if log_file and os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        # Read last few lines
                        lines = f.readlines()
                        p['recent_logs'] = ''.join(lines[-10:]) if lines else ''
                except Exception as e:
                    logger.error(f"Error reading log file: {str(e)}")
                    p['recent_logs'] = f"Error reading log: {str(e)}"
        
        return processes
    
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Asynchronous ML Workflow Runner')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup command
    subparsers.add_parser('setup', help='Set up resources')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Start training job')
    train_parser.add_argument('--model-type', type=str, default='random_forest',
                             help='Type of model to train (random_forest)')
    train_parser.add_argument('--hyperparameters', type=str, default='{}',
                             help='JSON string of hyperparameters')
    train_parser.add_argument('--s3-bucket', type=str, help='S3 bucket to use')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Start model deployment')
    deploy_parser.add_argument('--model-artifact', type=str, required=True,
                              help='Path to model artifact (.tar.gz file)')
    deploy_parser.add_argument('--endpoint-name', type=str,
                              help='Name for the endpoint')
    deploy_parser.add_argument('--port', type=int, default=8080,
                              help='Port to run the server on')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Start inference job')
    inference_parser.add_argument('--endpoint-name', type=str,
                                 help='Name of the endpoint to use')
    inference_parser.add_argument('--model-path', type=str,
                                 help='Path to the model file')
    inference_parser.add_argument('--input-data', type=str,
                                 help='Input data for inference (JSON string)')
    inference_parser.add_argument('--content-type', type=str, default='application/json',
                                 help='Content type of input data')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check job status')
    status_parser.add_argument('--pid', type=int, help='Process ID to check')
    status_parser.add_argument('--job-type', type=str, 
                              choices=['training', 'deployment', 'inference'],
                              help='Type of job to check')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if args.command == 'setup':
        bucket_name = setup()
        if bucket_name:
            logger.info(f"Setup completed successfully with bucket: {bucket_name}")
        else:
            logger.error("Setup failed")
    
    elif args.command == 'train':
        # Parse hyperparameters
        hyperparameters = {}
        if args.hyperparameters:
            try:
                hyperparameters = json.loads(args.hyperparameters)
            except json.JSONDecodeError:
                logger.error(f"Invalid hyperparameters JSON: {args.hyperparameters}")
                return
        
        process_info = start_training(
            model_type=args.model_type,
            hyperparameters=hyperparameters,
            s3_bucket=args.s3_bucket
        )
        
        if process_info:
            logger.info("\nTraining job started successfully!")
            logger.info(f"Process ID: {process_info['pid']}")
            logger.info(f"Log file: {process_info['log_file']}")
            logger.info("Training is running in the background")
            logger.info("You can continue using the terminal for other tasks")
            logger.info("\nTo check the status: python async_ml_workflow.py status")
        else:
            logger.error("Failed to start training job")
    
    elif args.command == 'deploy':
        process_info = start_deployment(
            model_artifact=args.model_artifact,
            endpoint_name=args.endpoint_name,
            port=args.port
        )
        
        if process_info:
            logger.info("\nDeployment started successfully!")
            logger.info(f"Process ID: {process_info['pid']}")
            logger.info(f"Log file: {process_info['log_file']}")
            logger.info(f"Endpoint will be available at: http://localhost:{args.port}/invocations")
            logger.info("Deployment is running in the background")
            logger.info("You can continue using the terminal for other tasks")
            logger.info("\nTo check the status: python async_ml_workflow.py status")
        else:
            logger.error("Failed to start deployment")
    
    elif args.command == 'inference':
        if not args.endpoint_name and not args.model_path:
            logger.error("Either endpoint-name or model-path must be provided")
            return
        
        process_info = start_inference(
            endpoint_name=args.endpoint_name,
            model_path=args.model_path,
            input_data=args.input_data,
            content_type=args.content_type
        )
        
        if process_info:
            logger.info("\nInference job started successfully!")
            logger.info(f"Process ID: {process_info['pid']}")
            logger.info(f"Log file: {process_info['log_file']}")
            logger.info("Inference is running in the background")
            logger.info("You can continue using the terminal for other tasks")
            logger.info("\nTo check the status: python async_ml_workflow.py status")
        else:
            logger.error("Failed to start inference job")
    
    elif args.command == 'status':
        processes = check_status(pid=args.pid, job_type=args.job_type)
        
        if processes:
            logger.info("\nJob Status:")
            logger.info("-" * 50)
            
            for p in processes:
                status = p.get('status', 'UNKNOWN')
                running = p.get('running', False)
                
                if running and status != 'COMPLETED':
                    status_str = "RUNNING"
                elif not running and status == 'RUNNING':
                    status_str = "COMPLETED"
                else:
                    status_str = status
                
                logger.info(f"Job Type: {p.get('job_type', 'Unknown')}")
                logger.info(f"PID: {p.get('pid', 'Unknown')}")
                logger.info(f"Status: {status_str}")
                logger.info(f"Start Time: {p.get('start_time', 'Unknown')}")
                logger.info(f"Log File: {p.get('log_file', 'Unknown')}")
                
                if p.get('job_type') == 'training':
                    logger.info(f"Model Type: {p.get('model_type', 'Unknown')}")
                elif p.get('job_type') == 'deployment':
                    logger.info(f"Endpoint Name: {p.get('endpoint_name', 'Unknown')}")
                    logger.info(f"Port: {p.get('port', 'Unknown')}")
                elif p.get('job_type') == 'inference':
                    logger.info(f"Endpoint/Model: {p.get('endpoint_name', 'Unknown') or p.get('model_path', 'Unknown')}")
                
                # Show recent logs
                if 'recent_logs' in p and p['recent_logs']:
                    logger.info("\nRecent Logs:")
                    logger.info("-" * 50)
                    logger.info(p['recent_logs'])
                
                logger.info("-" * 50)
        else:
            logger.info("No jobs found or status file doesn't exist")
    
    else:
        logger.error("Invalid command. Use one of: setup, train, deploy, inference, status")
        logger.info("\nExample commands:")
        logger.info("  python async_ml_workflow.py setup")
        logger.info("  python async_ml_workflow.py train --model-type random_forest")
        logger.info("  python async_ml_workflow.py deploy --model-artifact model/model.tar.gz --endpoint-name my-endpoint")
        logger.info("  python async_ml_workflow.py inference --endpoint-name my-endpoint")
        logger.info("  python async_ml_workflow.py status")


if __name__ == "__main__":
    main()
