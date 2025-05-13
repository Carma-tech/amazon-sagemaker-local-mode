#!/usr/bin/env python3
"""
SageMaker Local Mode Integration with LocalStack Step Functions

This script:
1. Creates resources in LocalStack (S3 buckets, IAM roles, Lambda functions)
2. Defines Step Function state machines for ML training and inference workflows
3. Allows long-running ML jobs to run in the background through Step Functions
"""

import os
import sys
import json
import uuid
import time
import boto3
import logging
import argparse
import tempfile
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('localstack-ml-workflow')

# Configuration
LOCALSTACK_ENDPOINT = os.environ.get('LOCALSTACK_ENDPOINT', 'http://localhost:4567')
REGION = 'us-east-1'
BUCKET_PREFIX = 'sagemaker-local'
RUN_ID = str(uuid.uuid4())[:8]
BUCKET_NAME = f"{BUCKET_PREFIX}-{RUN_ID}"
TRAINING_STATE_MACHINE = 'SageMakerTrainingWorkflow'
INFERENCE_STATE_MACHINE = 'SageMakerInferenceWorkflow'
DUMMY_IAM_ROLE = 'arn:aws:iam::123456789012:role/service-role/dummy-role'

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


def get_boto3_resource(service_name):
    """Create a boto3 resource for LocalStack with S3 acceleration disabled."""
    session = boto3.Session(
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name=REGION
    )
    return session.resource(
        service_name,
        endpoint_url=LOCALSTACK_ENDPOINT,
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
    s3_resource = get_boto3_resource('s3')

    try:
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} already exists")
        except Exception:
            # Create the bucket
            s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"S3 bucket created: {bucket_name}")

        # Create prefixes
        prefixes = ['data/', 'models/', 'output/', 'logs/']
        for prefix in prefixes:
            s3_resource.Object(bucket_name, prefix).put(Body='')
            logger.info(f"Created S3 prefix: {prefix}")

    except Exception as e:
        logger.error(f"Error creating bucket: {str(e)}")
        return None

    return bucket_name


def create_iam_role():
    """Create an IAM role for SageMaker and Step Functions."""
    role_name = f'sagemaker-role-{RUN_ID}'
    logger.info(f"Creating IAM role: {role_name}")
    
    iam_client = get_boto3_client('iam')
    
    # Define trust policy for SageMaker
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            },
            {
                "Effect": "Allow",
                "Principal": {"Service": "states.amazonaws.com"},
                "Action": "sts:AssumeRole"
            },
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        # Check if role exists
        try:
            response = iam_client.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            logger.info(f"Role {role_name} already exists with ARN: {role_arn}")
        except Exception:
            # Create the role
            response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
            role_arn = response['Role']['Arn']
            logger.info(f"Role {role_name} created with ARN: {role_arn}")
            
            # Attach policies
            policies = [
                'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
                'arn:aws:iam::aws:policy/AmazonS3FullAccess',
                'arn:aws:iam::aws:policy/AmazonStepFunctionsFullAccess',
                'arn:aws:iam::aws:policy/AWSLambdaFullAccess'
            ]
            
            for policy_arn in policies:
                iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy_arn
                )
                logger.info(f"Attached policy {policy_arn} to role {role_name}")
    
    except Exception as e:
        logger.error(f"Error creating IAM role: {str(e)}")
        return DUMMY_IAM_ROLE
    
    return role_arn


def create_lambda_function(function_name, handler, script_path, role_arn):
    """Create a Lambda function for Step Functions integration."""
    logger.info(f"Creating Lambda function: {function_name}")
    
    lambda_client = get_boto3_client('lambda')
    
    try:
        # Check if function exists
        try:
            response = lambda_client.get_function(FunctionName=function_name)
            logger.info(f"Lambda function {function_name} already exists")
            return response['Configuration']['FunctionArn']
        except Exception:
            # Create a zip file for Lambda function
            zip_file_path = f"/tmp/{function_name}.zip"
            with tempfile.NamedTemporaryFile(suffix='.py') as temp_file:
                # Write a simple Lambda handler to the temp file
                temp_file.write(f"""\
import os
import sys
import json
import subprocess
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info(f"Received event: {{json.dumps(event)}}")
    
    # Run the script as a subprocess
    try:
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, '{os.path.basename(script_path)}')
        
        # Prepare command to run the script
        cmd = [sys.executable, script_path]
        
        # Add parameters from the event
        for key, value in event.items():
            if isinstance(value, dict):
                cmd.extend(['--' + key, json.dumps(value)])
            else:
                cmd.extend(['--' + key, str(value)])
        
        # Run the command
        logger.info(f"Running command: {{' '.join(cmd)}}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Get the output
        stdout, stderr = process.communicate(timeout=120)
        
        # Check the result
        if process.returncode == 0:
            logger.info(f"Command succeeded with output: {{stdout}}")
            return {{
                'statusCode': 200,
                'body': {{
                    'output': stdout,
                    'script': script_path
                }}
            }}
        else:
            logger.error(f"Command failed with error: {{stderr}}")
            return {{
                'statusCode': 500,
                'body': {{
                    'error': stderr,
                    'script': script_path
                }}
            }}
    
    except Exception as e:
        logger.error(f"Error running script: {{str(e)}}")
        return {{
            'statusCode': 500,
            'body': {{
                'error': str(e),
                'script': '{script_path}'
            }}
        }}
""".encode('utf-8'))
                temp_file.flush()
                
                # Create a zip file with the temp file
                import zipfile
                with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                    zipf.write(temp_file.name, arcname='lambda_function.py')
            
            # Read the zip file
            with open(zip_file_path, 'rb') as f:
                zip_bytes = f.read()
            
            # Create the Lambda function
            response = lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=role_arn,
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': zip_bytes},
                Description=f"Lambda function for {handler}",
                Timeout=900,  # 15 minutes
                MemorySize=1024
            )
            
            logger.info(f"Lambda function {function_name} created with ARN: {response['FunctionArn']}")
            return response['FunctionArn']
    
    except Exception as e:
        logger.error(f"Error creating Lambda function: {str(e)}")
        return None


def create_training_state_machine(role_arn, bucket_name):
    """Create a Step Function state machine for training workflow."""
    logger.info(f"Creating Step Function state machine: {TRAINING_STATE_MACHINE}")
    
    sf_client = get_boto3_client('stepfunctions')
    
    # Create Lambda functions for the workflow
    train_function_arn = create_lambda_function(
        'train_model_lambda',
        'train_model',
        'local_train.py',
        role_arn
    )
    
    deploy_function_arn = create_lambda_function(
        'deploy_model_lambda',
        'deploy_model',
        'local_serve.py',
        role_arn
    )
    
    # Define the state machine
    state_machine_definition = {
        "Comment": "SageMaker Training Workflow with Local Mode",
        "StartAt": "TrainModel",
        "States": {
            "TrainModel": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": train_function_arn,
                    "Payload": {
                        "model_type": "random_forest",
                        "s3_bucket": bucket_name,
                        "output_dir": "model",
                        ".$": "$"
                    }
                },
                "ResultPath": "$.training_result",
                "Next": "EvaluateTraining"
            },
            "EvaluateTraining": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.training_result.statusCode",
                        "NumericEquals": 200,
                        "Next": "DeployModel"
                    }
                ],
                "Default": "TrainingFailed"
            },
            "DeployModel": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": deploy_function_arn,
                    "Payload": {
                        "model_artifact": "$.training_result.body.model_artifact",
                        "endpoint_name": "$.training_result.body.endpoint_name",
                        "port": 8080
                    }
                },
                "ResultPath": "$.deployment_result",
                "Next": "EvaluateDeployment"
            },
            "EvaluateDeployment": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.deployment_result.statusCode",
                        "NumericEquals": 200,
                        "Next": "TrainingSuccess"
                    }
                ],
                "Default": "DeploymentFailed"
            },
            "TrainingSuccess": {
                "Type": "Succeed"
            },
            "TrainingFailed": {
                "Type": "Fail",
                "Error": "TrainingFailed",
                "Cause": "Model training failed"
            },
            "DeploymentFailed": {
                "Type": "Fail",
                "Error": "DeploymentFailed",
                "Cause": "Model deployment failed"
            }
        }
    }
    
    try:
        # Check if state machine exists
        try:
            response = sf_client.describe_state_machine(
                stateMachineArn=f"arn:aws:states:{REGION}:000000000000:stateMachine:{TRAINING_STATE_MACHINE}"
            )
            logger.info(f"State machine {TRAINING_STATE_MACHINE} already exists")
            return response['stateMachineArn']
        except Exception:
            # Create the state machine
            response = sf_client.create_state_machine(
                name=TRAINING_STATE_MACHINE,
                definition=json.dumps(state_machine_definition),
                roleArn=role_arn,
                type='STANDARD'
            )
            logger.info(f"State machine {TRAINING_STATE_MACHINE} created with ARN: {response['stateMachineArn']}")
            return response['stateMachineArn']
    
    except Exception as e:
        logger.error(f"Error creating state machine: {str(e)}")
        return None


def create_inference_state_machine(role_arn, bucket_name):
    """Create a Step Function state machine for inference workflow."""
    logger.info(f"Creating Step Function state machine: {INFERENCE_STATE_MACHINE}")
    
    sf_client = get_boto3_client('stepfunctions')
    
    # Create Lambda function for inference
    inference_function_arn = create_lambda_function(
        'inference_lambda',
        'inference',
        'inference.py',
        role_arn
    )
    
    # Define the state machine
    state_machine_definition = {
        "Comment": "SageMaker Inference Workflow with Local Mode",
        "StartAt": "PerformInference",
        "States": {
            "PerformInference": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": inference_function_arn,
                    "Payload": {
                        "endpoint_name": "${endpoint_name}",
                        "model_path": "${model_path}",
                        "s3_bucket": bucket_name,
                        "data": {
                            "type": "california_housing",
                            "sample_size": 10
                        },
                        ".$": "$"
                    }
                },
                "ResultPath": "$.inference_result",
                "Next": "EvaluateInference"
            },
            "EvaluateInference": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.inference_result.statusCode",
                        "NumericEquals": 200,
                        "Next": "InferenceSuccess"
                    }
                ],
                "Default": "InferenceFailed"
            },
            "InferenceSuccess": {
                "Type": "Succeed"
            },
            "InferenceFailed": {
                "Type": "Fail",
                "Error": "InferenceFailed",
                "Cause": "Inference failed"
            }
        }
    }
    
    try:
        # Check if state machine exists
        try:
            response = sf_client.describe_state_machine(
                stateMachineArn=f"arn:aws:states:{REGION}:000000000000:stateMachine:{INFERENCE_STATE_MACHINE}"
            )
            logger.info(f"State machine {INFERENCE_STATE_MACHINE} already exists")
            return response['stateMachineArn']
        except Exception:
            # Create the state machine
            response = sf_client.create_state_machine(
                name=INFERENCE_STATE_MACHINE,
                definition=json.dumps(state_machine_definition),
                roleArn=role_arn,
                type='STANDARD'
            )
            logger.info(f"State machine {INFERENCE_STATE_MACHINE} created with ARN: {response['stateMachineArn']}")
            return response['stateMachineArn']
    
    except Exception as e:
        logger.error(f"Error creating state machine: {str(e)}")
        return None
def start_training_execution(state_machine_arn, input_data=None):
    """Start a training workflow execution in Step Functions."""
    logger.info("Starting training workflow execution")
    
    sf_client = get_boto3_client('stepfunctions')
    
    if input_data is None:
        input_data = {}
    
    # Add timestamp to the execution name to make it unique
    execution_name = f"training-{int(time.time())}"
    
    try:
        response = sf_client.start_execution(
            stateMachineArn=state_machine_arn,
            name=execution_name,
            input=json.dumps(input_data)
        )
        
        execution_arn = response['executionArn']
        logger.info(f"Started training execution: {execution_arn}")
        
        # Save execution details to a file
        execution_details = {
            'execution_arn': execution_arn,
            'state_machine_arn': state_machine_arn,
            'input': input_data,
            'start_time': datetime.now().isoformat(),
            'execution_name': execution_name
        }
        
        with open('training_execution.json', 'w') as f:
            json.dump(execution_details, f, indent=2)
        
        logger.info("Saved execution details to training_execution.json")
        return execution_arn
    
    except Exception as e:
        logger.error(f"Error starting training execution: {str(e)}")
        return None


def start_inference_execution(state_machine_arn, endpoint_name=None, model_path=None, input_data=None):
    """Start an inference workflow execution in Step Functions."""
    logger.info("Starting inference workflow execution")
    
    sf_client = get_boto3_client('stepfunctions')
    
    if input_data is None:
        input_data = {}
    
    # Add endpoint or model path information
    if endpoint_name:
        input_data['endpoint_name'] = endpoint_name
    
    if model_path:
        input_data['model_path'] = model_path
    
    # Add timestamp to the execution name to make it unique
    execution_name = f"inference-{int(time.time())}"
    
    try:
        response = sf_client.start_execution(
            stateMachineArn=state_machine_arn,
            name=execution_name,
            input=json.dumps(input_data)
        )
        
        execution_arn = response['executionArn']
        logger.info(f"Started inference execution: {execution_arn}")
        
        # Save execution details to a file
        execution_details = {
            'execution_arn': execution_arn,
            'state_machine_arn': state_machine_arn,
            'input': input_data,
            'start_time': datetime.now().isoformat(),
            'execution_name': execution_name
        }
        
        with open('inference_execution.json', 'w') as f:
            json.dump(execution_details, f, indent=2)
        
        logger.info("Saved execution details to inference_execution.json")
        return execution_arn
    
    except Exception as e:
        logger.error(f"Error starting inference execution: {str(e)}")
        return None


def check_execution_status(execution_arn):
    """Check the status of a Step Functions execution."""
    logger.info(f"Checking status of execution: {execution_arn}")
    
    sf_client = get_boto3_client('stepfunctions')
    
    try:
        response = sf_client.describe_execution(
            executionArn=execution_arn
        )
        
        status = response['status']
        logger.info(f"Execution status: {status}")
        
        if status == 'SUCCEEDED':
            logger.info("Execution succeeded!")
            if 'output' in response:
                try:
                    output = json.loads(response['output'])
                    logger.info(f"Execution output: {json.dumps(output, indent=2)}")
                except Exception:
                    logger.info(f"Raw execution output: {response['output']}")
        elif status == 'FAILED':
            logger.error("Execution failed!")
            if 'error' in response:
                logger.error(f"Error: {response['error']}")
            if 'cause' in response:
                logger.error(f"Cause: {response['cause']}")
        
        return status, response
    
    except Exception as e:
        logger.error(f"Error checking execution status: {str(e)}")
        return None, None


def check_localstack_running():
    """Check if LocalStack is running."""
    import requests
    try:
        # Try LocalStack Desktop API endpoint format
        response = requests.get(f"{LOCALSTACK_ENDPOINT}/_localstack/info")
        if response.status_code == 200:
            logger.info("LocalStack Desktop is running")
            return True
            
        # Fall back to standard LocalStack health check
        response = requests.get(f"{LOCALSTACK_ENDPOINT}/health")
        if response.status_code == 200:
            logger.info("LocalStack is running")
            return True
            
        logger.error(f"LocalStack returned status code: {response.status_code}")
        return False
    except Exception as e:
        logger.error(f"Error connecting to LocalStack: {str(e)}")
        logger.error(f"Make sure LocalStack is running at {LOCALSTACK_ENDPOINT}")
        logger.error(f"If using LocalStack Desktop, check if it's available at: http://localhost.localstack.cloud:4567")
        return False


def setup_resources():
    """Set up all necessary resources for the workflow."""
    logger.info("Setting up resources for ML workflow")
    
    # Check if LocalStack is running
    if not check_localstack_running():
        return None, None, None, None
    
    # Create S3 bucket
    bucket_name = create_bucket(BUCKET_NAME)
    if not bucket_name:
        logger.error("Failed to create S3 bucket")
        return None, None, None, None
    
    # Create IAM role
    role_arn = create_iam_role()
    if not role_arn or role_arn == DUMMY_IAM_ROLE:
        logger.warning("Using dummy IAM role")
    
    # Create state machines
    training_sm_arn = create_training_state_machine(role_arn, bucket_name)
    if not training_sm_arn:
        logger.error("Failed to create training state machine")
        return bucket_name, role_arn, None, None
    
    inference_sm_arn = create_inference_state_machine(role_arn, bucket_name)
    if not inference_sm_arn:
        logger.error("Failed to create inference state machine")
        return bucket_name, role_arn, training_sm_arn, None
    
    logger.info("Successfully set up all resources!")
    return bucket_name, role_arn, training_sm_arn, inference_sm_arn


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SageMaker Local Mode with LocalStack Step Functions')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up resources')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Start training workflow')
    train_parser.add_argument('--model-type', type=str, default='random_forest',
                             help='Type of model to train (random_forest)')
    train_parser.add_argument('--hyperparameters', type=str, default='{}',
                             help='JSON string of hyperparameters')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Start inference workflow')
    inference_parser.add_argument('--endpoint-name', type=str, help='Name of the endpoint to use')
    inference_parser.add_argument('--model-path', type=str, help='Path to the model file')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check execution status')
    status_parser.add_argument('--execution-arn', type=str, required=True,
                              help='ARN of the execution to check')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if args.command == 'setup':
        bucket_name, role_arn, training_sm_arn, inference_sm_arn = setup_resources()
        
        if bucket_name and role_arn and training_sm_arn and inference_sm_arn:
            logger.info("\nSetup completed successfully!")
            logger.info(f"S3 Bucket: {bucket_name}")
            logger.info(f"IAM Role ARN: {role_arn}")
            logger.info(f"Training State Machine ARN: {training_sm_arn}")
            logger.info(f"Inference State Machine ARN: {inference_sm_arn}")
            logger.info("\nTo start training: python localstack_ml_workflow.py train")
            logger.info("To start inference: python localstack_ml_workflow.py inference --endpoint-name <name> --model-path <path>")
        else:
            logger.error("Setup failed")
    
    elif args.command == 'train':
        bucket_name, role_arn, training_sm_arn, _ = setup_resources()
        
        if training_sm_arn:
            # Parse hyperparameters if provided
            try:
                hyperparameters = json.loads(args.hyperparameters)
            except json.JSONDecodeError:
                hyperparameters = {}
            
            input_data = {
                'model_type': args.model_type,
                'hyperparameters': hyperparameters,
                's3_bucket': bucket_name
            }
            
            execution_arn = start_training_execution(training_sm_arn, input_data)
            
            if execution_arn:
                logger.info("\nTraining workflow started successfully!")
                logger.info(f"Execution ARN: {execution_arn}")
                logger.info("Training is running in the background via Step Functions")
                logger.info("You can continue using the terminal for other tasks")
                logger.info("\nTo check the status: python localstack_ml_workflow.py status --execution-arn <arn>")
            else:
                logger.error("Failed to start training workflow")
        else:
            logger.error("Failed to create or retrieve training state machine")
    
    elif args.command == 'inference':
        bucket_name, role_arn, _, inference_sm_arn = setup_resources()
        
        if inference_sm_arn:
            if not args.endpoint_name and not args.model_path:
                logger.error("Either endpoint-name or model-path must be provided for inference")
                return
            
            input_data = {
                's3_bucket': bucket_name
            }
            
            execution_arn = start_inference_execution(
                inference_sm_arn,
                endpoint_name=args.endpoint_name,
                model_path=args.model_path,
                input_data=input_data
            )
            
            if execution_arn:
                logger.info("\nInference workflow started successfully!")
                logger.info(f"Execution ARN: {execution_arn}")
                logger.info("Inference is running in the background via Step Functions")
                logger.info("You can continue using the terminal for other tasks")
                logger.info("\nTo check the status: python localstack_ml_workflow.py status --execution-arn <arn>")
            else:
                logger.error("Failed to start inference workflow")
        else:
            logger.error("Failed to create or retrieve inference state machine")
    
    elif args.command == 'status':
        status, response = check_execution_status(args.execution_arn)
        
        if status:
            logger.info(f"\nExecution Status: {status}")
            if status == 'RUNNING':
                logger.info("The workflow is still running in the background")
                logger.info("You can check the status again later")
            elif status == 'SUCCEEDED':
                logger.info("The workflow has completed successfully!")
            elif status == 'FAILED':
                logger.error("The workflow has failed")
        else:
            logger.error("Failed to check execution status")
    
    else:
        logger.error("Invalid command. Use one of: setup, train, inference, status")
        logger.info("\nExample commands:")
        logger.info("  python localstack_ml_workflow.py setup")
        logger.info("  python localstack_ml_workflow.py train --model-type random_forest")
        logger.info("  python localstack_ml_workflow.py inference --endpoint-name my-endpoint")
        logger.info("  python localstack_ml_workflow.py status --execution-arn <arn>")


if __name__ == "__main__":
    main()
