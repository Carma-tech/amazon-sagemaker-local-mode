# LocalStack + SageMaker Local Mode Integration

This project demonstrates how to run an AWS SageMaker ML workflow locally using LocalStack and SageMaker local mode. The implementation provides multiple approaches, with the recommended option highlighted below.

1. **Direct Pipeline** (`direct_pipeline.py`): Simple script that runs SageMaker locally with LocalStack for storage
2. **Step Functions Definition** (`step_functions_definition.py`): Creates a Step Functions state machine but has Lambda execution issues
3. **Complete Workflow** (`complete_workflow.py`): Creates Lambda functions but they remain in Pending state
4. **Direct Step Functions** (`direct_step_functions.py`): Combines Step Functions orchestration with SageMaker execution
5. **✅ Direct Integration** (`direct_integration.py`): **RECOMMENDED** - Integrates real ML model outputs with Step Functions

## Overview

This integration allows you to:

- Train and deploy machine learning models locally without AWS costs
- Use AWS Step Functions for orchestration 
- Store data and artifacts in S3 (emulated by LocalStack)
- Execute multiple ML frameworks (TensorFlow, XGBoost, PyTorch)
- Maintain code compatibility with real AWS

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- AWS CLI
- LocalStack CLI (`pip install localstack awscli`)
- The AWS Toolkit extension for your IDE (optional, for visualizing Step Functions)

## Setup

### 1. Start LocalStack with Required Services

```bash
cd /Users/user/projects/carma_tech_projects/amazon-sagemaker-local-mode/z_localstack_deployment
docker-compose up -d
```

This starts a LocalStack container with all necessary services:
- S3 (for data storage)
- StepFunctions (for workflow orchestration)
- IAM (for roles and permissions)
- Lambda (for workflow steps)
- SQS (for task communication)
- CloudWatch (for logs)

### 2. Configure AWS CLI for LocalStack

```bash
# Add this to your ~/.aws/credentials file
[localstack]
aws_access_key_id = test
aws_secret_access_key = test
region = us-east-1

# Add this to your ~/.aws/config file
[profile localstack]
region = us-east-1
output = json
endpoint_url = http://localhost:4567
```

### 3. Verify LocalStack is Running Properly

Ensure all services are properly running:

```bash
curl http://localhost:4567/_localstack/health
```

Confirm that the services show as "available" or "running" in the output.

## Running the Workflows

### ✅ Recommended Approach: Direct Integration

This approach creates a Step Functions workflow with REAL model predictions and metrics from an actual ML model:

```bash
python direct_integration.py
```

This script:
1. Trains an actual ML model on the California Housing dataset
2. Captures real predictions, actual values, and model metrics (MSE, R²)
3. Uses these real values in the Step Functions state machine
4. Visualizes the complete workflow in AWS Toolkit with actual model results
5. Shows detailed comparison of predictions vs. actual values

### Alternative: Direct Step Functions 

If you need more control over SageMaker script execution:

```bash
python direct_step_functions.py [--non-interactive] [--model-dir DIR_NAME] [--script-name SCRIPT_NAME]
```

This script:
1. Creates a Step Functions state machine using Pass states
2. Visualizes the ML workflow for AWS Toolkit
3. Allows selection of different SageMaker model frameworks
4. Works in both interactive and non-interactive modes

### Alternative: Direct Pipeline

For simple testing without Step Functions orchestration:

```bash
python direct_pipeline.py
```

This script directly executes SageMaker local mode operations while using LocalStack for S3 storage.

## Testing Different SageMaker Model Frameworks

The scripts can be adapted to work with any SageMaker model framework. To use a different framework:

### XGBoost

```bash
# Modify the SAGEMAKER_SCRIPT path in direct_step_functions.py
SAGEMAKER_SCRIPT = os.path.join(
    PARENT_DIR, 
    'xgboost_script_mode_local_training_and_serving',
    'your_script_name.py'
)
```

### PyTorch

```bash
# Modify the SAGEMAKER_SCRIPT path in direct_step_functions.py
SAGEMAKER_SCRIPT = os.path.join(
    PARENT_DIR, 
    'pytorch_script_mode_local_training_and_serving',
    'your_script_name.py'
)
```

### PyTorch NLP or Other Models

Follow the same pattern, updating the script path to point to your specific model implementation.

## Monitoring and Visualizing Workflows

### Using AWS Toolkit

1. Open AWS Toolkit in your IDE
2. Connect to the LocalStack endpoint (http://localhost:4567)
3. Navigate to Step Functions
4. Select "sagemaker-local-direct" to view the workflow
5. Click on any execution to see its status and details

### Using AWS CLI

```bash
# List all state machines
aws --endpoint-url=http://localhost:4567 stepfunctions list-state-machines

# Get details of a specific state machine
aws --endpoint-url=http://localhost:4567 stepfunctions describe-state-machine \
  --state-machine-arn arn:aws:states:us-east-1:000000000000:stateMachine:sagemaker-local-direct

# List executions for a state machine
aws --endpoint-url=http://localhost:4567 stepfunctions list-executions \
  --state-machine-arn arn:aws:states:us-east-1:000000000000:stateMachine:sagemaker-local-direct

# Get details of a specific execution
aws --endpoint-url=http://localhost:4567 stepfunctions describe-execution \
  --execution-arn arn:aws:states:us-east-1:000000000000:execution:sagemaker-local-direct:run-XXXXXXXX
```

### Exploring S3 Data

```bash
# List all S3 buckets
aws --endpoint-url=http://localhost:4567 s3 ls

# List objects in a specific bucket (replace BUCKET_ID with yours)
aws --endpoint-url=http://localhost:4567 s3 ls s3://ml-pipeline-data-BUCKET_ID/

# View workflow metadata
aws --endpoint-url=http://localhost:4567 s3 cp \
  s3://ml-pipeline-data-BUCKET_ID/metrics/training_metadata.json -
```

### Interacting with the Model

When running the direct pipeline script, it will:
1. Train the model
2. Deploy it to a local endpoint
3. Run inference automatically with sample data 

You can see the predictions and target values in the output.

If you want to manually test the deployed model after running the direct pipeline, you can use the SageMaker local mode script directly:

```bash
# Run just the inference part against an existing endpoint
cd /Users/user/projects/carma_tech_projects/amazon-sagemaker-local-mode
python tensorflow_script_mode_california_housing_local_training_and_serving/tensorflow_script_localstack_mode.py --inference-only
```

### Checking LocalStack Resources

```bash
# List S3 buckets
aws --endpoint-url=http://localhost:4567 s3 ls

# List objects in a bucket
aws --endpoint-url=http://localhost:4567 s3 ls s3://ml-pipeline-data-<bucket-id>/

# Get a specific object
aws --endpoint-url=http://localhost:4567 s3 cp s3://ml-pipeline-data-<bucket-id>/models/model_metadata.json .
```

## Troubleshooting

### LocalStack Issues

- If you encounter issues with LocalStack, try restarting it:
  ```bash
  docker-compose down
  docker-compose up -d
  ```

- Check LocalStack logs:
  ```bash
  docker logs localstack
  ```

- Verify LocalStack is healthy:
  ```bash
  curl http://localhost:4567/_localstack/health
  ```

### SageMaker Local Mode Issues

- SageMaker local mode runs Docker containers - ensure Docker is running
- Check for Docker permission issues if you encounter "permission denied" errors
- SageMaker containers can be large - ensure you have enough disk space

### Step Functions Issues

- If executions fail, check the state machine definition:
  ```bash
  cat step_functions_definition.json
  ```

- Verify Lambda functions exist before referencing them in the state machine

## Architecture

```
┌─────────────────────────────┐      ┌─────────────────────────────┐
│                             │      │                             │
│  LocalStack                 │      │  SageMaker Local Mode       │
│                             │      │                             │
│  ┌─────────────┐            │      │  ┌─────────────┐            │
│  │             │            │      │  │             │            │
│  │     S3      │◄──────────┼──────┼──┤  Training    │            │
│  │             │            │      │  │             │            │
│  └─────────────┘            │      │  └─────────────┘            │
│                             │      │                             │
│  ┌─────────────┐            │      │  ┌─────────────┐            │
│  │             │            │      │  │             │            │
│  │ StepFunctions│◄─────────┼──────┼──┤  Deployment  │            │
│  │             │            │      │  │             │            │
│  └─────────────┘            │      │  └─────────────┘            │
│                             │      │                             │
│  ┌─────────────┐            │      │  ┌─────────────┐            │
│  │             │            │      │  │             │            │
│  │   Lambda    │◄──────────┼──────┼──┤  Inference   │            │
│  │             │            │      │  │             │            │
│  └─────────────┘            │      │  └─────────────┘            │
│                             │      │                             │
└─────────────────────────────┘      └─────────────────────────────┘
```

## Limitations

- The free version of LocalStack has some limitations with Lambda functions, which is why we've included the direct pipeline approach
- SageMaker local mode doesn't support all SageMaker features
- Step Functions in LocalStack may have limited support for some advanced features

## Implementation Details

### Direct Integration Implementation

Our recommended solution (`direct_integration.py`) provides a complete integration of real ML model outputs with Step Functions:

1. **Real Model Training**: Actually trains a real ML model on the California Housing dataset
2. **Actual Predictions**: Uses real model predictions in the Step Functions state machine
3. **Real Performance Metrics**: Calculates and displays MSE and R² metrics from the model
4. **Visualization-Ready**: The state machine definition is properly structured for AWS Toolkit visualization
5. **Comprehensive Output**: Detailed comparison of model predictions vs. actual values

### Advantages Over Previous Approaches

Unlike previous implementations, `direct_integration.py`:

1. **Uses Real Data**: No mock/dummy hardcoded values - everything comes from an actual model
2. **Proper Execution Order**: First trains the model, then creates Step Functions with those results
3. **Works With LocalStack Free Tier**: Avoids Lambda function issues by using Pass states
4. **Proper Error Handling**: Better handling of S3 and AWS service errors

### State Machine Structure

The workflow has these states:
- **DataPreparation**: Prepares training and test data
- **ModelTraining**: Trains the model using real metrics from an actual model
- **ModelDeployment**: Simulates model deployment with real metadata
- **ModelInference**: Contains actual model predictions and target values
- **Success**: Indicates successful workflow completion

## Troubleshooting

### LocalStack Issues

- **Service not available**: Ensure the service is enabled in docker-compose.yml
  ```
  SERVICES=stepfunctions,batch,iam,s3,cloudwatch,lambda,logs,events,sqs
  ```

- **S3 Accelerate errors**: Use proper boto3 configuration:
  ```python
  config=boto3.session.Config(
      s3={'addressing_style': 'path', 'use_accelerate_endpoint': False},
      signature_version='s3v4'
  )
  ```

- **Lambda Pending state**: Use the direct_step_functions.py approach which doesn't rely on Lambda execution

### SageMaker Local Mode Issues

- SageMaker local mode requires Docker to be running
- SageMaker containers can be large (~4-5GB) and may take time to download initially
- Set appropriate timeouts when running SageMaker operations

## Architecture Diagram

```
┌──────────────────────────────────────┐    ┌────────────────────────────────┐
│                                      │    │                                │
│  LocalStack                          │    │  SageMaker Local Mode          │
│                                      │    │                                │
│  ┌────────────┐       ┌────────────┐ │    │  ┌────────────┐               │
│  │            │       │            │ │    │  │            │               │
│  │     S3     │◄──────┤ Step       │ │    │  │  Local     │               │
│  │  (Storage) │       │ Functions  │ │    │  │  Docker    │               │
│  └────────────┘       │ Workflow   │ │    │  │  Containers│               │
│                       │            │ │    │  │            │               │
│                       └──────┬─────┘ │    │  └────┬───────┘               │
│                              │       │    │       │                        │
│                              │       │    │       │                        │
│                              └───────┼────┼───────┘                        │
│                                      │    │                                │
└──────────────────────────────────────┘    └────────────────────────────────┘
```

## Future Enhancements

The following enhancements are planned for future implementation:

### 1. Multi-Framework Support

A flexible framework adapter system will be added to support multiple ML model types:

```python
def get_model_handler(model_type):
    """Return the appropriate model handler based on model type."""
    handlers = {
        'tensorflow': TensorFlowModelHandler,
        'pytorch': PyTorchModelHandler,
        'xgboost': XGBoostModelHandler
    }
    return handlers.get(model_type, TensorFlowModelHandler)()
```

Each handler will implement standardized methods for model training, deployment, and inference.

### 2. Enhanced Metrics & Visualizations

Adding comprehensive model metrics and visualizations:

- **Extended Metrics**: MAE, RMSE, explained variance, convergence rate, etc.
- **Automated Visualizations**: Learning curves, prediction vs. actual scatter plots, feature importance charts
- **S3 Integration**: Storing visualization artifacts in S3 for later retrieval

### 3. Robust Error Handling 

Improving resiliency with:

- Structured logging
- Retry mechanisms for AWS service calls
- Graceful fallbacks for service failures
- Comprehensive error reporting
- Automated resource cleanup

## FAQ

### Why doesn't the original Step Functions approach work?

The free tier of LocalStack has limitations where Lambda functions remain in "Pending" state and cannot be executed.

### Can I use this with any SageMaker model?

Yes! Both the `direct_integration.py` and `direct_step_functions.py` approaches can be extended to support any SageMaker framework.

### How do I get real model metrics in Step Functions?

The `direct_integration.py` approach already does this by training a real model and passing the actual predictions and metrics to the Step Functions workflow.

### Can I integrate this with my CI/CD pipeline?

Absolutely. The scripts are designed to be easily automated and can be incorporated into CI/CD workflows for testing ML pipelines.

For development, you might want to use the direct pipeline, while the Step Functions workflow better represents a production architecture.
