# Asynchronous SageMaker Local Mode with LocalStack Integration

This project implements an integrated workflow for AWS SageMaker in local mode with LocalStack, featuring an asynchronous execution model that allows long-running ML training and inference jobs to run in the background. The implementation is especially optimized for inference workloads and TensorFlow model support.

## Key Components

1. **🔄 Asynchronous ML Workflow** (`async_ml_workflow.py`): **MAIN ENTRY POINT** - Runs ML jobs in the background with comprehensive status tracking
2. **🏋️ Training Module** (`local_train.py`): Trains models using scikit-learn or TensorFlow with California Housing dataset
3. **🚀 Deployment Module** (`local_serve.py`): Deploys models to local Flask-based endpoints
4. **🔮 Inference Module** (`inference.py`): **FOCUS AREA** - Flexible inference with comprehensive metrics and model type detection
5. **☁️ LocalStack Integration** (`localstack_ml_workflow.py`): Sets up AWS resources locally for zero-cost development

## Key Benefits

This asynchronous workflow provides:

- **Background Execution**: ML jobs run asynchronously, freeing up the terminal for other tasks
- **Real-time Status Updates**: Monitor training and deployment progress without blocking
- **Multiple Model Support**: Use scikit-learn or TensorFlow models interchangeably
- **Framework Detection**: Auto-detects model framework during inference
- **LocalStack Integration**: Full AWS compatibility without cloud costs
- **Step Functions Compatibility**: Works with Step Functions for orchestration
- **Comprehensive Metrics**: Detailed metrics for model evaluation during inference
- **TensorFlow Optimization**: Special handling for TensorFlow inference

## Prerequisites

- Python 3.9+
- Docker (for LocalStack Desktop)
- LocalStack Desktop (recommended) or LocalStack CLI
- Required Python packages: boto3, scikit-learn, tensorflow, flask

## Setup

### 1. Start LocalStack (Recommended: Docker Compose)

**The most reliable and reproducible way to run LocalStack for this ML workflow is via Docker Compose. This ensures all required AWS services are started with the correct configuration and ports.**

From the `z_localstack_deployment` directory:
```bash
docker-compose up -d
```

This will start LocalStack with all necessary services for ML workflows, including S3, Lambda, Step Functions, IAM, CloudWatch, and more. The configuration is controlled by `docker-compose.yml` in this directory. You can check and modify the list of enabled services in that file (see the `SERVICES` environment variable).

**Alternatives:**
- LocalStack Desktop (GUI, for local development)
- LocalStack CLI (`localstack start`)

> **Note:** For team setups, CI, or reproducibility, always prefer the Docker Compose method.

This setup provides local versions of essential AWS services:
- S3: Model storage and data management
- Lambda: Function execution
- Step Functions: Workflow orchestration
- IAM: Permission management
- CloudWatch: Logging and monitoring

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

## Running the Asynchronous ML Workflow

### Setting Up LocalStack Resources

```bash
python async_ml_workflow.py setup
```
This initializes all required resources in LocalStack:
- Creates S3 buckets with proper configuration
- Sets up required prefixes (data, models, output, logs)
- Applies correct S3 endpoint settings for LocalStack compatibility

### Training a Model in the Background

```bash
python async_ml_workflow.py train --model-type random_forest
```
Alternative model types:
- `--model-type linear_regression`
- `--model-type tensorflow`

The training job runs in the background, freeing your terminal. The model is automatically saved as a tarball artifact.

### Deploying a Model as a Background Service

```bash
python async_ml_workflow.py deploy --model-artifact model/model.tar.gz --endpoint-name my-endpoint --port 8080
```

This starts a Flask-based local endpoint in the background that's compatible with SageMaker's protocol.

### Running Inference (Focus Area)

```bash
python inference.py --endpoint-name my-endpoint --method direct
```

Inference supports multiple methods:
- `--method direct`: HTTP requests directly to the endpoint (fastest)
- `--method boto3`: Uses boto3 SageMaker client
- `--method predictor`: Uses SageMaker Python SDK Predictor

Direct model inference (without endpoint):
```bash
python inference.py --model-artifact model/model.tar.gz
```

The inference system automatically:
- Detects the model framework (scikit-learn or TensorFlow)
- Loads appropriate test data
- Computes comprehensive metrics (MSE, R², prediction differences)
- Handles serialization/deserialization appropriate for the model type

### Checking Job Status

```bash
python async_ml_workflow.py status
```

Displays the status of all background jobs, including:
- Job type (training, deployment, inference)
- Process ID
- Status (RUNNING, COMPLETED, FAILED)
- Start time
- Job-specific parameters
- Recent log entries

### Special Inference Features

#### TensorFlow Model Support

The inference system is optimized for TensorFlow models:
- Auto-detects TensorFlow SavedModel format
- Loads TF models using the correct serving signatures
- Handles TF-specific serialization requirements
- Processes both regression and classification models

#### Multiple Inference Patterns

1. **Endpoint-based inference**: Standard SageMaker pattern using HTTP endpoints
2. **Direct model loading**: Load models directly from artifacts without deploying
3. **Lambda-triggered inference**: Allows integration with Step Functions

#### Performance Metrics

The inference system generates detailed model performance metrics:
- Mean Squared Error (MSE)
- R-squared (coefficient of determination)
- Sample-by-sample prediction differences
- Comparison visualizations (when using Jupyter)

## Important Files and Components

The system consists of these key files:

1. **async_ml_workflow.py** - **MAIN ENTRY POINT**: Central script for all asynchronous operations
   - Handles background job management
   - Provides unified command interface
   - Tracks job status and process management

2. **local_train.py**: Training logic
   - Supports multiple model types (random_forest, linear_regression, tensorflow)
   - Creates model artifacts in standardized format
   - Includes Lambda handler for Step Functions integration

3. **local_serve.py**: Model deployment logic
   - Creates Flask-based local endpoints
   - Maps model types to appropriate serving logic
   - Includes Lambda handler for Step Functions integration
   
4. **inference.py**: **INFERENCE FOCUS**
   - Provides multiple inference methods (direct, boto3, predictor)
   - Auto-detects model frameworks
   - Generates comprehensive metrics
   - Supports both endpoint and direct model inference
   - Includes Lambda handler for workflow integration

5. **localstack_ml_workflow.py**: LocalStack integration
   - Creates S3 buckets with proper configuration
   - Sets up IAM roles
   - Configures proper S3 endpoint options

## LocalStack Integration Notes

When working with LocalStack for inference:

1. **S3 Configuration**:
   - S3 acceleration must be disabled
   - Path-style addressing must be used
   - LocalStack endpoint: http://localhost:4567

2. **Environment Variables**:
   ```
   S3_USE_ACCELERATE_ENDPOINT=false
   AWS_S3_DISABLE_ACCELERATE_ENDPOINT=true
   ```

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
