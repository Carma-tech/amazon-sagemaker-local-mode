# Running SageMaker Local Mode in Docker

This document provides instructions for building and running the SageMaker Local Mode examples using Docker. The Dockerfile in this repository creates a containerized environment with all necessary dependencies pre-installed, making it easy to run the examples without manual setup.

## Prerequisites

- Docker installed on your machine
- Docker daemon running

## Building the Docker Image

1. Navigate to the repository root directory:
   ```bash
   cd /path/to/amazon-sagemaker-local-mode
   ```

2. Build the Docker image:
   ```bash
   docker build -t sagemaker-local-mode .
   ```

   This will create a Docker image named `sagemaker-local-mode` with all the required dependencies installed.

## Running Examples

### Important: AWS Region Configuration

SageMaker Local Mode requires an AWS region to be set, even when running locally. You must specify a region when running the container using the `-e AWS_REGION=<region>` flag.

### Default Example

To run the default example (TensorFlow California Housing):

```bash
docker run -it --rm \
  -e AWS_REGION=us-east-1 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  sagemaker-local-mode
```

### Running a Specific Example

To run a specific example, provide the path to the example script as an argument:

```bash
docker run -it --rm \
  -e AWS_REGION=us-east-1 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  sagemaker-local-mode \
  pytorch_script_mode_local_training_and_serving/pytorch_script_mode_local_training_and_serving.py
```

### Examples You Can Run

Here are some examples you can run:

1. **TensorFlow California Housing** (default):
   ```bash
   docker run -it --rm -e AWS_REGION=us-east-1 -v /var/run/docker.sock:/var/run/docker.sock sagemaker-local-mode
   ```

2. **PyTorch CIFAR-10**:
   ```bash
   docker run -it --rm -e AWS_REGION=us-east-1 -v /var/run/docker.sock:/var/run/docker.sock sagemaker-local-mode pytorch_script_mode_local_training_and_serving/pytorch_script_mode_local_training_and_serving.py
   ```

3. **XGBoost**:
   ```bash
   docker run -it --rm -e AWS_REGION=us-east-1 -v /var/run/docker.sock:/var/run/docker.sock sagemaker-local-mode xgboost_script_mode_local_training_and_serving/xgboost_script_mode_local_training_and_serving.py
   ```

## Using Your AWS Credentials

If you need to use your own AWS credentials for pulling images from ECR, you can mount your credentials into the container:

```bash
docker run -it --rm \
  -e AWS_REGION=us-east-1 \
  -v ~/.aws:/root/.aws:ro \
  -v /var/run/docker.sock:/var/run/docker.sock \
  sagemaker-local-mode
```

## Important Notes

- The `-v /var/run/docker.sock:/var/run/docker.sock` mount is required because SageMaker local mode needs to create Docker containers.
- The `-e AWS_REGION=us-east-1` flag is required for SageMaker Local Mode to function properly. Replace with your preferred AWS region.
- Each example will download the necessary SageMaker framework containers from Amazon ECR automatically. The first run may take some time to download these images.
- The container will create Docker sibling containers on your host machine for training and inference.
- Training and model data are stored within the containers.

## For Windows Users

For Windows, modify the Docker run command as follows:

```bash
docker run -it --rm ^
  -e AWS_REGION=us-east-1 ^
  -v //var/run/docker.sock:/var/run/docker.sock ^
  sagemaker-local-mode
```

## Troubleshooting

1. **Error: Cannot connect to the Docker daemon**
   - Make sure the Docker daemon is running on your host.
   - Ensure the Docker socket is correctly mounted.

2. **Error: Permission denied on Docker socket**
   - You may need to run the Docker command with sudo or adjust permissions on your host.

3. **AWS Region Error**
   - If you see an error about AWS configuration, make sure you're passing the AWS_REGION environment variable.

4. **Container startup issues**
   - Check if your AWS configuration is correct if container images fail to download.
   - Some examples may require more memory than the default Docker settings allow.
