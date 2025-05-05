FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Default AWS region (will be overridden when container starts)
ENV AWS_REGION=us-east-1
ENV AWS_DEFAULT_REGION=us-east-1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    lsb-release \
    apt-transport-https \
    ca-certificates \
    software-properties-common \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /sagemaker-local-mode

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the repository
COPY . .

# Install example-specific requirements
RUN find . -name "requirements.txt" -not -path "./requirements.txt" -exec pip install --no-cache-dir -r {} \;

# Create an entrypoint script with AWS region checking
RUN echo '#!/bin/bash\n\
if [ -z "$AWS_REGION" ] && [ -z "$AWS_DEFAULT_REGION" ]; then\n\
  echo "ERROR: AWS_REGION or AWS_DEFAULT_REGION environment variable must be set."\n\
  echo "Run with: docker run -e AWS_REGION=<your-region> -v /var/run/docker.sock:/var/run/docker.sock sagemaker-local-mode"\n\
  exit 1\n\
fi\n\
\n\
if [ "$1" != "" ]; then\n\
  echo "Running example: $1"\n\
  python "$1"\n\
else\n\
  echo "No example specified, running default TensorFlow California Housing example"\n\
  python tensorflow_script_mode_california_housing_local_training_and_serving/tensorflow_script_mode_california_housing_local_training_and_serving.py\n\
fi' > /sagemaker-local-mode/entrypoint.sh \
    && chmod +x /sagemaker-local-mode/entrypoint.sh

# Set Docker socket environment variable
ENV DOCKER_HOST=unix:///var/run/docker.sock

# Default entrypoint
ENTRYPOINT ["/sagemaker-local-mode/entrypoint.sh"]
