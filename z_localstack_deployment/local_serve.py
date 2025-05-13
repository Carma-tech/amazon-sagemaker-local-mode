#!/usr/bin/env python3
"""
Local Model Server for Inference

This script:
1. Takes a trained model artifact (tar.gz)
2. Extracts it and serves it via a Flask API
3. Mimics the SageMaker endpoint interface for compatibility
"""

import os
import sys
import json
import time
import tarfile
import logging
import argparse
import tempfile
import pickle
import numpy as np
from pathlib import Path
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('local-serve')

# Constants
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


def save_model_outputs(model_outputs, deployment_outputs, file_path=None):
    """Save model and deployment outputs to a JSON file."""
    if not file_path:
        file_path = MODEL_OUTPUTS_FILE
    
    # Combine outputs while preserving existing values
    existing_outputs = load_model_outputs(file_path)
    outputs = {**existing_outputs, **model_outputs, **deployment_outputs}
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(outputs, f, indent=2)
    
    logger.info(f"Saved model outputs to {file_path}")
    return outputs


def extract_model(model_artifact, extract_dir):
    """Extract the model artifact to a directory."""
    try:
        logger.info(f"Extracting model from {model_artifact} to {extract_dir}")
        
        # If model_artifact starts with file://, remove it
        if model_artifact.startswith('file://'):
            model_artifact = model_artifact[7:]
            
        # Create extract directory
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract the tar.gz file
        with tarfile.open(model_artifact, 'r:gz') as tar:
            tar.extractall(path=extract_dir)
            
        logger.info(f"Model extracted successfully to {extract_dir}")
        return True
    except Exception as e:
        logger.error(f"Error extracting model: {str(e)}")
        return False


def load_model(model_dir):
    """Load a model from a directory."""
    try:
        # Look for pickle files in model directory
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if model_files:
            model_path = os.path.join(model_dir, model_files[0])
            logger.info(f"Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded model of type: {type(model)}")
            return model
        else:
            logger.error("No model file found in model directory")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


def start_flask_app(model, port, endpoint_name):
    """Start a Flask application to serve the model."""
    try:
        from flask import Flask, request, jsonify
        import threading
        
        app = Flask(endpoint_name)
        
        @app.route('/ping', methods=['GET'])
        def ping():
            # Health check endpoint
            if model is None:
                return jsonify({"status": "error", "message": "Model not loaded"}), 500
            return jsonify({"status": "ok"}), 200
        
        @app.route('/invocations', methods=['POST'])
        def invoke():
            # Endpoint for model predictions, mimicking SageMaker
            if model is None:
                return jsonify({"status": "error", "message": "Model not loaded"}), 500
            
            # Parse input data
            if request.content_type == 'application/json':
                try:
                    data = request.get_json()
                    logger.info(f"Received request: {data}")
                    
                    # Handle different input formats
                    if isinstance(data, dict) and 'instances' in data:
                        # SageMaker batch transform format
                        instances = np.array(data['instances'])
                    elif isinstance(data, list):
                        # Direct list format
                        instances = np.array(data)
                    else:
                        return jsonify({"status": "error", "message": f"Unsupported format: {type(data)}"}), 400
                    
                    logger.info(f"Input shape: {instances.shape}")
                    
                    # Make predictions
                    predictions = model.predict(instances)
                    
                    # Format response
                    response = {
                        "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
                    }
                    
                    logger.info(f"Generated predictions: {predictions[:5]}")
                    return jsonify(response)
                    
                except Exception as e:
                    logger.error(f"Error during prediction: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return jsonify({"status": "error", "message": str(e)}), 500
            else:
                return jsonify({"status": "error", "message": f"Unsupported content type: {request.content_type}"}), 400
        
        # Start Flask app in a thread
        thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=port, debug=False))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Flask app started on port {port}")
        return thread
    
    except Exception as e:
        logger.error(f"Error starting Flask app: {str(e)}")
        return None


def deploy_model(model_artifact=None, endpoint_name=None, port=8080):
    """Deploy the model to a local Flask server."""
    logger.info("Starting local model deployment")
    
    # Load model outputs if no artifact provided
    if not model_artifact:
        model_outputs = load_model_outputs()
        if 'model_artifact' in model_outputs:
            model_artifact = model_outputs['model_artifact']
            logger.info(f"Using model artifact from outputs: {model_artifact}")
        else:
            logger.error("No model artifact provided and none found in model outputs")
            return None
    
    # Generate endpoint name if not provided
    if not endpoint_name:
        endpoint_name = f"local-endpoint-{int(time.time())}"
    
    logger.info(f"Deploying to endpoint: {endpoint_name} on port {port}")
    
    # Create temporary directory for model extraction
    extract_dir = tempfile.mkdtemp()
    
    try:
        # Extract the model
        if not extract_model(model_artifact, extract_dir):
            return None
        
        # Load the model
        model = load_model(extract_dir)
        if model is None:
            return None
        
        # Start Flask app
        thread = start_flask_app(model, port, endpoint_name)
        if thread is None:
            return None
        
        # Store deployment information
        deployment_outputs = {
            'endpoint_name': endpoint_name,
            'endpoint_status': 'InService',
            'deployment_timestamp': time.time(),
            'endpoint_url': f'http://localhost:{port}/invocations',
            'port': port
        }
        
        # Update model outputs
        model_outputs = {
            'model_artifact': model_artifact
        }
        
        # Save the combined outputs
        save_model_outputs(model_outputs, deployment_outputs)
        
        logger.info(f"Model deployed successfully to endpoint: {endpoint_name}")
        logger.info(f"To test the endpoint: python inference.py --endpoint-name {endpoint_name} --endpoint-url http://localhost:{port}/invocations")
        
        # Keep the script running to maintain the Flask app
        logger.info("Press Ctrl+C to stop the server")
        while True:
            time.sleep(1)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error in deployment: {str(e)}")
    finally:
        # Always clean up the temporary directory
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Deploy a model using a local Flask server')
    parser.add_argument('--model-artifact', type=str, help='Path to model artifact (tar.gz file)')
    parser.add_argument('--endpoint-name', type=str, help='Name for the endpoint')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    
    args = parser.parse_args()
    
    # Deploy the model
    deploy_model(
        model_artifact=args.model_artifact,
        endpoint_name=args.endpoint_name,
        port=args.port
    )


def lambda_handler(event, context=None):
    """AWS Lambda handler for model deployment."""
    try:
        logger.info(f"Received Lambda event: {event}")
        
        # Extract parameters from the event
        model_artifact = event.get('model_artifact')
        endpoint_name = event.get('endpoint_name')
        port = event.get('port', 8080)
        
        # Run in a separate process to avoid blocking
        import subprocess
        import sys
        
        cmd = [sys.executable, __file__]
        if model_artifact:
            cmd.extend(['--model-artifact', model_artifact])
        if endpoint_name:
            cmd.extend(['--endpoint-name', endpoint_name])
        cmd.extend(['--port', str(port)])
        
        # Start the process in the background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait a little to allow the process to start
        import time
        time.sleep(5)
        
        # Get some initial output
        stdout, stderr = process.stdout.readline(), process.stderr.readline()
        
        return {
            'statusCode': 200,
            'body': {
                'endpoint_name': endpoint_name,
                'endpoint_url': f'http://localhost:{port}/invocations',
                'model_artifact': model_artifact,
                'process_id': process.pid,
                'initial_output': stdout,
                'initial_error': stderr if stderr else None
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
