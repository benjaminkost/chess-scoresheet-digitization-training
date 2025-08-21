import logging
import os
import subprocess
import mlflow
from dotenv import load_dotenv

WHITE = "\033[37m"
RESET = "\033[0m"

# Logger configure
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console-Handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Formatter with ANSI Escape Code for white letters
formatter = logging.Formatter(f'{WHITE}%(asctime)s - %(name)s - %(levelname)s - %(message)s{RESET}')
handler.setFormatter(formatter)

# Handler for Logger added
logger.addHandler(handler)

def dockerize_mlflow_model_with_cli(docker_image_name, mlflow_model_name: str):

    if len(docker_image_name.split("/")) != 3:
        load_dotenv()
        docker_username = os.getenv("DOCKER_USERNAME")
        docker_image_version = os.getenv("DOCKER_IMAGE_VERSION")

        docker_image_name = f"{docker_username}/{docker_image_name}:{docker_image_version}"

    # Get latest mlflow model uri of the model name defined in input
    ## Get the latest version for the model
    client = mlflow.MlflowClient()
    model_version = client.get_latest_versions(name=mlflow_model_name)[0].version

    # Construct the model URI
    model_uri = f'models:/{mlflow_model_name}/{model_version}'

    logger.info(f"model_uri: {model_uri}")

    # Use mlflow command to create a docker image out of a register mlflow model
    cmd = [
        "mlflow", "models", "build-docker", "--name",
        docker_image_name, "--model-uri", model_uri
    ]

    # Execute command
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check for error
    if result.returncode != 0:
        logging.error(f"Something went wrong with creating the docker image out of {model_uri}")
        raise RuntimeError(f"Something went wrong with creating the docker image out of {model_uri}:\n{result.stderr}")
    else:
        logging.info(f"Creating docker image {docker_image_name} was successful:\n{result.stderr}")

def dockerize_mlflow_model_with_python(docker_image_name, mlflow_model_name: str):
    if len(docker_image_name.split("/")) != 3:
        load_dotenv()
        docker_username = os.getenv("DOCKER_USERNAME")
        docker_image_version = os.getenv("DOCKER_IMAGE_VERSION")

        docker_image_name = f"{docker_username}/{docker_image_name}:{docker_image_version}"

        # Get latest mlflow model uri of the model name defined in input
        ## Get the latest version for the model
    client = mlflow.MlflowClient()
    model_version = client.get_latest_versions(name=mlflow_model_name)[0].version

    # Construct the model URI
    model_uri = f'models:/{mlflow_model_name}/{model_version}'

    logger.info(f"model_uri: {model_uri}")

    # Use mlflow method
    mlflow.models.build_docker(
        model_uri=model_uri,
        name=docker_image_name,
        enable_mlserver=True
    )