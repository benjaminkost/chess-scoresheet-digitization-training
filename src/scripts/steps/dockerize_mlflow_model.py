import logging
import os
import subprocess
import mlflow
from dotenv import load_dotenv
from zenml import step
import dagshub

@step
def dockerize_mlflow_model_with_cli(docker_image_name, mlflow_model_name: str):
    dagshub.init(repo_owner=os.environ["DAGSHUB_MLFLOW_TRACKING_USERNAME"], repo_name=os.environ["DAGSHUB_REPOSITORY"], mlflow=True)

    if len(docker_image_name.split("/")) != 3:
        load_dotenv()
        docker_username = os.environ["DOCKER_USERNAME"]
        docker_image_version = os.environ["DOCKER_IMAGE_VERSION"]

        docker_image_name = f"{docker_username}/{docker_image_name}:{docker_image_version}"

    # Get latest mlflow model uri of the model name defined in input
    ## Get the latest version for the model
    client = mlflow.MlflowClient()
    model_version = client.get_latest_versions(name=mlflow_model_name)[0].version

    # Construct the model URI
    model_uri = f'models:/{mlflow_model_name}/{model_version}'

    print(f"model_uri: {model_uri}")

    logging.info(f"model_uri: {model_uri}")

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

@step
def dockerize_mlflow_model_with_python(docker_image_name, mlflow_model_name: str):
    dagshub.init(repo_owner=os.environ["DAGSHUB_MLFLOW_TRACKING_USERNAME"], repo_name=os.environ["DAGSHUB_REPOSITORY"], mlflow=True)

    if len(docker_image_name.split("/")) != 3:
        load_dotenv()
        docker_username = os.environ["DOCKER_USERNAME"]
        docker_image_version = os.environ["DOCKER_IMAGE_VERSION"]

        docker_image_name = f"{docker_username}/{docker_image_name}:{docker_image_version}"

        # Get latest mlflow model uri of the model name defined in input
        ## Get the latest version for the model
    client = mlflow.MlflowClient()
    model_version = client.get_latest_versions(name=mlflow_model_name)[0].version

    # Construct the model URI
    model_uri = f'models:/{mlflow_model_name}/{model_version}'

    print(f"model_uri: {model_uri}")

    logging.info(f"model_uri: {model_uri}")

    # Use mlflow method
    mlflow.models.build_docker(
        model_uri=model_uri,
        name=docker_image_name,
        enable_mlserver=True
    )

    # mlflow_home="/opt/conda/lib/python3.11/site-packages/mlflow",
    # extra_pip_requirements=["torch==2.0.1", "transformers==4.51.3", "numpy==1.23.5", "opencv-python==4.11.0.86", "pandas==2.2.3", "Pillow==11.2.1", "scikit-learn==1.6.1", "starlette==0.46.2"]
    # install_mlflow=False,