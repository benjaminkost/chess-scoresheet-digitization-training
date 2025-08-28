import mlflow

from src.scripts.steps.connect_to_dagshub import connect_to_dagshub
from src.scripts.steps.dockerize_mlflow_model import dockerize_mlflow_model_with_python
from src.scripts.steps.push_docker_image_to_docker_hub_step import push_docker_image_to_docker_hub

def deployment_pipeline(tracking_uri:str, docker_image_name: str, model_name: str,tracking_with_dagshub=False):
    if tracking_with_dagshub:
        connect_to_dagshub()
    else:
        mlflow.set_tracking_uri(tracking_uri)

    # Create a Docker image out of the model
    dockerize_mlflow_model_with_python(docker_image_name, model_name)

    # Push Docker image to docker hub
    push_docker_image_to_docker_hub()