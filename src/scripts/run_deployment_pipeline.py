import os
from pathlib import Path

from dotenv import load_dotenv
from src.scripts.pipelines.deployment_pipeline import deployment_pipeline


def run_deployment_pipeline():
    """Run the image predictor deployment pipeline"""
    # load enviroment variables
    load_dotenv()

    model_name = os.getenv("MODEL_NAME")
    model_flavor = "pytorch"
    docker_image_name = os.getenv("DOCKER_IMAGE_NAME")

    # Set up mlflow experiment enviroment to dagshub
    tracking_uri = str(Path(__file__).parent.parent.parent / "mlruns")


    # run deployment pipeline
    deployment_pipeline(tracking_uri=tracking_uri, docker_image_name=docker_image_name, model_name=model_name)

if __name__ == "__main__":
    run_deployment_pipeline()