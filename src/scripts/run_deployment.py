import os
import dagshub
from dotenv import load_dotenv
from src.scripts.deployment_pipeline import deployment_pipeline

def run_main():
    """Run the image predictor deployment pipeline"""
    # load enviroment variables
    load_dotenv()

    env_model_name = (os.environ["MODEL_NAMES"]
                  .replace("\"", "")
                  .replace("(", "")
                  .replace("(", "")
                  .replace(")", ""))

    env_images = (os.environ["IMAGES"]
                  .replace("\"", "")
                  .replace("(", "")
                  .replace("(", "")
                  .replace(")", ""))

    model_name = env_model_name
    docker_image_name = env_images

    # Set up mlflow experiment enviroment to dagshub
    dagshub.init(repo_owner=os.environ["DAGSHUB_MLFLOW_TRACKING_USERNAME"], repo_name=os.environ["DAGSHUB_REPOSITORY"], mlflow=True)

    # run deployment pipeline
    deployment_pipeline(docker_image_name, model_name)

if __name__ == "__main__":
    run_main()