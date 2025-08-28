import logging
import mlflow
from dotenv import load_dotenv

from src.scripts.steps.connect_to_dagshub import connect_to_dagshub

# Logger definition
# ANSI Escape Code for white letters
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

def log_standard_model(experiment_name:str, run_name:str, model_flavor:str, model) -> None:
    # Set experiment
    mlflow.set_experiment(experiment_name)

    # Load environment variables
    load_dotenv()

    # Set up mlflow experiment environment to dagshub
    connect_to_dagshub()

    with mlflow.start_run(run_name=run_name):
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

        # log the Python function model
        if model_flavor == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path="model")

            logger.info(f"MLflow PyTorch model from path '{model}' was successfully logged")
        else:
            logger.error(f"Model flavor {model_flavor} is not supported.")