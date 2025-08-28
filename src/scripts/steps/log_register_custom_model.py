import logging
import mlflow
from dotenv import load_dotenv
from pathlib import Path

from src.scripts.steps.connect_to_dagshub import connect_to_dagshub

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def log_register_custom_model(python_model, conda_env, artifacts, artifact_path, registered_model_name) -> None:
    # Load enviroment variables
    load_dotenv()

    # Set up mlflow experiment enviroment to dagshub
    connect_to_dagshub()

    # code path
    code_path = str(Path.cwd() / "mlflow_models")

    with mlflow.start_run():
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        # log the Python function model
        mlflow.pyfunc.log_model(
            python_model=python_model,
            conda_env=conda_env,
            artifacts=artifacts,
            code_paths=[code_path],
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )