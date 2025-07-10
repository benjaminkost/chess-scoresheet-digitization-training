import logging
import os
import dagshub
import mlflow
from zenml import step
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@step
def log_register_custom_model(python_model, conda_env, artifacts, artifact_path, registered_model_name) -> None:
    # Load enviroment variables
    load_dotenv()

    # Set up mlflow experiment enviroment to dagshub
    dagshub.init(repo_owner=os.environ["DAGSHUB_MLFLOW_TRACKING_USERNAME"], repo_name=os.environ["DAGSHUB_REPOSITORY"], mlflow=True)

    with mlflow.start_run():
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        # log the Python function model
        mlflow.pyfunc.log_model(
            python_model=python_model,
            conda_env=conda_env,
            artifacts=artifacts,
            code_paths=["../mlflow_models"],
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )

        # infer_code_paths=True,