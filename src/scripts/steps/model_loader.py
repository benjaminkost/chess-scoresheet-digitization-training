import logging
import os
import dagshub
import mlflow
from dotenv import load_dotenv
from transformers import VisionEncoderDecoderModel

# configure logger
logging.basicConfig(
    level=logging.INFO,  # Log-Ebene (z. B. DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log-Format
)
logger = logging.getLogger(__name__)  # Logger mit Modulnamen beziehen

def load_model(model_name: str):
    """

    :return: loaded a model
    """
    # set mlflow tracking uri
    load_dotenv()

    dagshub.init(repo_owner=os.getenv("DAGSHUB_MLFLOW_TRACKING_USERNAME"), repo_name=os.environ["DAGSHUB_REPOSITORY"], mlflow=True)

    # Get the latest version for the model
    logger.info(f"Getting latest version for model: {model_name}")
    client = mlflow.MlflowClient()

    if len(client.search_registered_models(filter_string=f"name='{model_name}'")) > 0:
        model_version = client.get_latest_versions(name=model_name)[0].version

        # Construct the model URI
        logger.info(f"Constructing model URI")
        model_uri = f'models:/{model_name}/{model_version}'

        # Load the model
        logger.info(f"Loading model: {model_uri}")
        logger.info(f"Mlflow tracking uri: {mlflow.get_tracking_uri()}")
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        logger.info(f"Model loaded successfully")
    else:
        logger.info(f"Model with this name does not exist in mlflow tracking uri: {mlflow.get_tracking_uri()}")

        default_model_uri = os.getenv("HF_MODEL_URI")

        logger.info(f"Default model with Huggingface URI '{default_model_uri}' will be loaded")

        model = VisionEncoderDecoderModel.from_pretrained(default_model_uri)

        logger.info(f"Model loaded successfully")

    return model