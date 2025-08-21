import logging

import mlflow

from src.scripts.steps.log_standard_model import log_standard_model
from src.scripts.steps.connect_to_dagshub import connect_to_dagshub

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

def tracking_standard_model_pipeline(tracking_uri: str, experiment_name:str, run_name:str, model, model_flavor: str,
                                     tracking_with_dagshub=False):
    # Set tracking uri
    if tracking_with_dagshub:
        connect_to_dagshub()
    else:
        mlflow.set_tracking_uri(tracking_uri)

    logger.info(f"Tracking URI set to {mlflow.get_tracking_uri()}")

    # log model
    log_standard_model(experiment_name=experiment_name, run_name=run_name, model=model, model_flavor=model_flavor)



