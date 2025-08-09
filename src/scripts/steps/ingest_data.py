import logging

from src.scripts.scripts_for_steps.ingest_data_strategy import HuggingFaceImageDataIngestorStrategy

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

def ingest_data_from_hf(owner:str, dataset_name:str):
    try:
        ingestor = HuggingFaceImageDataIngestorStrategy()

        dataset = ingestor.ingest_data(owner=owner, dataset_name=dataset_name)

        if dataset is None or "train" not in dataset:
            raise ValueError("Invalid dataset structure")

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")