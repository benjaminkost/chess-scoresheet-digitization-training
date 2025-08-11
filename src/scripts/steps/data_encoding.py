import logging

from src.scripts.scripts_for_steps.preprocessing_strategy import EncodingStrategy, TrOCREncoder

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

def encode_dataset(processor, dataset, split, feature_column, target_column):
    """
    Encode the labels so that the labels are correctly formatted for training.
    This reassigns the _dataset attribute.

    """
    encoding_strategy = TrOCREncoder(processor)
    encoded_dataset = encoding_strategy.encode_dataset(dataset, split, feature_column, target_column)

    logger.info(f"Dataset was encoded")

    return encoded_dataset