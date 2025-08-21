import logging

from src.scripts.scripts_for_steps.preprocessing_strategy import PreprocessingStrategy, HuggingFacePreprocessingStrategy

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

def preprocess_dataset(dataset):
    preprocessing_strategy = HuggingFacePreprocessingStrategy()

    preprocessed_dataset = preprocessing_strategy.preprocess_dataset(dataset)

    logger.info(f'Dataset was preprocessed with shape of {preprocessed_dataset.shape}')

    return preprocessed_dataset