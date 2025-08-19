import logging

from src.scripts.scripts_for_steps.trainer_module import TransformerTrainerWrapper

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

def train_transformer_model(trainer: TransformerTrainerWrapper, run_name: str, experiment_name:str, model_flavor="pytorch"):
    logger.info("Training step started")

    trainer.train(run_name, experiment_name, model_flavor)