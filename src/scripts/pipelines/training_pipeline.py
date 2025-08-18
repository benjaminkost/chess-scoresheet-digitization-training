import logging

from src.scripts.scripts_for_steps.trainer_module import CNNTransformerTrainer, TransformerTrainerWrapper
from src.scripts.steps.data_encoding import encode_dataset
from src.scripts.steps.data_splitter import split_train_test
from src.scripts.steps.ingest_data import ingest_data_from_hf
from src.scripts.steps.model_loader import load_model
from src.scripts.steps.train_model import train_transformer_model

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

def training_pipeline_without_preprocessing_for_transformer_models(owner:str, dataset_name:str, split:str, feature_column:str, target_column:str, processor, model_name: str, run_name: str, experiment_name:str, model_flavor:str, tags,
                                                                   predict_with_generate=True,
                                                                   eval_strategy="steps",
                                                                   per_device_train_batch_size=8,
                                                                   per_device_eval_batch_size=8,
                                                                   fp16=True,
                                                                   output_dir="./results",
                                                                   logging_steps=2,
                                                                   save_steps=1000,
                                                                   eval_steps=100,
                                                                   disable_tqdm=False,
                                                                   report_to="none",
                                                                   max_length=64,
                                                                   early_stopping=True,
                                                                   no_repeat_ngram_size=3,
                                                                   length_penalty=2.0,
                                                                   num_beams=4,
                                                                   ):
    # Ingest Data

    try:
        raw_dataset = ingest_data_from_hf(owner=owner, dataset_name=dataset_name)
    except:
        pass

    # Split dataset
    logger.info(f"Test if module was newly imported successfully")
    train_dataset, test_dataset = split_train_test(raw_dataset, split, feature_column, target_column)

    # Encode datasets
    encoded_train_dataset = encode_dataset(processor, train_dataset, feature_column, target_column)
    encoded_test_dataset = encode_dataset(processor, test_dataset, feature_column, target_column)

    # Load model
    model = load_model(model_name)

    # train
    trainer = CNNTransformerTrainer(
        train_dataset=encoded_train_dataset,
        test_dataset=encoded_test_dataset,
        model=model,
        processor=processor,
        predict_with_generate=predict_with_generate,
        eval_strategy=eval_strategy,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=fp16,
        output_dir=output_dir,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        disable_tqdm=disable_tqdm,
        report_to=report_to,
        max_length = max_length,
        early_stopping = early_stopping,
        no_repeat_ngram_size = no_repeat_ngram_size,
        length_penalty = length_penalty,
        num_beams = num_beams,
    )

    train_transformer_model(trainer, run_name, experiment_name, model_flavor, tags)