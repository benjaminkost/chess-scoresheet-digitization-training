from src.scripts.scripts_for_steps.trainer_module import CNNTransformerTrainer


def training_pipeline_without_preprocessing_for_transformer_models():
    # train
    training_class = CNNTransformerTrainer()