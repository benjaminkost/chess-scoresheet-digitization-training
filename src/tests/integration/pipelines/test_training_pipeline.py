import pytest
from datasets import Dataset, DatasetDict
from PIL import Image
from transformers import TrOCRProcessor
import torch
import pandas as pd
from importlib.resources import files

from src.scripts.pipelines.training_pipeline import training_pipeline_without_preprocessing_for_transformer_models

@pytest.fixture
def set_up():
    return None

def test_training_pipeline_without_preprocessing_for_transformer_models_integration_load_model_and_training_should_throw_no_exceptions(set_up, mocker):
    # Give
    owner = "BenjaminKost"
    dataset_name = "processed_hcs"
    split = "train"
    feature_column = "image"
    target_column = "label"

    processor_name = "trocr-base-handwritten-chess-notation-tokenizer"
    processor = TrOCRProcessor.from_pretrained(f"{owner}/{processor_name}")
    model_name = "TrOCR-Base-Fined-On-HCS"

    run_name = "Model-TrOCR-Base-Fined-On-HCS-Training-Nr_1"
    experiment_name = "Train for model digitalizing hand written chess game notations"
    model_flavor = "pytorch"
    tags = ["chess", "handwritten", "hcr", "ocr", "chess game notation"]

    ## mocks
    data_path = files("src.tests.data")

    image = Image.open(f"{data_path}/001_0_1_black.png")
    dataset_sample = Dataset.from_dict(
        {
         "image": [image],
         "label": ["Nf6"],
        })
    raw_dataset_sample = DatasetDict({"train": dataset_sample})

    mock_ingest_data = mocker.patch("src.scripts.pipelines.training_pipeline.ingest_data_from_hf")
    mock_ingest_data.return_value = raw_dataset_sample

    mock_train_test_dataset = mocker.patch("src.scripts.pipelines.training_pipeline.split_train_test")
    mock_train_test_dataset.return_value = (raw_dataset_sample, raw_dataset_sample)

    mock_encode_dataset = mocker.patch("src.scripts.pipelines.training_pipeline.encode_dataset")
    df_encoded_label = pd.read_csv(f"{data_path}/encoded_label_for_Nf6.csv")
    df_encoded_image = pd.read_csv(f"{data_path}/encoded_image_001_0_1_black.csv")
    encoded_label = torch.Tensor(df_encoded_label.values)
    encoded_image = torch.Tensor(df_encoded_image.values)

    encoded_dataset_sample = Dataset.from_dict({
        "image": [encoded_image],
        "label": [encoded_label],
    })

    mock_encode_dataset.return_value = encoded_dataset_sample

    # When and Then
    try:
        training_pipeline_without_preprocessing_for_transformer_models(
            owner=owner,
            dataset_name=dataset_name,
            split=split,
            feature_column=feature_column,
            target_column=target_column,
            processor=processor,
            model_name=model_name,
            run_name=run_name,
            experiment_name=experiment_name,
            model_flavor=model_flavor,
            tags=tags,
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
        )
        mock_ingest_data.assert_called_once()
    except Exception as ex:
        pytest.fail(f"Training pipeline failed. Error: {ex}")