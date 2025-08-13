from transformers import TrOCRProcessor
from src.scripts.pipelines.training_pipeline import training_pipeline_without_preprocessing_for_transformer_models

def run_training_pipeline():
    # Input data
    ## For Data
    owner = "BenjaminKost"
    dataset_name = "processed_hcs"
    split = "train"
    feature_column = "image"
    target_column = "label"

    ## For Trainer, Processor and Model
    processor_name = "trocr-base-handwritten-chess-notation-tokenizer"
    processor = TrOCRProcessor.from_pretrained(f"{owner}/{processor_name}")
    model_name = "TrOCR-Base-Fined-On-HCS"

    run_name = "Model-TrOCR-Base-Fined-On-HCS-Training-Nr_1"
    experiment_name = "Train for model digitalizing hand written chess game notations"
    model_flavor = "pytorch"
    tags = ["chess", "handwritten", "hcr", "ocr", "chess game notation"]

    # Call pipeline
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

if __name__ == "__main__":
    run_training_pipeline()