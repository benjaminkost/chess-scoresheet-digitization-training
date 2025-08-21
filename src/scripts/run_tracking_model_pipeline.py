from pathlib import Path
from transformers import VisionEncoderDecoderModel

from src.scripts.pipelines.tracking_model_pipeline import tracking_standard_model_pipeline

def log_and_register_model_pipeline():
    tracking_uri = str(Path(__file__).parent.parent.parent / "mlruns")
    run_name = "Model-TrOCR-Base-Fined-On-HCS-Insert-after-training"
    experiment_name = "Train for model digitalizing hand written chess game notations"
    model_flavor = "pytorch"
    path_to_model = str(Path(__file__).parent.parent.parent / "models" / "checkpoint-4119")
    model = VisionEncoderDecoderModel.from_pretrained(path_to_model)

    tracking_standard_model_pipeline(tracking_uri=tracking_uri, experiment_name=experiment_name, run_name=run_name, model=model, model_flavor=model_flavor)

if __name__ == "__main__":
    log_and_register_model_pipeline()