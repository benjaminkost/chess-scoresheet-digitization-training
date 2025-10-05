import mlflow
from dotenv import load_dotenv
from transformers import VisionEncoderDecoderModel
from transformers import Seq2SeqTrainingArguments

from src.scripts.steps.connect_to_dagshub import connect_to_dagshub


def log_manually_single_run_in_dagshub(experiment_name, dir, run_name, args=None):
    load_dotenv()
    connect_to_dagshub()

    print(mlflow.get_tracking_uri())

    # Setze das MLflow-Experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Modell laden
        model = VisionEncoderDecoderModel.from_pretrained(dir)

        # Modell loggen
        mlflow.pytorch.log_model(model, artifact_path="model")

        # Logge die Checkpoint-Nummer
        step = int(dir.split("-")[-1])
        mlflow.log_param("checkpoint", dir)
        mlflow.log_param("step", step)

        # TrainingArguments loggen
        for arg_name, value in vars(args).items():
            # Einige Objekte (wie Logging dir) sind nicht JSON-kompatibel → überspringen
            try:
                mlflow.log_param(arg_name, value)
            except:
                continue

        # Optional: Tags oder weitere Metadaten
        mlflow.set_tag("logged_late", "true")

if __name__ == "__main__":
    experiment_name = "TrOCR_Nachträgliche_Checkpoints"
    checkpoint_dir = "./training-results/checkpoint-4000"
    run_name = "checkpoint-4000"

    training_args = Seq2SeqTrainingArguments(
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
    )
    log_manually_single_run_in_dagshub(experiment_name,checkpoint_dir, run_name, args=training_args)