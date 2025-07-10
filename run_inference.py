import click
from PIL import Image

from app.ml.pipelines.inference_pipeline import inference_pipeline

@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def run_main(stop_service: bool):
    # Define file path
    file_path = "../../data/raw_data/unprocessed_hcs_data/images/001_1.png"

    # Define model name
    model_name = "trocr-base-handwritten-with-pre-and-post-processing"

    # get prediction pgn string
    pgn_str = inference_pipeline(file_path, model_name)

    # Show image and pgn prediction
    file = Image.open(file_path)
    file.show()
    print(pgn_str)

if __name__ == "__main__":
    run_main()
