import json
import mlflow
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class HFTransformerImageModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None
        self.processor = None

    def load_context(self, context):
        # Get huggingface model name from context
        with open(context.artifacts["hf_model"], "r") as f:
            load_json_file = json.load(f)
            hf_model_owner = load_json_file["hf_model_owner"]
            hf_model_name = load_json_file["hf_model_name"]
        
        hf_model_uri = f"{hf_model_owner}/{hf_model_name}"

        # Initialize Model
        self.processor = TrOCRProcessor.from_pretrained(hf_model_uri)
        self.model = VisionEncoderDecoderModel.from_pretrained(hf_model_uri)

    def predict(self, model_input) -> dict:

        # Generate pixel_values with processor
        pixel_values = self.processor(model_input, return_tensors="pt").pixel_values

        # Generate ids from model
        generated_ids = self.model.generate(pixel_values)

        # Decode the result from the model
        prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return { "prediction" : prediction }

