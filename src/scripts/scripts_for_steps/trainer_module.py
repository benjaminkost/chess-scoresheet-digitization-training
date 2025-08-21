import logging
from abc import ABC, abstractmethod

from datasets import Dataset
from evaluate import load
import mlflow
from transformers import ProcessorMixin, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, default_data_collator, Seq2SeqTrainer
from src.scripts.scripts_for_steps.hyperparameter_util import ModelHyperparameters

# Logger definition
# ANSI Escape Code for white letters
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


class Trainer(ModelHyperparameters, ABC):
    #getter/setter
    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass

    # business logic
    @abstractmethod
    def train(self, run_name: str, experiment_name: str, model_flavor="pytorch") -> None:
        """
        Train the model

        :param run_name: Name of the mlflow run
        :param experiment_name: Name of the mlflow experiment
        :param model_flavor: Type of model to train (pytorch, keras, pyfunc etc.)
        :return: None (model is tracked in mlflow)
        """
        pass


class TransformerTrainerWrapper(Trainer, ABC):

    # getter/setter
    @abstractmethod
    def get_trainer(self):
        pass

    @abstractmethod
    def get_training_args(self):
        pass

    @abstractmethod
    def get_processor(self):
        pass

    # business logic
    def compute_metrics(self, pred) -> dict:
        """
        Compute metrics based on the prediction result.

        :param pred: prediction from the model
        :return: dictionary of metrics
        """
        pass

    def set_up_trainer_args_class(self, *args, **kwargs):
        """
        Set up a trainer args class from a library like huggingfaces "transformers"
        """
        pass

    def set_up_trainer_class(self):
        """
        Set up a trainer class from a library like huggingfaces "transformers"
        """
        pass


class CNNTransformerTrainer(TransformerTrainerWrapper):

    def __init__(self,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 model: VisionEncoderDecoderModel,
                 processor: ProcessorMixin,
                 predict_with_generate:bool,
                 eval_strategy:str,
                 per_device_train_batch_size:int,
                 per_device_eval_batch_size:int,
                 fp16:bool,
                 output_dir:str,
                 logging_steps:int,
                 save_steps:int,
                 eval_steps:int,
                 disable_tqdm:bool,
                 report_to:str,
                 max_length=64,
                 early_stopping=True,
                 no_repeat_ngram_size=3,
                 length_penalty=2.0,
                 num_beams=4,
                 ):
        # Save processing classes
        self._model = model
        self._processor = processor
        self._training_args = self.set_up_trainer_args_class(
            predict_with_generate, eval_strategy, per_device_train_batch_size,
            per_device_eval_batch_size, fp16, output_dir, logging_steps, save_steps,
            eval_steps, disable_tqdm, report_to)

        # set later
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset

        self._trainer = self.set_up_trainer_class()

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model", "data_module", "processor", "trainer"])

        # set special tokens used for creating the decoder_input_ids from the labels
        self._model.config.decoder_start_token_id = self._processor.tokenizer.cls_token_id
        self._model.config.pad_token_id = self._processor.tokenizer.pad_token_id
        # make sure vocab size is set correctly
        self._model.config.vocab_size = model.config.decoder.vocab_size

        # Beam search params
        self._model.config.eos_token_id = self._processor.tokenizer.sep_token_id
        self._model.config.max_length = max_length
        self._model.config.early_stopping = early_stopping
        self._model.config.no_repeat_ngram_size = no_repeat_ngram_size
        self._model.config.length_penalty = length_penalty
        self._model.config.num_beams = num_beams

    def get_trainer(self):
        return self._trainer

    def get_training_args(self):
        return self._training_args

    def get_processor(self):
        return self._processor

    def get_train_dataset(self):
        return self._train_dataset

    def get_test_dataset(self):
        return self._test_dataset

    def get_model(self):
        return self._model

    def train(self, run_name: str, experiment_name: str, model_flavor="pytorch"):
        logger.info(f"MLflow uri is: {mlflow.get_tracking_uri()}")

        logger.info(f"Training model with flavor {model_flavor}")

        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow experiment set to '{experiment_name}'")

        # Log parameters, metrics, model signature
        mlflow.autolog()

        # Load model
        model = self.get_model()

        logger.info(f"Model was loaded")

        with mlflow.start_run(run_name=run_name):

            # Train model
            logger.info(f"Training starting")
            self.get_trainer().train()

            logger.info(f"Model was trained")

            unwrapped_model = self.get_trainer().accelerator.unwrap_model(model)
            model_metadata = mlflow.pytorch.log_model(unwrapped_model, artifact_path="model")

            logger.info(f"Model was logged with metadata: {model_metadata}")

    def compute_metrics(self, pred) -> dict:
        """
        Compute metrics based on the prediction result.

        :param pred: prediction from the model
        :return: dictionary of metrics
        """
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = self.get_processor().batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.get_processor().tokenizer.pad_token_id
        label_str = self.get_processor().batch_decode(labels_ids, skip_special_tokens=True)

        cer_metric = load("cer")
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    def set_up_trainer_args_class(self, predict_with_generate:bool, eval_strategy:str, per_device_train_batch_size:int,
                                  per_device_eval_batch_size:int, fp16:bool, output_dir:str, logging_steps:int,
                                  save_steps:int, eval_steps:int, disable_tqdm:bool, report_to:str) -> Seq2SeqTrainingArguments:
        """
        Set up a trainer args class Seq2SeqTrainingArguments from a library the huggingfaces "transformers"

        :return: Seq2SeqTrainingArguments
        """

        training_args = Seq2SeqTrainingArguments(
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
        )

        return training_args

    def set_up_trainer_class(self):
        """
        Set up a trainer class Seq2SeqTrainer from a library from huggingfaces "transformers"
        """

        trainer = Seq2SeqTrainer(
            model=self.get_model(),
            processing_class=self.get_processor(),
            args=self._training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=self._train_dataset,
            eval_dataset=self._test_dataset,
            data_collator=default_data_collator
        )

        return trainer
