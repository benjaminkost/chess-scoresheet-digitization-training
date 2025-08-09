import logging
from abc import ABC, abstractmethod

import mlflow

from src.scripts.scripts_for_steps.data_module import DataModule, DataModuleTransformer
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

class Trainer(ABC, ModelHyperparameters):
    #getter/setter
    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def set_dataset(self, dataset):
        pass

    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def set_train_dataset(self, train_dataset):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass

    @abstractmethod
    def set_test_dataset(self, test_dataset):
        pass

    # business logic
    @abstractmethod
    def ingest_data(self, datasource):
        pass

    @abstractmethod
    def preprocess_dataset(self):
        pass

    @abstractmethod
    def encode_labels(self, split, feature_column, target_column):
        pass

    @abstractmethod
    def split_train_test(self, split, feature_column, target_column):
        pass

    @abstractmethod
    def convert_feature_label_lists_into_dict(self, feature_list: list, label_list: list, feature_name: str, label_name: str):
        pass

    @abstractmethod
    def train(self, run_name: str, experiment_name:str, model_flavor="pytorch", tags=None):
        pass

    @abstractmethod
    def save_model(self):
        pass

class TransformerTrainer(Trainer, ABC):

    @abstractmethod
    def __init__(self,
                 model,
                 data_module: DataModuleTransformer,
                 processor,
                 trainer=None,
                 training_args=None
                 ):
        self._model = model
        self._data_module = data_module
        self._processor = processor
        self._trainer = trainer
        self._training_args = training_args

        # set later
        self._dataset = None
        self._train_dataset = None
        self._test_dataset = None

    # getter/setter
    @abstractmethod
    def get_trainer(self):
        pass

    @abstractmethod
    def get_training_args(self):
        pass

    @abstractmethod
    def get_model_config(self):
        pass

    @abstractmethod
    def set_model_config(self, model_config):
        pass

    @abstractmethod
    def get_processor(self):
        pass

    # business logic

    @abstractmethod
    def tokenize_features(self):
        pass

class CNNTransformerTrainer(TransformerTrainer):

    def __init__(self,
                 model,
                 data_module: DataModuleTransformer,
                 processor,
                 trainer,
                 training_args,
                 model_config
                 ):
        # save hyperparameters
        self.save_hyperparameters(ignore=["model", "data_module", "processor", "trainer"]) # save training_args

        # Save
        self._model = model
        self._data_module = data_module
        self._processor = processor
        self._trainer = trainer

        # set later
        self._dataset = None
        self._train_dataset = None
        self._test_dataset = None

    def get_trainer(self):
        pass

    def get_training_args(self):
        pass

    def get_model_config(self):
        pass

    def set_model_config(self, model_config):
        pass

    def get_processor(self):
        pass

    def tokenize_features(self):
        pass

    def get_dataset(self):
        pass

    def set_dataset(self, dataset):
        pass

    def get_train_dataset(self):
        pass

    def set_train_dataset(self, train_dataset):
        pass

    def get_test_dataset(self):
        pass

    def set_test_dataset(self, test_dataset):
        pass

    def get_model(self):
        pass

    def ingest_data(self, datasource):
        """
        fetching a dataset from huggingface and initializing the _dataset attributes

        :param datasource: huggingface uri
        """

        hf_owner = datasource.split("/")[0]
        hf_dataset_name = datasource.split("/")[1]

        dataset = self._data_module.ingest_data(hf_owner, hf_dataset_name)
        self.set_dataset(dataset)

        logger.info(f"Dataset attribute initialized with huggingface Dataset: {hf_owner}/{hf_dataset_name}")

    def preprocess_dataset(self):
        """
        Preprocess the dataset so that the images are changed as needed for training.
        Then the methods reassigns the _dataset attributes.
        """

        dataset = self._data_module.preprocess_dataset(self.get_dataset())

        self.set_dataset(dataset)

        logger.info(f"Dataset was preprocessed")

    def encode_labels(self, split, feature_column, target_column):
        """
        Encode the labels so that the labels are correctly formatted for training.
        This reassigns the _dataset attribute.

        """
        encoded_dataset = self._data_module.encode_dataset(self.get_dataset(), split, feature_column, target_column)

        self.set_dataset(encoded_dataset)

        logger.info(f"Dataset was encoded")

    def split_train_test(self, split, feature_column, target_column):
        """
        Split the dataset into train and test sets.

        """
        try:
            X_train, X_test, y_train, y_test = self._data_module.split_data(self.get_dataset(), split, feature_column, target_column)

            train_dataset = self.convert_feature_label_lists_into_dict(X_train, y_train, feature_column, target_column)
            test_dataset = self.convert_feature_label_lists_into_dict(X_test, y_test, feature_column, target_column)

            self.set_train_dataset(train_dataset)
            self.set_test_dataset(test_dataset)
        except Exception as e:
            logger.error(f"Dataset was not found, cannot split into train and test sets. Error: {e}")

    def convert_feature_label_lists_into_dict(self, feature_list: list, label_list: list, feature_name: str, label_name: str):
        """
        Convert the feature list and the corresponding label list into a dictionary.

        :param feature_list: list of pixel values per image
        :param label_list: list of tokens for each label

        """
        res_dataset = {}

        for feature, label in zip(feature_list, label_list):
            res_dataset[feature_name] = feature
            res_dataset[label_name] = label

        return res_dataset

    def train(self, run_name: str, experiment_name:str, model_flavor="pytorch", tags=None):
        logger.info(f"MLflow uri is: {mlflow.get_tracking_uri()}")

        logger.info(f"Training model with flavor {model_flavor}")

        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow experiment \"Experiment\" {experiment_name} was created")

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            if self.get_training_args() is not None:
                for arg_name, arg_value in vars(self.get_training_args()).items():
                    try:
                        mlflow.log_params(arg_name, arg_value)

                        logger.info(f"Logged parameter {arg_name} was set to {arg_value}")
                    except:
                        continue

            # Load model
            model = self.get_model()

            logger.info(f"Model was loaded")

            # Train model
            logger.info(f"Training starting")
            self.get_model().train()

            logger.info(f"Model was trained")

            # Log model
            model_metadata = mlflow.pytorch.log_model(model, artifact_path="model")

            logger.info(f"Model was logged with metadata: {model_metadata}")

            # Set tags
            if tags is not None:
                for index, tag in enumerate(tags):
                    mlflow.log_metric(tag, tags[index])


    def save_model(self):
        pass