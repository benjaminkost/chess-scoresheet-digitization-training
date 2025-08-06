import logging
from abc import ABC, abstractmethod
from src.scripts.scripts_for_steps.data_module import DataModule

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

class Trainer(ABC):
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
    def get_features_train(self):
        pass

    @abstractmethod
    def set_features_train(self, features_train):
        pass

    @abstractmethod
    def get_features_test(self):
        pass

    @abstractmethod
    def set_features_test(self, features_test):
        pass

    @abstractmethod
    def get_labels_train(self):
        pass

    @abstractmethod
    def set_labels_train(self, labels_train):
        pass

    @abstractmethod
    def get_labels_test(self):
        pass

    @abstractmethod
    def set_labels_test(self, labels_test):
        pass

    # business logic
    @abstractmethod
    def ingest_data(self, datasource):
        pass

    @abstractmethod
    def preprocess_dataset(self):
        pass

    @abstractmethod
    def encode_labels(self):
        pass

    @abstractmethod
    def split_train_test(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

class TransformerTrainer(Trainer, ABC):

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
                 data_module: DataModule,
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
        self._features_train = None
        self._features_test = None
        self._labels_train = None
        self._labels_test = None

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

    def get_features_train(self):
        pass

    def set_features_train(self, features_train):
        pass

    def get_features_test(self):
        pass

    def set_features_test(self, features_test):
        pass

    def get_labels_train(self):
        pass

    def set_labels_train(self, labels_train):
        pass

    def get_labels_test(self):
        pass

    def set_labels_test(self, labels_test):
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

    def encode_labels(self):
        """
        Encode the labels so that the labels are correctly formatted for training.


        """

    def split_train_test(self):
        pass

    def train(self):
        pass

    def save_model(self):
        pass