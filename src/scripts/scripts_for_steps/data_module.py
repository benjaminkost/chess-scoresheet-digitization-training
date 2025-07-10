import logging
from abc import ABC, abstractmethod
from datasets import Dataset

from .data_loader_strategy import DataLoaderStrategy
from .data_splitter_strategy import DataSplittingStrategy
from .ingest_data_strategy import DataIngestorStrategy
from .preprocessing_strategy import PreprocessingStrategy

# ANSI Escape Code for white letters
WHITE = "\033[37m"
RESET = "\033[0m"  # Zum Zurücksetzen der Farbe

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

# Define an abstract class for Data Ingestor
class DataModule(DataIngestorStrategy, PreprocessingStrategy, DataSplittingStrategy, DataLoaderStrategy, ABC):
    @abstractmethod
    def set_ingest_data_strategy(self, ingest_data_strategy: DataIngestorStrategy):
        pass

    @abstractmethod
    def set_preprocessing_strategy(self, preprocessing_strategy: PreprocessingStrategy):
        pass

    @abstractmethod
    def set_data_splitter_strategy(self, data_splitter_strategy: DataSplittingStrategy):
        pass

    @abstractmethod
    def set_dataloader_strategy(self, dataloader_strategy: DataLoaderStrategy):
        """Abstract method to get dataloader"""
        pass

# Implement a concrete class for Image Data Ingestion
class HuggingFaceImageDataModule(DataModule):

    def __init__(self, ingest_data_strategy: DataIngestorStrategy,
                 preprocessing_strategy: PreprocessingStrategy,
                 data_splitter_strategy: DataSplittingStrategy,
                 dataloader_strategy: DataLoaderStrategy):
        logger.info("Initializing strategies for DataModule.")

        self._ingest_data_strategy = ingest_data_strategy
        self._preprocessing_strategy = preprocessing_strategy
        self._data_splitter_strategy = data_splitter_strategy
        self._dataloader_strategy = dataloader_strategy

    # Setter
    def set_ingest_data_strategy(self, ingest_data_strategy: DataIngestorStrategy):
        logger.info("Switching ingesting data strategy.")
        self._ingest_data_strategy = ingest_data_strategy

    def set_preprocessing_strategy(self, preprocessing_strategy: PreprocessingStrategy):
        logger.info("Switching preprocessing strategy.")
        self._preprocessing_strategy = preprocessing_strategy

    def set_data_splitter_strategy(self, data_splitter_strategy: DataSplittingStrategy):
        logger.info("Switching data splitter strategy.")
        self._data_splitter_strategy = data_splitter_strategy

    def set_dataloader_strategy(self, dataload_strategy: DataLoaderStrategy):
        logger.info("Switching data loader strategy.")
        self._dataloader_strategy = dataload_strategy

    # Executer
    def ingest_data(self, owner: str, dataset_name: str) -> Dataset:
        if not self._ingest_data_strategy:
            raise ValueError("Ingest data strategy is not set")
        return self._ingest_data_strategy.ingest_data(owner, dataset_name)

    def preprocess_dataset(self, dataset):
        if not self._preprocessing_strategy:
            raise ValueError("Preprocessing strategy is not set")
        return self._preprocessing_strategy.preprocess_dataset(dataset)

    def split_data(self, dataset: Dataset, split: str, feature_column: str, target_column: str):
        if not self._data_splitter_strategy:
            raise ValueError("Data splitter strategy is not set")
        return self._data_splitter_strategy.split_data(dataset, split, feature_column, target_column)

    def load_batch(self, dataset: Dataset, batch_size=None, batch_start_index=None, shuffle=False):
        if not self._dataloader_strategy:
            raise ValueError("Dataloader strategy is not set")
        return self._dataloader_strategy.load_batch(dataset, batch_size, batch_start_index, shuffle)

    def reset_batch_start(self):
        if not self._dataloader_strategy:
            raise ValueError("Dataloader strategy is not set")
        self._dataloader_strategy.reset_batch_start()
