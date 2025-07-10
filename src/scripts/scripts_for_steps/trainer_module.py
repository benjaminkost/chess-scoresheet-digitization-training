from datasets import Dataset
from data_module import DataModule
from hyperparameter_util import ModelHyperparameter

class Trainer(ModelHyperparameter):
    """The base class for training models with data"""
    def __init__(self,
                 max_epochs,
                 data_module: DataModule,
                 trainer=None,
                 num_gpus=0,
                 gradient_clip_val=0
                 ):
        self.model = None
        self.train_test_data = None
        self.save_hyperparameters()
        self.trainer = trainer
        assert num_gpus == 0, 'No GPU support yet'
        self.data_module = data_module

    def prepare_data_from_hf(self, hf_dataset_owner, hf_dataset_name, split: str, feature_column: str, target_column: str):
        data = self.data_module.ingest_data(hf_dataset_owner, hf_dataset_name)

        self.prepare_data(data, split, feature_column, target_column)

    def prepare_data(self, dataset: Dataset, split: str, feature_column: str, target_column: str):
        self.train_test_data = self.data_module.split_data(dataset, dataset, split, feature_column, target_column)

    def prepare_model(self, model):
        self.model = model

    def fit(self, model, data):
        pass


        