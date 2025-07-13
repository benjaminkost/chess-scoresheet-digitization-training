import torch
from torch import nn
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from evaluation_graphs import ProgressBoard
from hyperparameter_util import ModelHyperparameter

# Base classes
class Module(nn.Module, ModelHyperparameter):
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, value.to(self.cpu()).detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError

    def cpu(self):
        """Get the CPU device."""
        return torch.device('cpu')

class Classifier(Module):
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions."""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare

class TransformerModelWrapper(Classifier):
    """The base class for transformer models"""
    def __init__(self, hf_processor_owner, hf_processor_name, hf_model_owner, hf_model_name):
        super().__init__()
        self.save_hyperparameters()
        # initialize processor
        self.processor = self.load_processor(hf_processor_owner, hf_processor_name)
        # initialize model
        self.model = self.load_transformer_model(hf_model_owner, hf_model_name)

    def load_processor(self, hf_processor_owner, hf_processor_name):
        return NotImplementedError

    def load_transformer_model(self, hf_model_owner, hf_model_name):
        return NotImplementedError


# Implementations for model classes
class TrOCRWrapper(TransformerModelWrapper):
    def load_processor(self, hf_processor_owner, hf_processor_name):
        hf_processor_uri = f"{hf_processor_owner}/{hf_processor_name}"
        return TrOCRProcessor.from_pretrained(hf_processor_uri)

    def load_transformer_model(self, hf_model_owner, hf_model_name):
        hf_model_uri = f"{hf_model_owner}/{hf_model_name}"
        return VisionEncoderDecoderModel.from_pretrained(hf_model_uri)

    def forward(self, X):
        # Generate pixel_values with processor
        pixel_values = self.processor(X, return_tensors="pt").pixel_values

        # Generate ids from model
        generated_ids = self.model.generate(pixel_values)

        # Decode the result from the model
        prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return prediction




