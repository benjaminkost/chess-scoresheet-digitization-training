import inspect
from abc import abstractmethod, ABC


# interface
class Hyperparameters(ABC):
    @abstractmethod
    def save_hyperparameters(self):
        """
        Saves all parameters passed into a class as attributes of that class
        """
        pass

# Implementation
class ModelHyperparameters(Hyperparameters):
    def save_hyperparameters(self, ignore=[]):
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items(): setattr(self, k, v)
