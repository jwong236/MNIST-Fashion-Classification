from abc import ABC, abstractmethod

class AbstractBaseModel(ABC):
    def __init__(self, config, training_data, validation_data, logger):
        self.config = config
        self.training_data = training_data
        self.validation_data = validation_data
        self.logger = logger

    @abstractmethod
    def initialize_model(self):
        """
        Initialize model parameters and hyperparameters
        """
        pass

    @abstractmethod
    def train_and_validate_model(self):
        """
        Train model to update parameters with an optimization algorithm and minimize loss function
        """
        pass

    @abstractmethod
    def test_model(self, test_data):
        """
        Test model with test_data, returning classification results
        """
        pass

    @abstractmethod
    def evaluate_model(self, test_data_results):
        """
        Return confusion matrix of results of test_model
        """
        pass

    @abstractmethod
    def print_results(self):
        """
        Print confusion matrix
        """
        pass

    @abstractmethod
    def report_results(self):
        """
        Write confusion matrix to report.txt
        """
        pass
