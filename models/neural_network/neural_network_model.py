from utils.abstract_base_model import AbstractBaseModel

class NeuralNetworkModel(AbstractBaseModel):
    def __init__(self, config, training_data, validation_data, logger):
        super().__init__(config, training_data, validation_data, logger)
        
    def initialize_model(self):
        pass

    def train_and_validate_model(self):
        pass

    def test_model(self, test_data):
        pass

    def evaluate_model(self, test_data_results):
        pass

    def print_results(self):
        pass

    def report_results(self):
        pass
