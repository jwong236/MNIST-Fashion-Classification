from typing import Tuple

class Preprocessor:
    def __init__(self, config):
        # Get preprocessing configs
        self.seed = config.getint('PREPROCESSING', 'seed')
        self.training_split = config.getfloat('PREPROCESSING', 'training_split')
        self.validation_split = config.getfloat('PREPROCESSING', 'validation_split')

    def preprocess(self, X_data, y_data) -> Tuple:
        """
        This function will preprocess the data.
        """
        return X_data, y_data

    def split_data(self, data, training_split: float, validation_split: float) -> Tuple:
        """
        This function will split the data into training and validation sets.
        """
        return data, data
