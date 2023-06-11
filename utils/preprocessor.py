from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Preprocessor:
    def __init__(self, config, logger):
        # Get preprocessing configs
        self.seed = config.getint('PREPROCESSING', 'seed')

        self.logger = logger

        self.training_split = config.getfloat('PREPROCESSING', 'training_split')
        self.validation_split = config.getfloat('PREPROCESSING', 'validation_split')

    def preprocess(self, X, y):
        """
        This function will preprocess the data.
        """
        self.logger.info(f"Reshaping data...")
        print(f"Reshaping data...")
        # Check original shape of data
        original_shape = X.shape[1:]
        if original_shape != (28, 28):
            # Reshape data
            self.logger.info(f"Data before reshaping: {X}")
            X = X.reshape(-1, 28, 28)
            self.logger.info(f"Data after reshaping: {X}")
        else:
            self.logger.info(f"Data already in correct shape, no reshaping necessary")

        # Normalize data
        self.logger.info(f"Normalizing data...\n")
        print(f"Normalizing data...\n")
        self.logger.info(f"First datapoint before normalizing (sample): \n(Note: only prints center 18x18)\n{X[0, 5:23, 5:23]}")
        self.logger.info(f"Last datapoint before normalizing (sample): \n(Note: only prints center 18x18)\n{X[-1, 5:23, 5:23]}\n")
        X = X / 255.0
        self.logger.info(f"First datapoint after normalizing (sample): \n(Note: only prints center 18x18)\n{X[0, 5:23, 5:23]}")
        self.logger.info(f"Last datapoint after normalizing (sample): \n(Note: only prints center 18x18)\n{X[-1, 5:23, 5:23]}\n")
        return X, y

    def split_data(self, X_training_data, y_training_data):
        """
        This function will split the data into training and validation sets.
        """
        # Split data using sklearn
        train_X, val_X, train_y, val_y = train_test_split(X_training_data, y_training_data,
                                                      train_size=self.training_split,
                                                      test_size=self.validation_split,
                                                      random_state=self.seed)
        self.logger.info(f"The training data has been split, with {self.training_split * 100}% allocated for training and {self.validation_split * 100}% allocated for validation.\n")
        print(f"The training data has been split, with {self.training_split * 100}% allocated for training and {self.validation_split * 100}% allocated for validation.\n")
        return (train_X, train_y), (val_X, val_y)
    
    def visualize_data(self, X, y):
        """
        This function will visualize the first 9 datapoints. Function is taken from Professor K's homework 2.
        """
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

        figure, axes = plt.subplots(3, 3, figsize=(6, 6))
        for i in range(9):
            row = i // 3
            col = i % 3
            axes[row, col].imshow(X[i].reshape(28,28))
            axes[row, col].set_title("Label: " + str(y[i]))
        plt.tight_layout()
        plt.savefig("data_visualization.png")

