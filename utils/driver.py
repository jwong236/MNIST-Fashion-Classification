import configparser

from keras.datasets import fashion_mnist

from utils.logger import Logger
from utils.preprocessor import Preprocessor

from models.knn.knn_model import KNNModel
from models.logistic_regression.logistic_regression_model import LogisticRegressionModel
from models.neural_network.neural_network_model import NeuralNetworkModel

class Driver:
    def __init__(self):
        # Read configs
        self.config = configparser.ConfigParser()
        self.config.read('configs.ini')

        # Get file path
        self.report_path = self.config.get('DEFAULT', 'report_path')
        
        # Initialize data
        self.training_data = (None, None)
        self.validation_data = (None, None)
        self.testing_data = (None, None)

        self.logger = Logger.get_logger('driver.log')

    def load_preprocess_split_data(self):
        # Load Data: Load MNIST data
        self.logger.info(f"Loading Data...\n")
        print(f"Loading Data...\n")
        (original_X_training_data, original_y_training_data), (original_X_testing_data, original_y_testing_data) = fashion_mnist.load_data()

        # Preprocess Data
        # Instantiate preprocessor
        preprocessor = Preprocessor(self.config, Logger.get_logger('preprocessor.log'))

        # Preprocess training and validation data
        self.logger.info(f"Preprocessing training and validation data...")
        print(f"Preprocessing training and validation data...")
        training_and_validation_data = preprocessor.preprocess(original_X_training_data, original_y_training_data)

        # Preprocess testing data
        self.logger.info(f"Preprocessing testing data...")
        print(f"Preprocessing testing data...")
        self.testing_data = preprocessor.preprocess(original_X_testing_data, original_y_testing_data)

        # Split training data into training and validation data
        self.logger.info(f"Splitting training data...")
        print(f"Splitting training data...")
        self.training_data, self.validation_data = preprocessor.split_data(training_and_validation_data[0], training_and_validation_data[1])


    def create_knn_model(self, training_data, validation_data):
        try:
            # Instantiate
            self.logger.info(f"Instantiating KNNModel...")
            print(f"Instantiating KNNModel...")
            knn_model = KNNModel(self.config, training_data, validation_data, Logger.get_logger('knn.log'))

            # Train and Validate
            self.logger.info(f"Training and Validating KNNModel...")
            print(f"Training and Validating KNNModel...")
            knn_model.train_and_validate_model()

            # Test
            self.logger.info(f"Testing KNNModel...")
            print(f"Testing KNNModel...")
            test_results = knn_model.test_model(self.testing_data)

            # Evaluate
            self.logger.info(f"Evaluating KNNModel...")
            print(f"Evaluating KNNModel...")
            confusion_matrix = knn_model.evaluate_model(test_results)

            # Print and report
            knn_model.print_results(confusion_matrix)
            knn_model.report_results(confusion_matrix, self.report_file_path)
            
        except Exception as e:
            self.logger.warning(f"KNNModel not available yet. Error: {str(e)}")
            print(f"KNNModel not available yet. Error: {str(e)}")

    def create_logistic_regression_model(self, training_data, validation_data):
        try:
            # Instantiate
            self.logger.info(f"Instantiating LogisticRegressionModel...")
            print(f"Instantiating LogisticRegressionModel...")
            lr_model = LogisticRegressionModel(self.config, training_data, validation_data, Logger.get_logger('logistic_regression.log'))

            # Train and Validate
            self.logger.info(f"Training and Validating LogisticRegressionModel...")
            print(f"Training and Validating LogisticRegressionModel...")
            lr_model.train_and_validate_model()

            # Test
            self.logger.info(f"Testing LogisticRegressionModel...")
            print(f"Testing LogisticRegressionModel...")
            test_results = lr_model.test_model(self.testing_data)

            # Evaluate
            self.logger.info(f"Evaluating LogisticRegressionModel...")
            print(f"Evaluating LogisticRegressionModel...")
            confusion_matrix = lr_model.evaluate_model(test_results)

            # Print and report
            lr_model.print_results(confusion_matrix)
            lr_model.report_results(confusion_matrix, self.report_file_path)
            
        except Exception as e:
            self.logger.warning(f"LogisticRegressionModel not available yet. Error: {str(e)}")
            print(f"LogisticRegressionModel not available yet. Error: {str(e)}")

    def create_neural_network_model(self, training_data, validation_data):
        try:
            # Instantiate
            self.logger.info(f"Instantiating NeuralNetworkModel...")
            print(f"Instantiating NeuralNetworkModel...")
            nn_model = NeuralNetworkModel(self.config, training_data, validation_data, Logger.get_logger('neural_network.log'))

            # Train and Validate
            self.logger.info(f"Training and Validating NeuralNetworkModel...")
            print(f"Training and Validating NeuralNetworkModel...")
            nn_model.train_and_validate_model()

            # Test
            self.logger.info(f"Testing NeuralNetworkModel...")
            print(f"Testing NeuralNetworkModel...")
            test_results = nn_model.test_model(self.testing_data)

            # Evaluate
            self.logger.info(f"Evaluating NeuralNetworkModel...")
            print(f"Evaluating NeuralNetworkModel...")
            confusion_matrix = nn_model.evaluate_model(test_results)

            # Print and report
            nn_model.print_results(confusion_matrix)
            nn_model.report_results(confusion_matrix, self.report_file_path)
            
        except Exception as e:
            self.logger.warning(f"NeuralNetworkModel not available yet. Error: {str(e)}")
            print(f"NeuralNetworkModel not available yet. Error: {str(e)}")

    def run(self):
        self.logger.info(f"Running Driver class")
        print(f"Running Driver class")
        self.load_preprocess_split_data()
        self.create_knn_model(self.training_data, self.validation_data)
        self.create_logistic_regression_model(self.training_data, self.validation_data)
        self.create_neural_network_model(self.training_data, self.validation_data)
