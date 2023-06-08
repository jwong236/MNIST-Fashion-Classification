from utils.abstract_base_model import AbstractBaseModel
from sklearn.metrics import accuracy_score
import numpy as np

class KNNModel(AbstractBaseModel):
    def __init__(self, config, training_data, validation_data, logger):
        super().__init__(config, training_data, validation_data, logger)

        self.k = int()
        self.k_max = 21
        self.labels = None

        self.predict_counter = int()
        self.total_points = len(self.validation_data[0])

    def initialize_model(self):
        """
        Initialize model parameters and hyperparameters
        """
        self.k = self.config.getint('KNN', 'n_neighbors')
        self.logger.info(f"Model parameters initialized. K set to: {self.k} and K_max set to: {self.k_max}")

    @staticmethod
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def train_and_validate_model(self):
        """
        Train model to update parameters with an optimization algorithm and minimize loss function
        """
        self.labels = self.training_data[1]
        self.logger.info("Starting model training and validation.")

        best_k = None
        best_accuracy = -1
        self.logger.info(f"Initial best_k: {best_k}, best_accuracy: {best_accuracy}")

        for k in range(1, self.k_max):
            self.k = k
            self.logger.info(f"Training with k set to: {self.k}")
            self.predict_counter = 0
            predictions = [self.predict(point) for point in self.validation_data[0]]
            accuracy = accuracy_score(self.validation_data[1], predictions)
            self.logger.info(f"Validation accuracy for k={k}: {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                self.logger.info(f"New best k found: {best_k} with accuracy: {best_accuracy}")

        self.k = best_k
        self.logger.info(f"Training completed. Best k: {self.k} with best accuracy: {best_accuracy}")

    def predict(self, point):
        """
        Predict the label of a single point, using the trained model.
        """
        self.predict_counter += 1
        self.logger.info(f"[k = {self.k}] -> point {self.predict_counter}/{self.total_points}: Predicting...")
        distances = np.array([self.euclidean_distance(point, train_point) for train_point in self.training_data[0]])
        nearest_indices = distances.argsort()[:self.k]
        nearest_labels = self.labels[nearest_indices]
        counts = np.bincount(nearest_labels)
        prediction = np.argmax(counts)
        self.logger.info(f"[k = {self.k}] -> point {self.predict_counter}/{self.total_points}: Predicted label: {prediction}")
        return prediction

    def test_model(self, test_data):
        """
        Test model with test_data, returning classification results
        """
        self.logger.info("Testing model on test data.")
        predictions = np.array([self.predict(point) for point in test_data[0]])
        self.logger.info("Model testing completed.")
        return {"features": test_data[0], "labels": predictions}

    def evaluate_model(self, test_data_results):
        """
        Return confusion matrix of results of test_model
        """
        self.logger.info("Evaluating model.")
        # Implement confusion matrix calculation here, depending on how you want it.

    def print_results(self):
        """
        Print confusion matrix
        """
        self.logger.info("Printing model results.")
        # Implement print functionality here

    def report_results(self):
        """
        Write confusion matrix to report.txt
        """
        self.logger.info("Reporting model results to report.txt.")
        # Implement report writing functionality here
