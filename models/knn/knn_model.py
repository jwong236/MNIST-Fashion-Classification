from utils.abstract_base_model import AbstractBaseModel
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

class KNNModel(AbstractBaseModel):
    def __init__(self, config, training_data, validation_data, logger):
        super().__init__(config, training_data, validation_data, logger)

        self.k = int()
        self.k_max = 23
        self.labels = None

        self.predict_counter = int()
        self.total_points = len(self.validation_data[0])
        self.k_metrics = {}

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
        for k in range(1, self.k_max + 1, 2):
            self.k = k
            self.predict_counter = 0

            self.logger.info(f"Training with k set to: {self.k}")
            validation_predictions = []
            for i, point in enumerate(self.validation_data[0]):
                prediction = self.predict(point)
                validation_predictions.append(prediction)
                self.logger.info(f"[k = {self.k}] -> point {i+1}/{self.total_points}: Predicted label: {prediction} for validation data")
            validation_accuracy = accuracy_score(self.validation_data[1], validation_predictions)
            validation_error = 1 - validation_accuracy
                
            self.k_metrics[k] = {"validation_error": validation_error}
            self.logger.info(f"Validation error for k={k}: {validation_error}")
        self.k = min(self.k_metrics, key=lambda k: self.k_metrics[k]['validation_error'])
        self.logger.info(f"Training and validation phase completed. Best k found: {self.k} with validation error: {self.k_metrics[self.k]['validation_error']}")
        print(f"Training and validation phase completed. Best k found: {self.k} with validation error: {self.k_metrics[self.k]['validation_error']}")

    def predict(self, point):
        """
        Predict the label of a single point, using the trained model.
        """
        distances = np.array([self.euclidean_distance(point, train_point) for train_point in self.training_data[0]])
        nearest_indices = distances.argsort()[:self.k]
        nearest_labels = self.labels[nearest_indices]
        counts = np.bincount(nearest_labels)
        prediction = np.argmax(counts)
        return prediction


    def test_model(self, test_data):
        """
        Test model with test_data, returning classification results
        """
        self.predict_counter = 0
        total_test_points = len(test_data[0])
        self.logger.info(f"Starting testing phase...")
        print(f"Starting testing phase...")
        predictions = []
        for i, point in enumerate(test_data[0]):
            prediction = self.predict(point)
            predictions.append(prediction)
            self.logger.info(f"[k = {self.k}] -> point {i+1}/{total_test_points}: Predicted label: {prediction} for test data")
        test_accuracy = accuracy_score(test_data[1], predictions)
        test_error = 1 - test_accuracy

        self.k_metrics[self.k]["test_error"] = test_error
        self.logger.info("Testing phase completed.")
        print("Testing phase completed.")
        return {"features": test_data[0], "label_predictions": predictions, "true_labels": test_data[1]}


    def evaluate_model(self, test_data_results):
        """
        Return figures of error rate and accuracy for each value of k, and the confusion matrix of results of test_model
        """
        self.logger.info(f"Evaluating model.")
        print(f"Evaluating model.")

        # Confusion matrix
        true_labels = test_data_results['true_labels']
        predicted_labels = test_data_results['label_predictions']
        cm = confusion_matrix(true_labels, predicted_labels)

        # Error figure
        k_values = list(self.k_metrics.keys())
        training_errors = [metrics["validation_error"] for metrics in self.k_metrics.values()]
        testing_errors = [metrics.get("test_error", 0) for metrics in self.k_metrics.values()]

        plt.figure(figsize=(10, 5))
        plt.plot(k_values, training_errors, label='Training Error')
        plt.plot(k_values, testing_errors, label='Testing Error')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        plt.title('Error Rate Change with Different K Values')
        error_figure = plt.gcf()

        # Accuracy figure
        training_accuracies = [1 - error for error in training_errors]
        testing_accuracies = [1 - error for error in testing_errors]
        
        plt.figure(figsize=(10, 5))
        plt.plot(k_values, training_accuracies, label='Training Accuracy')
        plt.plot(k_values, testing_accuracies, label='Testing Accuracy')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.title('Accuracy Change with Different K Values')
        accuracy_figure = plt.gcf()
        return cm, error_figure, accuracy_figure


    def print_results(self, evaluation_results):
        """
        Print confusion matrix, error rates, and accuracy rates for each k value
        """
        class_labels = [str(i) for i in range(10)]

        self.logger.info(f"Printing model results.")
        print(f"Printing model results.")

        cm, error_figure, accuracy_figure = evaluation_results


        self.logger.info("Confusion Matrix:")
        print("Confusion Matrix:")
        cm_header = "\t" + "\t".join([f"Predicted {label}" for label in class_labels])
        cm_rows = []
        for i, row in enumerate(cm):
            cm_rows.append(f"Actual {class_labels[i]}\t" + "\t".join([str(cell) for cell in row]))
        cm_str = cm_header + "\n" + "\n".join(cm_rows)
        self.logger.info(f"\n{cm_str}\n")
        print(f"\n{cm_str}\n")


        for k, metrics in self.k_metrics.items():
            self.logger.info(f"Results for k={k}:")
            print(f"Results for k={k}:")
            self.logger.info(f"Validation Error: {metrics['validation_error']}")
            print(f"Validation Error: {metrics['validation_error']}")
            if 'test_error' in metrics:
                self.logger.info(f"Testing Error: {metrics['test_error']}")
                print(f"Testing Error: {metrics['test_error']}")
            self.logger.info(f"Validation Accuracy: {1 - metrics['validation_error']}")
            print(f"Validation Accuracy: {1 - metrics['validation_error']}")
            if 'test_error' in metrics:
                self.logger.info(f"Testing Accuracy: {1 - metrics['test_error']}")
                print(f"Testing Accuracy: {1 - metrics['test_error']}")
            self.logger.info("-----")
            print("-----")
        
        #error_figure.show()
        #accuracy_figure.show()

    def report_results(self, evaluation_results):
        self.logger.info("Reporting model results.")
        print("Reporting model results.")

        cm, error_figure, accuracy_figure = evaluation_results

        error_figure.savefig('error_figure.png')
        accuracy_figure.savefig('accuracy_figure.png')

        class_labels = [str(i) for i in range(10)]
        cm_header = "\t" + "\t".join([f"Predicted {label}" for label in class_labels])
        cm_rows = []
        for i, row in enumerate(cm):
            cm_rows.append(f"Actual {class_labels[i]}\t" + "\t".join([str(cell) for cell in row]))
        cm_str = cm_header + "\n" + "\n".join(cm_rows)

        with open("reports.txt", 'w') as f:
            f.write("Confusion Matrix:\n")
            f.write(cm_str)
            f.write("\n\n")

            for k, metrics in self.k_metrics.items():
                f.write(f"Results for k={k}:\n")
                f.write(f"Validation Error: {metrics['validation_error']}\n")
                if 'test_error' in metrics:
                    f.write(f"Testing Error: {metrics['test_error']}\n")
                f.write(f"Validation Accuracy: {1 - metrics['validation_error']}\n")
                if 'test_error' in metrics:
                    f.write(f"Testing Accuracy: {1 - metrics['test_error']}\n")
                f.write("-----\n")

        self.logger.info("Added to reports")
        print("Added to reports")

