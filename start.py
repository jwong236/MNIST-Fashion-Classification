from utils.driver import Driver

def main():
    # Create an instance of the Driver class
    driver = Driver()
    
    # Uncomment this when all classifiers are implemented
    # driver.run()



    # Delete these lines of code once all classifiers are implemented
    driver.load_preprocess_split_data()
        
    # Uncomment below lines when you're ready to implement the corresponding classifier
    # driver.create_knn_model(driver.training_data, driver.validation_data)
    # driver.create_logistic_regression_model(driver.training_data, driver.validation_data)
    # driver.create_neural_network_model(driver.training_data, driver.validation_data)

if __name__ == '__main__':
    main()
