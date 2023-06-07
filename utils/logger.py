import logging
import os

class Logger:

    @staticmethod
    def get_logger(log_file):
        log_directory = 'logs'
        log_path = os.path.join(log_directory, log_file)
        
        logger = logging.getLogger(log_file)
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            # Prevent logging from propagating to the root logger
            logger.propagate = 0

            # Create logs directory if doesn't exist
            if not os.path.exists(log_directory):
                os.makedirs(log_directory)

            # Create file handler
            file_handler = logging.FileHandler(log_path, mode = 'w')
            file_handler.setLevel(logging.DEBUG)

            # Create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            # Add the handlers to the logger
            logger.addHandler(file_handler)
        
        return logger
