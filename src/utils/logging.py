import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os

class LoggerUtility:
    """
    A utility class for creating and managing loggers.
    """
    @staticmethod
    def setup_logger(
        name: str,
        log_directory: str = "./files/logs",
        file_date= datetime.now().strftime('%Y-%m-%d %H-%M'),
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 3,
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
    ) -> logging.Logger:
        """
        Sets up a logger with a rotating file handler and console handler.

        Args:
            name (str): The name of the logger.
            log_directory (str): The directory to save log files.
            level (int): The logging level.
            max_bytes (int): Maximum size of a log file before rotation.
            backup_count (int): Number of backup files to keep.
            log_format (str): The log message format.

        Returns:
            logging.Logger: Configured logger instance.
        """
        # Ensure log directory exists
        name=name.split("/")[-1].split(".")[0]#get the file name only
        log_directory+=f"/{name}"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        # Create a logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Define log format
        formatter = logging.Formatter(log_format)

        # Create a log file with the current date in its name
        log_file = os.path.join(log_directory, f"{file_date}.log")

        # Create a rotating file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

# Example Usage
if __name__ == "__main__":
    # Set up logger
    logger = LoggerUtility.setup_logger(
        name="MyAppLogger",
        log_directory="./files/logs",
        level=logging.DEBUG
    )

    # Log some messages
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
