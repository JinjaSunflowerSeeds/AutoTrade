import logging
import colorlog

class MyLogger:
    def __init__(self,under_name=None, testing_mode=None) -> logging.Logger:
        log_format = (
            '[%(asctime)s]'
            '[%(funcName)s]:'
            '%(message)s'
        )
        bold_seq = '\033[1m'
        colorlog_format = (
            f'{bold_seq} '
            '%(log_color)s '
            f'{log_format}'
        )
        colorlog.basicConfig(format=colorlog_format)
        logger = logging.getLogger(under_name)

        if testing_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Output full log
        fh = logging.FileHandler('app.log')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Output warning log
        fh = logging.FileHandler('app.warning.log')
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Output error log
        fh = logging.FileHandler('app.error.log')
        fh.setLevel(logging.ERROR)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        self.log = logger
