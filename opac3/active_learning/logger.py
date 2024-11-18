# active_learning/logger.py

import logging

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent adding multiple handlers to the logger
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
