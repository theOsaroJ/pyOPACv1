# opac3/utils/logger.py

import logging

def get_logger(name, level=logging.INFO):
    """
    Configure and return a logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers are already added
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
