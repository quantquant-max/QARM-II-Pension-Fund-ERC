import logging

def get_logger(name: str = "core"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger
