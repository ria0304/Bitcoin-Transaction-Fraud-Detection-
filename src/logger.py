"""Central logger: console + file."""
import logging, os

def get_logger(name="elligat", log_file=None, level="INFO"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
