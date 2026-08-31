# coding: utf-8
import logging
import logging.handlers

def setup_logging(logfile="yn40mss.log", log_level=logging.INFO):
    """ Setup logging configuration """

    # Console formatter, mention name
    cfmt = logging.Formatter(('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'))

    # File formatter, mention time
    ffmt = logging.Formatter(('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(cfmt)

    # File handler
    fh = logging.handlers.RotatingFileHandler(logfile, maxBytes=100*1024*1024, backupCount=100)
    fh.setLevel(log_level)
    fh.setFormatter(ffmt)

    logging.basicConfig(
        level = log_level,
        handlers=[ch, fh],
    )