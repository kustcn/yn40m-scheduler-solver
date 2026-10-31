import logging
import logging.handlers


def log_level_str_to_int(log_level_str):
    if log_level_str == "DEBUG":
        return logging.DEBUG
    elif log_level_str == "INFO":
        return logging.INFO
    elif log_level_str == "WARNING":
        return logging.WARNING
    elif log_level_str == "ERROR":
        return logging.ERROR
    elif log_level_str == "CRITICAL":
        return logging.CRITICAL
    else:
        return logging.INFO

def setup_logging(logfile="yn40mss.log", log_level=logging.INFO):
    """ Setup logging configuration """

    # Console formatter, mention name
    cfmt = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s')

    # File formatter, mention time
    ffmt = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s')

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
