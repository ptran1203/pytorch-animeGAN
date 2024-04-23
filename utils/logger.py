import logging


def get_logger(path, *args, **kwargs):
    # logger = logging.getLogger('train')
    # logger.setLevel(logging.NOTSET)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # # add filehandler
    # fh = logging.FileHandler(path)
    # fh.setLevel(logging.NOTSET)
    # fh.setFormatter(formatter)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.ERROR)
    # logger.addHandler(fh)
    # logger.addHandler(ch)
    # return logger
    logging.basicConfig(format = '%(asctime)s %(message)s',
                    datefmt = '%m/%d/%Y %I:%M:%S %p',
                    handlers=[
                        logging.FileHandler(path),
                        logging.StreamHandler()
                    ],
                    level=logging.DEBUG)
    return logging