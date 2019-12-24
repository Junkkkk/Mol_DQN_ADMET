import logging
import logging.handlers
from keras.callbacks import TensorBoard


def get_logger(logger_name, logger_path):
    logger = logging.getLogger(logger_name)

    formatter = logging.Formatter('[%(filename)s|%(asctime)s] %(message)s')

    # 스트림과 파일로 로그를 출력하는 핸들러를 각각 만든다.
    file_handler = logging.FileHandler(logger_path)
    stream_handler = logging.StreamHandler()

    # 각 핸들러에 포매터를 지정한다.
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 로거 인스턴스에 스트림 핸들러와 파일핸들러를 붙인다.
    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)
    return logger

def get_tb(path):
    return TensorBoard(path, histogram_freq=1)