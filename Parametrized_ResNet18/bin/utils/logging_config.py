from typing import Optional
import logging
from logging.handlers import RotatingFileHandler
from logging.handlers import TimedRotatingFileHandler


DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR


def name_to_level(level_name):
    """
    Get logging level code by name.
    """
    level_name = level_name.upper()
    if level_name in logging._nameToLevel:
        return logging._nameToLevel[level_name]
    return INFO

def get_logger(name: str,
               level: Optional[int] = logging.INFO,
               log_to_console: Optional[bool] = True,
               log_file_name: Optional[str] = None,
               error_file_name: Optional[str] = None,
               datefmt: Optional[str] = '%d-%m-%y %H:%M:%S',
               midnight: Optional[bool] = True,
               log_rotation: Optional[int] = 7) -> logging.getLogger:
    """
    Функция для создания лога.

    Parameters:
        name: str - Название лога
        level: logging.INFO (optional) - Уровень лога
        log_to_console: bool (optional) - Писать ли в консоль
        log_file_name: str (optional) - Путь для создание файла лога
        error_file_name: str (optional) - Путь для создание файла для записи ошибок
        datefmt: str (optional) - Формат времени при записи в логах
        midnight: bool (optional) - Лог пересоздается каждую полночь и хранится day_ratation дней
        log_rotation: int (optional) - Срок в днях сколько хранится файл лога, применяется только при midnight = True

    Returns: 
        log: logging.getLogger- Объект логгера
    """
    
    log = logging.getLogger(name)
    log.propagate = False

    if isinstance(level, str):
        level = name_to_level(level)
    log.setLevel(level)

    if level == DEBUG:
        strfrm = ('[%(asctime)s.%(msecs)03d] %(levelname)s: %(filename)s(%(lineno)d): %(message)s')
    else:
        strfrm = ('[%(asctime)s.%(msecs)03d] %(levelname)s: %(filename)s: %(message)s')

    log_fmt = logging.Formatter(strfrm, datefmt=datefmt)
    log.handlers = []

    if log_to_console:
        console = logging.StreamHandler()
        console.setFormatter(log_fmt)
        log.addHandler(console)

    if log_file_name:
        if midnight:
            log_file = TimedRotatingFileHandler(log_file_name,
                                                backupCount=log_rotation,
                                                when='midnight')
        else:
            log_file = logging.FileHandler(log_file_name)
        log_file.setFormatter(log_fmt)
        log.addHandler(log_file)

    if error_file_name:
        error_file = RotatingFileHandler(
            error_file_name, mode='a', backupCount=1, maxBytes=10_485_760)
        error_file.setLevel(ERROR)

        err_frm = ('[%(asctime)s.%(msecs)03d] %(levelname)s: ' + '%(filename)s(%(lineno)d): %(message)s')
        error_file.setFormatter(logging.Formatter(err_frm, datefmt='%y-%m-%d %H:%M:%S'))
        log.addHandler(error_file)

    return log
