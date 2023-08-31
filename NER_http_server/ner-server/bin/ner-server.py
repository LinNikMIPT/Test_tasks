#!/usr/bin/python3

from flask import Flask, request
from flask_api import status
from datetime import datetime
from time import time
from pathlib import Path
import torch

from utils.logging_config import get_logger
from ner_model import BaseNerModel

import sys
import os
sys.path.insert(0, os.path.abspath("pytorch_ner"))

from pytorch_ner.model import init_model_from_config


OK_RESPONSE = {"Status": "Ok",
               "Error_num": 0,
               "Response": None}
ERROR_BASE_RESPONSE = {"Status": "Error",
                       "Error_num": 1,
                       "Response": None}
INVALID_PARAM_RESPONSE = {"Status": "Error",
                          "Error_num": 2,
                          "Response": None}
LANGUAGES = ["Russian", "English"]
PATH2MODEL_CONFIG = "model/config.yaml"
PATH2MODEL_WEIGHTS = "model/model.pth"


def timer2log(func, logger: get_logger, time_rounding_accuracy: int = 6):
    """
    Декоратор, подсчитывающий время выполнения функции в миллисекундах и записывающий результат в лог.
    
    Parameters:
        func - декорируемая функция
        logger - объект логера, в который записывается результат работы декоратора
        time_rounding_accuracy - точность округления времени
    """
    def wrapped(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        logger.info(f"{func.__name__} time: {round(10**3 * (end_time - start_time), time_rounding_accuracy)} ms")
        return result
    return wrapped


def init_logging(log_rotation=7):
    file = Path(__file__).absolute()

    # Установим директорию для логов
    logdir = file.parent.parent / 'logs'
    logdir.mkdir(exist_ok=True)

    # Зададим имя лог файла
    startup = datetime.fromtimestamp(time()).strftime('%d:%m:%y')
    log_file_name = str(logdir / f'{startup}_{file.stem}.log')

    logger = get_logger(name=os.path.basename(__file__),
                        log_file_name=log_file_name,
                        datefmt='%d-%m-%y %H:%M:%S',
                        midnight=True,
                        log_rotation=log_rotation)
    logger.info(f'Initialization logger for {os.path.basename(__file__)}')
    return logger


def is_valid_request(req_json: dict) -> bool:
    """
    Функция проверяет валидность запроса:
        - В запросе два ключа: "Language" и "Data"
        - Поле "Language" принимает одно из значений: "English" или "Russian"

    Parameters:
        - req_json - словарь запроса
    """
    if len(req_json) != 2:
        return False
    if "Language" not in req_json or "Data" not in req_json or req_json["Language"] not in LANGUAGES:
        return False
    return True


app = Flask(__name__, template_folder="./")
logger = init_logging()

english_ner_model = BaseNerModel()
logger.info(f'Initialization NER model {english_ner_model.__class__.__name__}')

russian_ner_model = init_model_from_config(PATH2MODEL_CONFIG)
russian_ner_model.load_state_dict(torch.load(PATH2MODEL_WEIGHTS))
logger.info(f'Initialization NER model {russian_ner_model.__class__.__name__}')

english_model_predict_with_timer = timer2log(english_ner_model.predict, logger)
# russian_model_predict_with_timer = timer2log(russian_ner_model.predict, logger)


@app.route('/ner', methods=['GET', 'POST'])
def ner():
    try:
        req_data = request.json
        logger.info(f"Get request json: {req_data}")
        
        # Проверка валидности запроса
        if not is_valid_request(req_data):
            return INVALID_PARAM_RESPONSE, status.HTTP_400_BAD_REQUEST
        
        try:
            prediction = english_model_predict_with_timer(req_data["Data"])
            # if req_data["Data"] == "English":
            #     prediction = english_model_predict_with_timer(req_data["Data"])
            # else:
            #     prediction = russian_model_predict_with_timer(req_data["Data"])
            OK_RESPONSE["Response"] = prediction
            return OK_RESPONSE, status.HTTP_200_OK
        except Exception as e:
            logger.exception(f"Error in model prediction!")
            return f"Error info: {e}", status.HTTP_400_BAD_REQUEST
    except Exception:
        logger.exception(f"Critical error in {ner.__name__}!")
        return ERROR_BASE_RESPONSE, status.HTTP_400_BAD_REQUEST


if __name__ == '__main__':
    import argparse
    from utils.config_reader_json import Config

    fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description=os.path.basename(__file__),
                                     formatter_class=fmt)
    parser.add_argument('config_path', type=str, help='Путь до конфигурационного файла',
                        default="../configurations/input/configurations-1.json")
    parser.add_argument('constrain_path', type=str, help='Путь до конфигурационного файла с параметрами по умолчанию',
                        default="../constraints/constraint.json")
    args = parser.parse_args()
    
    conf = Config(configurations_file=args.config_path, constraints_file=args.constrain_path)
    parameters = conf.cfg

    app.run(host=parameters["host"], port=parameters["port"], debug=False)
