import requests
import os
from time import time

DATA_PATH = "send_data2server"
NER_HTTP_SERVER_ADD = "http://127.0.0.1:7000/ner"


def timer2list(func, time_list: list, time_rounding_accuracy: int = 6):
    """
    Декоратор, подсчитывающий время выполнения функции в миллисекундах и добавляющий это время в список time_list.
    
    Parameters:
        func - декорируемая функция
        time_rounding_accuracy - точность округления времени
    """
    def wrapped(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        work_time = round(10**3 * (end_time - start_time), time_rounding_accuracy)
        time_list.append(work_time)
        print(f"{func.__name__} time: {work_time} ms", "\n")
        return result
    return wrapped


def sent_http_req(data2send: dict, add: str = NER_HTTP_SERVER_ADD) -> None:
    try:
        response = requests.post(add, json=data2send)
        print(f"Response code: {response.status_code}")
        print(f"Response json: {response.json()}")
    except Exception as e:
        print(f"Critical error in {sent_http_req.__name__}: {e}")


if __name__ == "__main__":    
    work_times = []
    docs = ["english_text.txt", "russian_text.txt"]

    sent_http_req_with_timer = timer2list(func=sent_http_req, time_list=work_times)

    for doc in docs:
        if "english" in doc:
            language = "English"
        elif "russian" in doc:
            language = "Russian"
        else:
            print(f"Unknown doc: {doc}. Skipping...")
            continue

        with open(os.path.join(DATA_PATH, doc), "r") as f:
            data = f.read()

        sentences = [x for x in data.split("\n") if x != ""]

        for sentence in sentences:
            data2send = {"Language": language,
                         "Data": sentence}
            sent_http_req_with_timer(data2send)
        
    print(200*"-")
    print(f"Среднее время ответа сервера: {sum(work_times)/len(work_times)} мс")
