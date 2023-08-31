FROM python:3.9-slim as base
LABEL maintainer="Lindemann Nikita,  lindemann-nikita@rambler.ru"

# Сборка зависимостей
ARG BUILD_DEPS="python3 -m spacy download en_core_web_sm"
RUN apt-get update && apt-get install -y $BUILD_DEPS

# Инициализация проекта
WORKDIR /opt/ner/ner-server

# отключение буферизации вывода (output)
ENV PYTHONDONTWRITEBYTECODE 1
# Python не будет пытаться создавать файлы .pyc
ENV PYTHONUNBUFFERED 1

# Копирование в контейнер папок и файлов.
COPY . . 

# Установка Python библиотек
RUN pip install -r bin/requirements.txt

# Устанавливаем порт, который будет использоваться для сервера
EXPOSE 7000

# Запускать интерпретатор python3 по умолчанию
ENTRYPOINT ["python3", "bin/ner-server.py"]

# В сочетании с использованием ENTRYPOINT интерпретатору python будет передан дополнительный аргумент
CMD ["configurations/input/configurations-1.json", "constraints/constraint.json"]