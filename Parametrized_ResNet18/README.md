## Parametrize the weight tensor of a convolutional layer

### Работа модели
В этом репозитории представлено решение задания Tech Task (AI Research Engineer).html.

### Параметры
Перед запуском модуля bin/parametrized_resnet18.py необходимо отредактировать файл настроек configurations/input/configurations-1.json. Описание параметров, их формат и ограничения на них описаны в constraints/constraint.json.

### Логирование
Логи запуска программы сохраняются в папку logs. Параметры с которыми был последний фактический запуск сервера сохраняются в папку configurations/output.

### Запуск:
_Запускать модули и ставить все зависимости рекомендуется в виртуальном окружении!_

Перед запуском необходимо установить зависимости: **pip install -r requirements.txt**

Есть несколько способов запуска программы: 
- Через терминал или tmux (из директории ner-server/bin): **python3 parametrized_resnet18.py ../configurations/input/configurations-1.json ../constraints/constraint.json**
- Через службу systemd (для этого необходимо создать файл /lib/systemd/system/ner-server.service): **sudo systemctl start ner-server.service**