import json
import os
from copy import deepcopy

json.encoder.FLOAT_REPR = lambda o: format(o, '.2f')


def check_type(type1, value):
    if type1 == 'string':
        return type(value) is str
    elif type1 == 'int':
        return type(value) is int
    elif type1 == 'float':
        return type(float(value)) is float
    elif type1 == 'bool':
        return type(value) is bool
    elif type1 == 'object':
        return type(value) is dict
    else:
        print(f'Новый тип значения: {type1}')


class Config:
    DEFAULT_CONSTRAINTS = 'constraints/'
    INPUT = 'configurations/input'
    OUTPUT = 'configurations/output'
    BASE_NAME = 'configurations'

    def __init__(self, configurations_file: str, constraints_file: str, main_path: str = None) -> None:
        """
        Класс для чтения конфигурационных файлов json.

        Parameters:
        configurations_file : str - Путь к основному конфигурационному файлу.
        constraints_file : str (optional) - Путь к дополнительному конфигурационному файлу. By the default is ''.
        """
        self.__main_path = main_path
        self.__output_name = ''
        self.__load_config(configurations_file, constraints_file)
        self.__output_cfg = None

        base_name = os.path.splitext(os.path.basename(configurations_file))[0]
        version_cfg = base_name.split(self.BASE_NAME)
        if version_cfg:
            self.__version_cfg = '@' + version_cfg[-1][1:]
        else:
            self.__version_cfg = '@'

    def __load_config(self, configurations_file, constraints_file):
        self.__constraints_file, cfg_base = self.__read_file(constraints_file, self.DEFAULT_CONSTRAINTS)
        self.__cfg_base, self.__cfg_type = self.__read_constraints(cfg_base)

        self.__cfg_main = deepcopy(self.__cfg_base)

        self.__configurations_file, cfg_main = self.__read_file(configurations_file, self.INPUT)
        self.__update_cfg(cfg_main)
        self.__time_last = os.path.getmtime(self.__configurations_file)

    def __read_file(self, path: str, default_path: str):
        """
        Чтение конфигурационного файла.
        """
        if not os.path.isfile(path):
            if self.__main_path:
                path = os.path.join(self.__main_path, default_path, path)
            elif '/bin/' in __file__:
                path = os.path.join(*__file__.split('/bin/')[:-1],
                                    default_path, path)
            else:
                path = os.path.join('..', default_path, path)
            if not os.path.isfile(path):
                raise Exception(f'Не найден файл конфига {path}')

        with open(path, "r") as f:
            cfg = json.load(f)

        return os.path.abspath(path), cfg

    def __read_constraints(self, cfg: dict):
        """
        Чтение дополнительного конфигурационного файла.
        """
        cfg_base = dict()
        cfg_type = dict()
        for name, def_value in cfg.items():
            cfg_base[name], cfg_type[name] = self.__div_constr(name, def_value)
        return cfg_base, cfg_type

    def __div_constr(self, name, def_value):
        cfg_type = {'Type': def_value['Type']}
        if def_value['Type'] == 'object':
            cfg_base = dict()
            for n, d_value in def_value.items():
                if isinstance(d_value, dict):
                    cfg_base[n], cfg_type[n] = self.__div_constr(n, d_value)
        else:
            if def_value['Type'] == 'array.object':
                df_value = def_value['Object']
                if 'Type' not in df_value:
                    df_value['Type'] = 'object'
                cfg_type['objects'] = self.__div_constr(0, df_value)

            if 'Max' in def_value and 'Min' in def_value:
                min_v = def_value['Min']
                max_v = def_value['Max']
                interval = [min_v, max_v]
            else:
                interval = None

            if 'Default' not in def_value:
                def_value['Default'] = None

            if not self.__check_value(def_value['Default'], def_value['Type'],
                                      interval):
                raise ValueError(
                    f"{name}: {def_value['Default']} не соответсвует")

            cfg_base = def_value['Default']
            cfg_type['interval'] = interval

        return cfg_base, cfg_type

    def __check_value(self, value, type_value, interval=None):
        if value is None:
            return True

        if interval and len(interval) == 2:
            min_v, max_v = interval
            pr = len(value) if isinstance(value, list) else value
            if pr > max_v or pr < min_v:
                print(f'Значение не входит в интервал {min_v}<={pr}<={max_v}')
                return False

        if isinstance(value, list) and 'array' in type_value:
            if not value:
                return True
            atype = type_value.split('array.')[-1]
            if not any([check_type(atype, x) for x in value]):
                print(f'Значение не соответствует типу {value} ' + f'({type(value)} - {type_value})')
                return False
        else:
            if not check_type(type_value, value):
                print(f'Значение не соответствует типу {value} ' + f'({type(value)} - {type_value})')
                return False
        return True

    def __update_cfg(self, cfg_main):
        list_param = list()
        self.__check_parameters(cfg_main, self.__cfg_base)
        for name, value in cfg_main.items():
            if self.__cfg_main[name] != value:
                self.__cfg_main[name] = self.__update_value(self.__cfg_main[name], value, self.__cfg_type[name])
                list_param.append(f'{name}: {value}')
        return list_param

    def __update_value(self, def_value, value, cfg_type):
        if cfg_type['Type'] == 'object':
            self.__check_parameters(value, def_value)
            res = dict()
            for n, d_value in cfg_type.items():
                if isinstance(d_value, dict):
                    if n not in value:
                        raise ValueError(
                            f'{n} параметр не описан из {value} в файле ' +
                            f'{os.path.basename(self.__configurations_file)}')
                    res[n] = self.__update_value(def_value[n], value[n],
                                                 d_value)
        else:
            res = value
            if cfg_type['Type'] == 'array.object':
                res = list()
                df_v, cfg_t = cfg_type['objects']
                for n, in_value in enumerate(value):
                    res.append(self.__update_value(df_v, in_value, cfg_t))
            if not self.__check_value(value, cfg_type['Type'], cfg_type['interval']):
                res = def_value
        return res

    def __check_parameters(self, cfg_main, cfg_base):
        # Загружаем значения по умолчанию
        set_input = set(cfg_base.keys())
        set_default = set(cfg_main.keys())
        diff_const = set_input.difference(set_default)
        diff_const.discard('ForDebug')
        if diff_const:
            raise ValueError(f'{diff_const} параметры не описаны в ' +
                             f'{os.path.basename(self.__configurations_file)}')
        diff_config = set_default.difference(set_input)
        diff_config.discard('ForDebug')
        if diff_config:
            raise ValueError(f'{diff_config} параметры не описаны в ' + f'{os.path.basename(self.__constraints_file)}')

    def update_cfg(self):
        if not os.path.isfile(self.__configurations_file):
            return

        time_last = os.path.getmtime(self.__configurations_file)

        if time_last != self.__time_last:
            try:
                with open(self.__configurations_file, "r") as f:
                    cfg = json.load(f)
            except Exception as err:
                print(err)
                return

            self.__time_last = time_last

            for name, value in self.__cfg_main.items():
                if self.__cfg_base[name] != value:
                    self.__cfg_base[name] = value
            return self.__update_cfg(cfg)

    def __save_cfg(self) -> None:
        """
        Сохранение загруженного конфига
        """
        path_dir, file = os.path.split(self.__configurations_file)
        path_dir = os.path.join(path_dir.split(self.INPUT, 1)[0], self.OUTPUT)

        self.__output_name = os.path.join(path_dir, file)
        os.makedirs(path_dir, exist_ok=True)
        self.__output_cfg = deepcopy(self.__cfg_main)

        with open(self.__output_name, "w") as f:
            json.dump(self.__output_cfg, f, indent=4)

    def save(self):
        if not self.__output_cfg == self.__cfg_main:
            self.__save_cfg()

    def delete_config(self):
        if os.path.isfile(self.__output_name):
            os.remove(self.__output_name)
            print(f'Файл {self.__output_name} успешно удален')

    @property
    def cfg(self):
        return self.__cfg_main

    @property
    def version(self):
        return self.__version_cfg

    @property
    def cfg_base(self):
        return self.__cfg_base

    @property
    def configurations_file(self):
        return self.__configurations_file

    @property
    def constraints_file(self):
        return self.__configurations_file


if __name__ == '__main__':
    """
        Если при использование параметра произошла ошибка, то берем значение
    по умолчанию. Для примера:

    path_model = conf.cfg_base['DetectorFace']['PathModel']
    if Все удачно заменяем его в value:
        conf.cfg['DetectorFace']['PathModel'] = path_model
    else:
        То закрываем программу
        raise Exception('Не удалось запустить детектор')
    """

    cf1 = '/home/nikit/Загрузки/configurations-1-1.json'
    cf2 = '/home/nikit/Загрузки/constraint.json'
    conf = Config(cf1, cf2)
    cfg = conf.cfg
    cfg1 = conf.cfg_base
