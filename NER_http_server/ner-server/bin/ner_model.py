import spacy
import torch
from typing import List
import sys
import os
sys.path.insert(0, os.path.abspath("pytorch_ner"))

from pytorch_ner.model import init_model_from_config

PATH2MODEL_CONFIG = "/home/evgeny/my_projects/ner/pytorch-ner/config.yaml"
PATH2MODEL_WEIGHTS = "model/model.pth"


class BaseNerModel:
    """
    Baseline модели для распознавания именованных сущностей. 
    Используется готовая предобученная модель без дообучения.
    """
    def __init__(self):
        self.__model = spacy.load("en_core_web_sm")
        
    def predict(self, row_text: str) -> List[tuple]:
        """
        Предсказание модели. Возвращает список пар (слово, тип именованной сущности).
        
        Parameters:
            row_text: str - текст, в котором необходимо найти именованные сущности
        """
        result = []
        doc = self.__model(row_text)

        for ent in doc.ents:
            result.append((ent.text, ent.label_))
            
        return result
    
    
if __name__ == "__main__":
    ner_english_model = BaseNerModel()
    print(f"Loaded {ner_english_model.__class__.__name__}")

    ner_russian_model = init_model_from_config(PATH2MODEL_CONFIG)
    ner_russian_model.load_state_dict(torch.load(PATH2MODEL_WEIGHTS))
    print(f"Loaded {ner_russian_model.__class__.__name__}")

    english_text = "Apple Inc. is a technology company based in Cupertino, California."
    russian_text = "Частные российские инвесторы подали иск к депозитарию 'Clearstream' в Арбитражный суд Москвы."

    print(f"Predict {ner_english_model.__class__.__name__}:", ner_english_model.predict(english_text))
    print(ner_russian_model)
    