import logging
import importlib
from collections import Counter
from typing import Any, Dict
import torch
import torch.nn as nn
import random
import numpy as np

from config import get_config
from logger import get_logger
from nn_modules.architecture import BiLSTM
from nn_modules.embedding import Embedding
from nn_modules.linear import LinearHead
from nn_modules.rnn import DynamicRNN
from prepare_data import get_label2idx, get_token2idx, prepare_conll_data_format


def set_global_seed(seed: int):
    """
    Set global seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str_to_class(module_name, class_name):
    """
    Convert string to Python class object.
    https://stackoverflow.com/questions/1176136/convert-string-to-python-class-object
    """

    # load the module, will raise ImportError if module cannot be loaded
    module = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    cls = getattr(module, class_name)
    return cls


def init_model(config: Dict[str, Any],
               logger: logging.Logger):
    """
    Main function to init NER model.
    """

    # log config
    with open(config["save"]["path_to_config"], mode="r") as fp:
        logger.info(f"Config:\n\n{fp.read()}")

    device = torch.device(config["torch"]["device"])
    set_global_seed(config["torch"]["seed"])

    # tokens / labels sequences
    train_token_seq, train_label_seq = prepare_conll_data_format(path=config["data"]["train_data"]["path"],
                                                                 sep=config["data"]["train_data"]["sep"],
                                                                 lower=config["data"]["train_data"]["lower"],
                                                                 verbose=config["data"]["train_data"]["verbose"])

    # token2idx / label2idx
    token2cnt = Counter([token for sentence in train_token_seq for token in sentence])
    label_set = sorted(set(label for sentence in train_label_seq for label in sentence))

    token2idx = get_token2idx(token2cnt=token2cnt,
                              min_count=config["data"]["token2idx"]["min_count"],
                              add_pad=config["data"]["token2idx"]["add_pad"],
                              add_unk=config["data"]["token2idx"]["add_unk"])

    label2idx = get_label2idx(label_set=label_set)

    # INIT MODEL
    # TODO: add more params to config.yaml
    # TODO: add pretrained embeddings
    # TODO: add dropout
    embedding_layer = Embedding(num_embeddings=len(token2idx),
                                embedding_dim=config["model"]["embedding"]["embedding_dim"])

    rnn_layer = DynamicRNN(rnn_unit=str_to_class(module_name="torch.nn",
                                                 class_name=config["model"]["rnn"]["rnn_unit"]),
                           input_size=config["model"]["embedding"]["embedding_dim"],  # ref to emb_dim
                           hidden_size=config["model"]["rnn"]["hidden_size"],
                           num_layers=config["model"]["rnn"]["num_layers"],
                           dropout=config["model"]["rnn"]["dropout"],
                           bidirectional=config["model"]["rnn"]["bidirectional"])

    # TODO: add attention if needed in config
    linear_head = LinearHead(
        linear_head=nn.Linear(
            in_features=((2 if config["model"]["rnn"]["bidirectional"] else 1) * config["model"]["rnn"]["hidden_size"]),
            out_features=len(label2idx),
        )
    )

    # TODO: add model architecture in config
    # TODO: add attention if needed
    model = BiLSTM(embedding_layer=embedding_layer,
                   rnn_layer=rnn_layer,
                   linear_head=linear_head)

    return model


def init_model_from_config(path_to_config: str):
    # load config
    config = get_config(path_to_config=path_to_config)

    # get logger
    logger = get_logger(path_to_logfile=config["save"]["path_to_save_logfile"])

    model = init_model(config=config,
                       logger=logger)

    return model


if __name__ == "__main__":
    path2config = "/home/evgeny/my_projects/ner/pytorch-ner/config.yaml"
    ner_model = init_model_from_config(path2config)

    print(ner_model)
