from .__main__ import train
import model
from .model import init_model_from_config

__version__ = "0.1.0"
__all__ = ["train", "model", "init_model_from_config"]
