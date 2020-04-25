from pytorch_src.models.aerts import Aerts
from pytorch_src.models.base_model import ImageSiamese


def load_model(model_name, height, width, depth):
    if model_name == "aerts":
        model = Aerts(height, width, depth)
    elif model_name == "imagesiamese":
        model = ImageSiamese()

    return model
