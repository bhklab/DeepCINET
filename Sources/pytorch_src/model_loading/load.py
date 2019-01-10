from pytorch_src.models.aerts import Aerts


def load_model(model_name, height, width, depth):
    if model_name == "aerts":
        model = Aerts(height, width, depth)

    return model