from .alexnet import AlexNet
from .resnet import make_resnet
from .vgg import make_vgg


def build_model(model, dataset, act, bn):
    if "vgg" in model.lower():
        model = make_vgg(model, dataset, act, bn)
    elif "resnet" in model.lower():
        model = make_resnet(model, dataset, act, bn)
    elif model.lower() == "alexnet":
        model = AlexNet(dataset, act, bn)
    elif model.lower() == "dnn":
        model = None
    return model
