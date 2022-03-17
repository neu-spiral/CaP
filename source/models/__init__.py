# small-caps refers to cifar-style models i.e., resnet18 -> for cifar vs ResNet18 -> standard arch.
from models.wrn import wrn28_10, wrn28_4

from models.resnet import resnet18, resnet34, resnet50

__all__ = [
    "wrn28_10",
    "wrn28_4",
    "resnet18",
    "resnet34",
    "resnet50",
]
