# small-caps refers to cifar-style models i.e., resnet18 -> for cifar vs ResNet18 -> standard arch.
from models.resnet import resnet18, resnet34, resnet50
from models.wrn import wrn16_8, wrn28_10, wrn28_4
from models.flashnet import InfoFusionThree


__all__ = [
    "wrn16_8",
    "wrn28_10",
    "wrn28_4",
    "resnet18",
    "resnet34",
    "resnet50",
    "InfoFusionThree"
]
