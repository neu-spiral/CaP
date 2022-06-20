# small-caps refers to cifar-style models i.e., resnet18 -> for cifar vs ResNet18 -> standard arch.
from .resnet import resnet18, resnet34, resnet50
from .escnet import EscFusion
from .wrn import wrn16_8, wrn28_10, wrn28_4
from .flashnet import InfoFusionThree


__all__ = [
    "wrn16_8",
    "wrn28_10",
    "wrn28_4",
    "resnet18",
    "resnet34",
    "resnet50",
    "InfoFusionThree",
    "EscFusion"
]
