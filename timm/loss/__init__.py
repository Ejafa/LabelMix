from .asymmetric_loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from .binary_cross_entropy import BinaryCrossEntropy
from .cross_entropy import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    LabelMixSoftTargetCrossEntropy,
)
from .plackett_luce import LabelMixPlackettLuceLoss
from .mixup_loss import LabelMixMixupLoss
from .jsd import JsdCrossEntropy
