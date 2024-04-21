##
# Project 2024
# neuraltest [WSL: Ubuntu-22.04]
# File description:
# __init__
##


from neuralib.constants import LossID

from .ILoss import ILoss

from .CrossEntropy import CategoricalCrossEntropy, BinaryCrossEntropy
from .MeanAbsoluteError import MAE
from .MeanSquaredError import MSE

default_loss_functions = {
    LossID.CROSSENTROPY_CATEGORICAL: CategoricalCrossEntropy,
    LossID.CROSSENTROPY_BINARY: BinaryCrossEntropy,
    LossID.MEAN_ABOSOLUTE_ERROR: MAE,
    LossID.MEAN_SQUARED_ERROR: MSE
}


def getLossFromID(id: str) -> ILoss:
    id = id.lower()
    if (id not in default_loss_functions):
        raise "Error loss function " + id + "not found "
    return (default_loss_functions[id]())
