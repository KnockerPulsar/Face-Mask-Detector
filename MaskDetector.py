import torch
import torch.nn.init as init
import pandas as pd
import pytorch_lightning as pl

from torch.nn import (Conv2d, CrossEntropyLoss, Linear,MaxPool2d, ReLU, Sequential)
from torchvision.transforms.transforms import ToPILImage, ToTensor, Compose
from pytorch_lightning.metrics.classification.accuracy import Accuracy
from torchmetrics.classification.accuracy import Accuracy
from sklearn.model_selection import train_test_split
from torch.nn.modules.linear import LazyLinear
from torch.nn.modules.flatten import Flatten
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from dataset import MaskDataset
from torch.optim import Adam
from pathlib import Path
from torch import Tensor

class MaskDetector(pl.LightningModule):
    """ MaskDetector PyTorch Lightning class
    """

    def __init__(
            self, maskDFPath: Path = None,
            lazy: bool = False,
            batch_size: int = 32,
            lr=1e-5,
            train_trns=Compose([ToPILImage(), ToTensor()]),
            val_trns=Compose([ToPILImage(), ToTensor()]),
            img_size=100
    ):

        super(MaskDetector, self).__init__()
        self.maskDFPath = maskDFPath

        self.maskDF = None
        self.trainDF = None
        self.validateDF = None
        self.crossEntropyLoss = None
        self.learningRate = lr

        self.trainAcc = Accuracy()
        self.valAcc = Accuracy()

        self.batch_size = batch_size
        self.train_trns = train_trns
        self.val_trns = val_trns

        self.convLayer1 = convLayer1 = Sequential(
            Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayer2 = convLayer2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayer3 = convLayer3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3, 3)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

    # Allows for variable image sizes
        linear = LazyLinear(out_features=1024) if lazy else Linear(
            in_features=2048, out_features=1024)

        if lazy:
            self.linearLayers = linearLayers = Sequential(
                Flatten(),
                linear,
                ReLU(),
                Linear(in_features=1024, out_features=2),
            )
        else:
            self.linearLayers = linearLayers = Sequential(
                linear,
                ReLU(),
                Linear(in_features=1024, out_features=2),
            )

        self.lazy = lazy

        # Initializing the lazy layer so that the for loop below doesn't crash
        if lazy:
            img = torch.zeros((3, img_size, img_size))
            self.forward(img.unsqueeze(0))

        # Initialize layers' weights
        for sequential in [convLayer1, convLayer2, convLayer3, linearLayers]:
            for layer in sequential.children():
                if isinstance(layer, (Linear, Conv2d)):
                    init.xavier_uniform_(layer.weight)

    def forward(self, x: Tensor):  # pylint: disable=arguments-differ
        """ forward pass
        """
        out = self.convLayer1(x)
        out = self.convLayer2(out)
        out = self.convLayer3(out)

        if not self.lazy:
            out = out.view(-1, 2048)

        out = self.linearLayers(out)
        return out

    def prepare_data(self) -> None:
        self.maskDF = maskDF = pd.read_csv(self.maskDFPath)
        train, validate = train_test_split(maskDF, test_size=0.3, random_state=0,
                                           stratify=maskDF['mask'])
        self.trainDF = MaskDataset(train, self.train_trns)
        self.validateDF = MaskDataset(validate, self.val_trns)

        # Create weight vector for CrossEntropyLoss
        maskNum = maskDF[maskDF['mask'] == 1].shape[0]
        nonMaskNum = maskDF[maskDF['mask'] == 0].shape[0]
        nSamples = [nonMaskNum, maskNum]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        self.crossEntropyLoss = CrossEntropyLoss(
            weight=torch.tensor(normedWeights))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainDF, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validateDF, batch_size=self.batch_size, num_workers=4)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learningRate)
