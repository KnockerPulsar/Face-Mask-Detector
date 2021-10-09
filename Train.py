from datetime import datetime
import torch as t
from torch._C import device
from torch.nn.modules import lazy
from torch.utils.data import Dataset, DataLoader

import torchvision as tv
from torchvision import utils, datasets
from torchvision.transforms import *

import imutils
from imutils.video import VideoStream

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy
import time
import os

import cv2
from torch.nn import *
from pathlib import Path
from MaskDetector import MaskDetector
from dataset import MaskDataset
from tqdm import tqdm
import argparse
from FaceDetector import FaceDetector

DEFAULT_DATA_PATH = "./data/mask_df.csv"
DEFAULT_IMG_SIZE = 100


def get_val_trns(IMG_SIZE: int) -> Compose:
    return Compose([
        ToPILImage(),
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor()
    ])


def get_train_trns(IMG_SIZE: int) -> Compose:
    return Compose([
        ToPILImage(),
        # RandomAffine(45, translate=(0.2, 0.2)),
        # RandomHorizontalFlip(0.5),
        # RandomResizedCrop(int(0.85*IMG_SIZE)),
        Resize((IMG_SIZE, IMG_SIZE)),

        ToTensor(),
    ])


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloaders[phase], desc=f"{phase} progress: \t", position=0, leave=True, ncols=70):
                inputs = data['image'].to(device)
                labels = data['mask'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with t.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs)
                    conf, preds = t.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += t.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


arg = argparse.ArgumentParser()
arg.add_argument("--csv-location", type=Path, default=Path(DEFAULT_DATA_PATH),
                 help="The path to the csv file that describes the train and validation data. If it doesn't exist then run data_preparation.py")
arg.add_argument("--lazy", type=bool, default=True,
                 help="Whether to use a LazyLinear Layer or a normal Linear layer. For more info, checck MaskDetector.__init__()")
arg.add_argument("--lr", type=float, default=1e-5,
                 help="The initial learning rate")
arg.add_argument("--img-size", type=int, default=100,
                 help="Can be overriden by --lazy, the image size input to the model. Higher allows for more detail, but uses more (V)RAM and trains slower")
arg.add_argument("--batch-size", type=int, default=64,
                 help="How many images to fetch at once. Higher values use more VRAM")
arg.add_argument("--epochs", type=int, default=10,
                 help="How many epochs the model will train for")
arg.add_argument("--use-gpu", type=bool, default=True,
                 help="Whether or not to use the GPU, depends on CUDA availiability")
arg = arg.parse_args()

if __name__ == "__main__":

    if not arg.csv_location:
        print("Data path not given, defaulting to {}")

    if not arg.csv_location.exists():
        print("mask_df.csv does not exist, please run data_preparation.py to download and split the data, then run this script again.")
        exit()
    else:
        print("CSV found, creating mask classifier model and loading dataset")
        img_size = arg.img_size

        if not arg.lazy:
            print(
                f"Not using lazy linear layer, setting img_size to {DEFAULT_IMG_SIZE} for proper operation")
            img_size = DEFAULT_IMG_SIZE

        mask_classifier = MaskDetector(
            Path(arg.csv_location),
            lazy=arg.lazy,
            lr=arg.lr,
            img_size=img_size,
            batch_size=arg.batch_size,
            train_trns=get_train_trns(img_size),
            val_trns=get_val_trns(img_size),
        )

        mask_classifier.prepare_data()

        train_dl = mask_classifier.train_dataloader()
        val_dl = mask_classifier.val_dataloader()

        dls = {
            "train": train_dl,
            "val": val_dl
        }

        if t.cuda.is_available() and arg.use_gpu:
            print("Training on GPU")
        else:
            print("Training on CPU")

        opt = mask_classifier.configure_optimizers()
        print(
            f"training with image size: {img_size}, epochs: {arg.epochs}, learning rate: {arg.lr}, and batch size: {arg.batch_size}")
        print(f"Optimizer: {type(opt).__name__}")

        mask_classifier, best_acc = train_model(
            mask_classifier, dls, mask_classifier.crossEntropyLoss, opt, t.optim.lr_scheduler.StepLR(opt, 2), arg.epochs)

        print("Classifier training complete, saving checkpoint...")

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")

        checkpoint_path = f"./checkpoints/{dt_string} - SZ{img_size}_EP{arg.epochs}_LR{arg.lr}_BS{arg.batch_size}_BA{best_acc*100:.2f}%.ckpt"
        mc = open(checkpoint_path, "wb")
        t.save(
            {
                'state_dict': mask_classifier.state_dict(),
                'lazy': mask_classifier.lazy,
                'img_size': img_size
            },
            mc
        )
        print(f"Checkpoint saved at {checkpoint_path}")
