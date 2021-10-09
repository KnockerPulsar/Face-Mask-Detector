from datetime import datetime
from genericpath import exists
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
from utils import detect_frame, load_mask_classifier

arg = argparse.ArgumentParser()
arg.add_argument("img_path", type=str,
                 help="The path to the image we want to classify")
arg.add_argument("--checkpoint", type=str, default="./checkpoints/face_mask.ckpt",
                 help="The path to the model checkpoint you want to classify with")
arg.add_argument("--show-result", type=bool, default=True,
                 help="Whether to show the classified image or not. Best used with --output-path")
arg.add_argument("--output-dir", type=str,
                 help="Where to output the classified image. Will not output if a directory is not given")

arg = arg.parse_args()


if __name__ == "__main__":
    img_path = Path(arg.img_path)
    out_path = Path(arg.output_dir)

    if not img_path.exists():
        print("The given image path is not valid, exiting...")
    elif arg.output_dir is not None and (not out_path.exists() or not out_path.is_dir()):
        print("The image output path is not valid, exiting...")
    else:
        img = cv2.imread(arg.img_path)
        print(f"Loaded image at {arg.img_path}")

        device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

        model, val_trns = load_mask_classifier(arg.checkpoint, device)

        face_detector = FaceDetector(prototype='./checkpoints/deploy.prototxt.txt',
                                     model='./checkpoints/res10_300x300_ssd_iter_140000.caffemodel')

        print("Face detector and mask classifier loaded successfully, passing image to mode...")
        img = detect_frame(img, face_detector, model, device, val_trns)


        if arg.output_dir is not None:
            split = list(os.path.split(arg.output_dir))
            split[-1] = "result_" + img_path.name
            op = os.path.sep.join(split)
            print(f"Writing result at {op}")

            cv2.imwrite(op, img)

        if arg.show_result:
            print("Showing results, press any key to exit.")
            cv2.imshow("Result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

