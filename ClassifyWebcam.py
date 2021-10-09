import torch as t

from torchvision.transforms import *

from imutils.video import VideoStream

import cv2
from MaskDetector import MaskDetector
import argparse
from FaceDetector import FaceDetector

from utils import detect_frame, load_mask_classifier

import traceback


arg = argparse.ArgumentParser()
arg.add_argument("--checkpoint", type=str,
                 help="Path to the checkpoint of the classifier model",
                 default="./checkpoints/face_mask.ckpt")
arg.add_argument("--viewer-res", type=int,
                 help="The output viewer resolutions", default=600)
arg.add_argument("--use-gpu", type=bool, default=True,
                 help="Whether or not to use the GPU. Depends on CUDA, so might not work if there are problems with it")
arg = arg.parse_args()

if __name__ == "__main__":


    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

    model, val_trns = load_mask_classifier(arg.checkpoint, device)

    face_detector = FaceDetector(prototype='./checkpoints/deploy.prototxt.txt',
                                 model='./checkpoints/res10_300x300_ssd_iter_140000.caffemodel')

    print("Successfully loaded model, initializing webcam...")
    print("Press q to quit")
    vs = VideoStream(src=0).start()

    while True:

        frame = vs.read()
        frame = detect_frame(frame, face_detector, model, device,val_trns)

        frame = cv2.resize(frame, (arg.viewer_res,arg.viewer_res))
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print("Exiting")
    vs.stop()
    cv2.destroyAllWindows()
