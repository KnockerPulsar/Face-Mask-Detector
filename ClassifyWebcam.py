import cv2
import argparse
import torch as t
from torch._C import device

from utils import detect_frame, load_mask_classifier
from FaceDetector import FaceDetector
from torchvision.transforms import *

arg = argparse.ArgumentParser()
arg.add_argument("--checkpoint", type=str,
                 help="Path to the checkpoint of the classifier model",
                 default="./checkpoints/face_mask.ckpt")
arg.add_argument("--viewer-res", type=int,
                 help="The output viewer resolutions", default=600)
arg = arg.parse_args()

if __name__ == "__main__":

    device = None
    if t.cuda.is_available():
        print("GPU available, using it...")
        device=t.device("cuda:0")
    else:
        print("GPU not available, using CPU...")
        device=t.device("cpu")

    model, val_trns = load_mask_classifier(arg.checkpoint, device)

    face_detector = FaceDetector(prototype='./checkpoints/deploy.prototxt.txt',
                                 model='./checkpoints/res10_300x300_ssd_iter_140000.caffemodel')

    print("Successfully loaded model, initializing webcam...")
    print("Press q to quit")
    vs = cv2.VideoCapture(0)
    cv2.namedWindow("Results", cv2.WINDOW_NORMAL)

    while True:

        ret, frame = vs.read()
        frame = detect_frame(frame, face_detector, model, device, val_trns)

        cv2.imshow("Results", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Exiting")
    vs.release()
    cv2.destroyWindow("Results")
