import os
import cv2
import argparse
import torch as t

from utils import detect_frame, load_mask_classifier
from skvideo.io import FFmpegWriter, vreader, ffprobe
from FaceDetector import FaceDetector
from torchvision.transforms import *
from pathlib import Path
from torch.nn import *
from tqdm import tqdm

arg = argparse.ArgumentParser()
arg.add_argument("vid_path", type=str,
                 help="The path to the image we want to classify")
arg.add_argument("--checkpoint", type=str, default="./checkpoints/face_mask.ckpt",
                 help="The path to the model checkpoint you want to classify with")
arg.add_argument("--show-result", type=bool, default=True,
                 help="Whether to show the classified image or not. Best used with --output-path")
arg.add_argument("--output-dir", type=str,
                 help="Where to output the classified image. Will not output if a directory is not given")
arg.add_argument("--bitrate", type=int, default=3000000,
                 help="The bitrate of the output video.")

# Seems to not work?
# arg.add_argument("--num-threads", type=int, default=4,
#                  help="The number of threads to use when writing the output video")

arg = arg.parse_args()


@t.no_grad()
def classify_video() -> None:
    vid_path = Path(arg.vid_path)

    if arg.output_dir is not None:
        out_path = Path(arg.output_dir)

    if not vid_path.exists():
        print("The given video path is not valid, exiting...")
    elif arg.output_dir is not None and (not out_path.exists() or not out_path.is_dir()):
        print("The image output path is not valid, exiting...")
    else:

        device = None
        if t.cuda.is_available():
            print("Using GPU")
            device = t.device("cuda:0")
        else:
            print("Using CPU")
            device = t.device("CPU")

        print("Attempting to load mask classifier checkpoint")
        model, val_trns = load_mask_classifier(arg.checkpoint, device)
        print("Mask classifier checkpoint successfully loaded")

        print("Attempting to load face detector checkpoint")
        face_detector = FaceDetector(prototype='./checkpoints/deploy.prototxt.txt',
                                     model='./checkpoints/res10_300x300_ssd_iter_140000.caffemodel')
        print("Face detector checkpoint successfully loaded")

        if arg.output_dir is not None:
            split = list(os.path.split(arg.output_dir))
            split[-1] = "result_" + vid_path.name
            op = os.path.sep.join(split)
            print(f"Will write result at result at {op}")

        print("Loading and classifying video frames")
        cv2.namedWindow("Results", cv2.WINDOW_NORMAL)
        classified_frames = []

        for frame in vreader(arg.vid_path):

            frame = detect_frame(frame, face_detector, model,
                                 device, val_trns, opencv_frame=False)

            if arg.output_dir:
                classified_frames.append(frame)

            if arg.show_result:
                # Since openCV wants a BGR image
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Results", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

        if arg.output_dir:
            print("Saving classified video...")
            metadata = ffprobe(arg.vid_path)
            writer = FFmpegWriter(
                op,
                inputdict={'-r': str(metadata['video']['@avg_frame_rate'])},
                # outputdict={'-pix_fmt': 'yuv444p', '-b': str(arg.bitrate), '-threads': str(arg.num_threads)})
                outputdict={'-pix_fmt': 'yuv444p', '-b': str(arg.bitrate)})

            for frame in tqdm(classified_frames):
                writer.writeFrame(frame)
            writer.close()

        if arg.output_dir:
            writer.close()


if __name__ == "__main__":
    classify_video()
