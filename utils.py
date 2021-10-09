from torchvision.transforms import *
from torch.nn import *
import torch as t
import cv2

from MaskDetector import MaskDetector


labels = ['Mask', 'No mask']
labelColor = [(10, 255, 0), (10, 0, 255)]
font = cv2.FONT_HERSHEY_SIMPLEX

def detect_frame(frame, face_detection_model, face_classifier_model, device, val_trns):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detection_model.detect(frame)

    for face in faces:
        x_start, y_start, x_end, y_end = face

        x_start, y_start = max(x_start, 0), max(y_start, 0)

        faceImg = frame[y_start:y_end, x_start:x_end]

        output = face_classifier_model(
            val_trns(faceImg).unsqueeze(0).to(device))

        # print(output)

        output = Softmax(dim=-1)(output)
        conf, predicted = t.max(output.data, 1)
        verdict = f"{labels[predicted]}: {conf.item()*100:.2f}%"

        cv2.rectangle(frame,
                      (x_start, y_start),
                      (x_end, y_end),
                      labelColor[predicted],
                      thickness=2)

        # draw prediction label
        cv2.putText(frame,
                    verdict,
                    (x_start, y_start-20),
                    font, 0.5, labelColor[predicted], 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def load_mask_classifier(checkpoint, device):
    checkpoint = t.load(checkpoint, map_location=device)

    lazy = False
    img_size = 100

    if 'lazy' in checkpoint.keys():
        lazy = checkpoint['lazy']

    if 'img_size' in checkpoint.keys():
        img_size = checkpoint['img_size']


    val_trns = Compose([
        ToPILImage(),
        Resize((img_size, img_size)),
        ToTensor()
    ])

    model = MaskDetector(".", lazy=lazy, img_size=img_size, batch_size=1,)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    model.to(device)

    return model, val_trns