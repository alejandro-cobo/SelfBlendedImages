import json
from PIL import Image
import numpy as np
import albumentations as alb
import cv2


def load_json(path):
    d = {}
    with open(path, mode="r") as f:
        d = json.load(f)
    return d


def IoUfrom2bboxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def crop_face(img, landmark=None, bbox=None, bbox_scale=1.3):
    assert bbox is not None or landmark is not None

    if bbox is not None:
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
    else:
        x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
        x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()

    w = x1 - x0
    h = y1 - y0
    center_x = x0 + w / 2
    center_y = y0 + h / 2
    side = max(w, h) * bbox_scale
    x0_new = int(center_x - side / 2)
    x1_new = int(center_x + side / 2)
    y0_new = int(center_y - side / 2)
    y1_new = int(center_y + side / 2)

    img_cropped = np.array(Image.fromarray(img).crop((x0_new, y0_new, x1_new, y1_new)))
    if landmark is not None:
        landmark_cropped = np.zeros_like(landmark)
        for i, (p, q) in enumerate(landmark):
            landmark_cropped[i] = [p - x0_new, q - y0_new]
    else:
        landmark_cropped = None
    if bbox is not None:
        bbox_cropped = np.zeros_like(bbox)
        for i, (p, q) in enumerate(bbox):
            bbox_cropped[i] = [p - x0_new, q - y0_new]
    else:
        bbox_cropped = None

    return img_cropped, landmark_cropped, bbox_cropped


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self, img, **params):
        return self.randomdownscale(img)

    def randomdownscale(self, img):
        keep_input_shape = True
        H, W, C = img.shape
        ratio_list = [2, 4]
        r = ratio_list[np.random.randint(len(ratio_list))]
        img_ds = cv2.resize(
            img, (int(W / r), int(H / r)), interpolation=cv2.INTER_NEAREST
        )
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W, H), interpolation=cv2.INTER_LINEAR)

        return img_ds
