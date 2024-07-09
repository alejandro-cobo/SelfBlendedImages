import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def extract_frames(filename, num_frames, model, image_size=(380, 380)):
    cap_org = cv2.VideoCapture(filename)

    if not cap_org.isOpened():
        print(f"Cannot open: {filename}")
        return []

    croppedfaces = []
    idx_list = []
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idxs = np.linspace(
        0, frame_count_org - 1, num_frames, endpoint=True, dtype=int
    )
    for cnt_frame in range(frame_count_org):
        ret_org, frame_org = cap_org.read()
        height, width = frame_org.shape[:-1]
        if not ret_org:
            tqdm.write(
                "Frame read {} Error! : {}".format(
                    cnt_frame, os.path.basename(filename)
                )
            )
            break

        if cnt_frame not in frame_idxs:
            continue

        frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)

        faces = model.predict_jsons(frame)
        try:
            if len(faces) == 0:
                tqdm.write(
                    "No faces in {}:{}".format(cnt_frame, os.path.basename(filename))
                )
                continue

            size_list = []
            croppedfaces_temp = []
            idx_list_temp = []

            for face_idx in range(len(faces)):
                x0, y0, x1, y1 = faces[face_idx]["bbox"]
                bbox = np.array([[x0, y0], [x1, y1]])
                cropped_face, _, __ = crop_face(frame, None, bbox, bbox_scale=1.3)
                croppedfaces_temp.append(cv2.resize(cropped_face, dsize=image_size,).transpose((2, 0, 1)))
                idx_list_temp.append(cnt_frame)
                size_list.append((x1 - x0) * (y1 - y0))

            max_size = max(size_list)
            croppedfaces_temp = [
                f
                for face_idx, f in enumerate(croppedfaces_temp)
                if size_list[face_idx] >= max_size / 2
            ]
            idx_list_temp = [
                f
                for face_idx, f in enumerate(idx_list_temp)
                if size_list[face_idx] >= max_size / 2
            ]
            croppedfaces += croppedfaces_temp
            idx_list += idx_list_temp
        except Exception as e:
            print(f"error in {cnt_frame}:{filename}")
            print(e)
            continue
    cap_org.release()

    return croppedfaces, idx_list


def extract_face(frame, model, image_size=(380, 380)):

    faces = model.predict_jsons(frame)

    if len(faces) == 0:
        print("No face is detected")
        return []

    croppedfaces = []
    for face_idx in range(len(faces)):
        x0, y0, x1, y1 = faces[face_idx]["bbox"]
        bbox = np.array([[x0, y0], [x1, y1]])
        croppedfaces.append(
            cv2.resize(
                crop_face(frame, None, bbox, crop_by_bbox=True, bbox_scale=1.3),
                dsize=image_size,
            ).transpose((2, 0, 1))
        )

    return croppedfaces


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
