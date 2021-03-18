import cv2
from typing import List
from icecream import ic
import numpy as np


def get_names_of_classes(names_file_path: str) -> List[str]:
    with open(names_file_path, 'r') as f:
        names = f.read().rstrip('\n').split('\n')

    return names


def load_network(cfg_file_path: str, weightfile_path: str):
    net = cv2.dnn.readNetFromDarknet(cfg_file_path, weightfile_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def get_inferences(net: cv2.dnn_Net, image: str) -> List[int]:
    blob = cv2.dnn.blobFromImage(image, 1/255, (608, 608), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    out_names = _get_out_names(net)
    outs = net.forward(out_names)
    return outs


def parse_inferences(outs: List[int], net: cv2.dnn_Net, image_width: int, image_height: int) -> dict:
    class_ids = []
    confidences = []
    boxes = []
    last_layer_type = _get_last_layer_type(net)
    if last_layer_type == 'DetectionOutput':
        return last_layer_type
    elif last_layer_type == 'Region':
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > 0.5:
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)
                    width = int(detection[2] * image_width)
                    height = int(detection[3] * image_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(classId)
                    confidences.append(round(float(confidence), 3))
                    boxes.append([left, top, width, height])
    else:
        raise Exception('Unknown layer type.')

    # Remove overlapping boxes with smaller confidence
    indices = apply_non_max_supression(net, class_ids, boxes, confidences)

    return indices, class_ids, boxes, confidences


def apply_non_max_supression(net: cv2.dnn_Net, class_ids: List[int], boxes: List[List[int]], confidences: List[float]):
    indices = []
    class_ids = np.array(class_ids)
    boxes = np.array(boxes)
    confidences = np.array(confidences)
    unique_classes = set(class_ids)
    if len(_get_out_names(net)) < 2:
        return np.arange(0, len(class_ids))

    for cl in unique_classes:
        class_indices = np.where(class_ids == cl)[0]
        conf = confidences[class_indices]
        box = boxes[class_indices].tolist()
        nms_indices = cv2.dnn.NMSBoxes(box, conf, 0.5, 0.4)
        nms_indices = nms_indices[:, 0] if len(nms_indices) else []
        indices.extend(class_indices[nms_indices])

    return indices


def _read_image(image_path: str) -> List[int]:
    return cv2.imread(image_path)


def _get_out_names(net: cv2.dnn_Net):
    return net.getUnconnectedOutLayersNames()


def _get_last_layer_type(net: cv2.dnn_Net):
    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)
    return lastLayer.type
